# %% [markdown]
# # phase1_cnn_encoder.py
#
# Phase 1 pipeline with pretrained 1D CNN encoder embeddings.
# Paths are configured for this repository layout:
#   open/*.csv -> input data
#   processed/*.csv -> imputed data for 1D CNN encoder input
#   best_model_1dcnn.pt -> pretrained CNN weights; only encode() is used
#   phase2_artifacts/*.npy -> Phase 1 + CNN artifacts consumed by Phase 2
#   submission_phase1_cnn_encoder.csv -> Phase 1 submission

# %% [markdown]
# # Phase 1: Target Encoding + Layout Clustering + Pseudo-labeling## 목적v6 (OOF 8.61, 제출 10.00, 84등) 위에 세 가지 새 시도를 순차적으로 추가:1. **Target Encoding**: layout_id, layout_type을 target 기반 수치로 변환2. **Layout Clustering**: 신규 layout 일반화 강화3. **Pseudo-labeling**: Test의 신뢰 높은 예측을 학습 데이터로 추가## 사전 조건 (필수 파일)- `train.csv`, `test.csv`, `layout_info.csv`, `sample_submission.csv`- 기존 .npy 파일들 (있으면 로드, 없으면 다시 학습):  - `oof_lgb.npy`, `test_lgb.npy`  - `oof_cb_avg.npy`, `test_cb_avg.npy`  - `oof_xgb.npy`, `test_xgb.npy`## 실행 흐름```Step 0: 데이터 로드 + 기존 결과 로드 또는 학습Step 1: Target Encoding 적용 → CatBoost 재학습 → StackingStep 2: Layout Clustering 추가 → CatBoost 재학습 → StackingStep 3: Pseudo-labeling → CatBoost 재학습 → StackingStep 4: 최종 비교 + 최고 결과 제출```## 예상 시간- Step 0 (데이터 로드): 5분- Step 1 (TE): 1시간- Step 2 (Cluster): 1시간- Step 3 (Pseudo): 1.5시간- Step 4 (Stacking + 제출): 30분- **총: 4시간**## 예상 결과- v6: 8.61 → 새 best: 8.40~8.55- 제출 예상: 9.79~9.94- 등수: 84 → 30~50등 가능

# %% [markdown]
# ## Step 0: 환경 설정 + 데이터 로드

# %%
import pandas as pd
import numpy as np
import gc
import os
import psutil
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def mem_check(label=""):
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    avail_gb = psutil.virtual_memory().available / 1024**3
    print(f"  [{label}] 사용: {mem_mb:.0f}MB | 가용: {avail_gb:.2f}GB")

def find_project_root():
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())
    for start in candidates:
        for path in [start, *start.parents]:
            if (path / 'open' / 'train.csv').exists():
                return path
    return candidates[0]

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / 'open'
PROCESSED_DIR = PROJECT_ROOT / 'processed'
ARTIFACT_DIR = PROJECT_ROOT / 'phase2_artifacts'
SUBMISSION_PATH = PROJECT_ROOT / 'submission_phase1_cnn_encoder.csv'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
LAYOUT_PATH = DATA_DIR / 'layout_info.csv'
SAMPLE_SUBMISSION_PATH = DATA_DIR / 'sample_submission.csv'
TRAIN_IMPUTED_PATH = PROCESSED_DIR / 'train_imputed.csv'
TEST_IMPUTED_PATH = PROCESSED_DIR / 'test_imputed.csv'
CNN_MODEL_PATH = PROJECT_ROOT / 'best_model_1dcnn.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SLOTS = 25

LAYOUT_TYPE_MAP = {'narrow': 0, 'grid': 1, 'hybrid': 2, 'hub_spoke': 3}
LAYOUT_NUM_COLS = [
    'aisle_width_avg', 'intersection_count', 'one_way_ratio',
    'pack_station_count', 'charger_count', 'layout_compactness',
    'zone_dispersion', 'robot_total', 'building_age_years',
    'floor_area_sqm', 'ceiling_height_m', 'fire_sprinkler_count',
    'emergency_exit_count',
]
GROUPS = {
    'A': ['order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order',
          'urgent_order_ratio', 'heavy_item_ratio', 'cold_chain_ratio',
          'sku_concentration', 'return_order_ratio', 'bulk_order_ratio',
          'order_wave_count', 'pick_list_length_avg', 'express_lane_util'],
    'B': ['robot_active', 'robot_idle', 'robot_charging', 'robot_utilization',
          'avg_trip_distance', 'task_reassign_15m', 'avg_idle_duration_min',
          'agv_task_success_rate', 'path_optimization_score',
          'fleet_age_months_avg', 'robot_firmware_update_days', 'robot_calibration_score'],
    'C': ['battery_mean', 'battery_std', 'low_battery_ratio',
          'charge_queue_length', 'avg_charge_wait', 'charge_efficiency_pct',
          'battery_cycle_count_avg'],
    'D': ['congestion_score', 'max_zone_density', 'blocked_path_15m',
          'near_collision_15m', 'aisle_traffic_score', 'intersection_wait_time_avg'],
    'E': ['pack_utilization', 'replenishment_overlap', 'staging_area_util',
          'pallet_wrap_time_min', 'loading_dock_util', 'outbound_truck_wait_min',
          'sort_accuracy_pct', 'quality_check_rate', 'packaging_material_cost'],
    'F': ['fault_count_15m', 'avg_recovery_time', 'manual_override_ratio'],
    'G': ['staff_on_floor', 'forklift_active_count', 'worker_avg_tenure_months',
          'safety_score_monthly', 'shift_handover_delay_min'],
    'H': ['wms_response_time_ms', 'network_latency_ms', 'wifi_signal_db',
          'scanner_error_rate', 'barcode_read_success_rate', 'label_print_queue',
          'ups_battery_pct', 'daily_forecast_accuracy', 'inventory_turnover_rate'],
    'I': ['warehouse_temp_avg', 'humidity_pct', 'co2_level_ppm', 'air_quality_idx',
          'hvac_power_kw', 'external_temp_c', 'wind_speed_kmh', 'precipitation_mm',
          'lighting_level_lux', 'lighting_zone_variance', 'ambient_noise_db',
          'floor_vibration_idx', 'cold_storage_temp_c', 'zone_temp_variance'],
    'J': ['storage_density_pct', 'vertical_utilization', 'racking_height_avg_m',
          'cross_dock_ratio', 'shift_hour', 'day_of_week', 'prev_shift_volume',
          'kpi_otd_pct', 'backorder_ratio', 'dock_to_stock_hours',
          'maintenance_schedule_score', 'conveyor_speed_mps', 'avg_package_weight_kg'],
}

def find_input_artifact(filename):
    """Reuse existing artifacts from phase2_artifacts first, then project root."""
    candidates = [ARTIFACT_DIR / filename, PROJECT_ROOT / filename]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

def save_artifact(filename, array):
    path = ARTIFACT_DIR / filename
    np.save(path, array)
    return path

class CNN1DBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + x)

class CNN1DModel(nn.Module):
    def __init__(self, n_features, d_model=32, n_blocks=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([CNN1DBlock(d_model, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
            nn.Softplus(),
        )

    @torch.no_grad()
    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x.permute(0, 2, 1)

def build_features_cnn(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df = df.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    df['slot_idx'] = df.groupby('scenario_id').cumcount().astype(np.float32)
    df['layout_type_enc'] = df['layout_type'].map(LAYOUT_TYPE_MAP).astype(np.float32)

    df['pack_saturated'] = (df['pack_utilization'] > 0.95).astype(np.float32)
    df['pack_idle_anomaly'] = ((df['pack_utilization'] < 0.1) & (df['slot_idx'] > 5)).astype(np.float32)
    df['low_battery_alert'] = (df['low_battery_ratio'] > 0.034).astype(np.float32)
    df['battery_critical'] = (df['battery_mean'] < 58).astype(np.float32)
    df['density_alert'] = (df['max_zone_density'] > 0.05).astype(np.float32)

    df['has_collision'] = (df['near_collision_15m'] > 0).astype(np.float32)
    df['has_block'] = (df['blocked_path_15m'] > 0).astype(np.float32)
    df['has_fault'] = (df['fault_count_15m'] > 0).astype(np.float32)
    df['has_reassign'] = (df['task_reassign_15m'] > 0).astype(np.float32)

    total_fleet = df['robot_active'] + df['robot_idle'] + df['robot_charging']
    df['idle_ratio'] = df['robot_idle'] / (total_fleet + 1e-6)
    df['charging_ratio'] = df['robot_charging'] / (total_fleet + 1e-6)
    df['active_ratio'] = df['robot_active'] / (total_fleet + 1e-6)

    df['pack_x_slot'] = df['pack_utilization'] * df['slot_idx']
    df['order_x_slot'] = df['order_inflow_15m'] * df['slot_idx']
    df['inflow_x_urgent'] = df['order_inflow_15m'] * df['urgent_order_ratio']
    df['effective_load'] = df['order_inflow_15m'] * (1 + 0.5 * df['urgent_order_ratio'])
    df['pack_x_battery'] = df['pack_utilization'] * df['low_battery_ratio']
    df['pack_x_charging'] = df['pack_utilization'] * df['charging_ratio']
    df['inflow_x_battery'] = df['order_inflow_15m'] * df['low_battery_ratio']
    df['cong_x_inflow'] = df['congestion_score'] * df['order_inflow_15m']

    df['cum_orders'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum()
    df['cum_collisions'] = df.groupby('scenario_id')['near_collision_15m'].cumsum()
    df['cum_faults'] = df.groupby('scenario_id')['fault_count_15m'].cumsum()
    df['cum_unique_sku'] = df.groupby('scenario_id')['unique_sku_15m'].cumsum()

    for col in ['battery_mean', 'low_battery_ratio', 'congestion_score',
                'pack_utilization', 'order_inflow_15m']:
        df[f'{col}_isna'] = df[col].isna().astype(np.float32)

    df['robots_per_pack'] = df['robot_total'] / (df['pack_station_count'] + 1e-6)
    df['robots_per_charger'] = df['robot_total'] / (df['charger_count'] + 1e-6)
    df['area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1e-6)
    df['pack_per_charger'] = df['pack_station_count'] / (df['charger_count'] + 1e-6)
    df['is_narrow'] = (df['layout_type'] == 'narrow').astype(np.float32)
    df['narrow_x_battery'] = df['is_narrow'] * df['low_battery_ratio']
    df['few_packs'] = (df['pack_station_count'] < 7).astype(np.float32)
    df['few_packs_x_pack_util'] = df['few_packs'] * df['pack_utilization']
    return df

def get_cnn_feature_cols():
    all_group_cols = [c for cols in GROUPS.values() for c in cols]
    engineered = [
        'slot_idx', 'layout_type_enc',
        'pack_saturated', 'pack_idle_anomaly', 'low_battery_alert',
        'battery_critical', 'density_alert',
        'has_collision', 'has_block', 'has_fault', 'has_reassign',
        'idle_ratio', 'charging_ratio', 'active_ratio',
        'pack_x_slot', 'order_x_slot', 'inflow_x_urgent', 'effective_load',
        'pack_x_battery', 'pack_x_charging', 'inflow_x_battery', 'cong_x_inflow',
        'cum_orders', 'cum_collisions', 'cum_faults', 'cum_unique_sku',
        'battery_mean_isna', 'low_battery_ratio_isna', 'congestion_score_isna',
        'pack_utilization_isna', 'order_inflow_15m_isna',
        'robots_per_pack', 'robots_per_charger', 'area_per_robot', 'pack_per_charger',
        'is_narrow', 'narrow_x_battery', 'few_packs', 'few_packs_x_pack_util',
    ]
    return all_group_cols + LAYOUT_NUM_COLS + engineered

def to_cnn_scenarios(df, feature_cols, scaler=None, fit_scaler=False):
    n_rows = len(df)
    assert n_rows % SLOTS == 0, f'행 수({n_rows})가 {SLOTS}의 배수가 아님'
    raw = df[feature_cols].values.astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0)
    if fit_scaler:
        scaler = StandardScaler()
        raw = scaler.fit_transform(raw)
    elif scaler is not None:
        raw = scaler.transform(raw)
    return raw.reshape(n_rows // SLOTS, SLOTS, len(feature_cols)), scaler

@torch.no_grad()
def extract_cnn_embeddings(model, X_scenarios, batch_size=256):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_scenarios, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    embs = []
    for (x_batch,) in loader:
        emb = model.encode(x_batch.to(DEVICE)).cpu().numpy()
        embs.append(emb)
    embs = np.concatenate(embs, axis=0)
    return embs.reshape(-1, embs.shape[2])

def load_cnn_encoder_embeddings(train_imp, test_imp, layout_df):
    train_cnn = build_features_cnn(train_imp.copy(), layout_df)
    test_cnn = build_features_cnn(test_imp.copy(), layout_df)
    feature_cols = [c for c in get_cnn_feature_cols() if c in train_cnn.columns]
    X_train_cnn, scaler = to_cnn_scenarios(train_cnn, feature_cols, fit_scaler=True)
    X_test_cnn, _ = to_cnn_scenarios(test_cnn, feature_cols, scaler=scaler)

    ckpt = torch.load(CNN_MODEL_PATH, map_location=DEVICE)
    d_model = ckpt['proj.0.weight'].shape[0]
    n_features = ckpt['proj.0.weight'].shape[1]
    n_blocks = sum(1 for k in ckpt if k.startswith('blocks.') and k.endswith('.net.0.weight'))
    if n_features != len(feature_cols):
        raise ValueError(f'CNN 입력 피처 수 불일치: checkpoint {n_features}, current {len(feature_cols)}')

    model = CNN1DModel(n_features=n_features, d_model=d_model, n_blocks=n_blocks).to(DEVICE)
    model.load_state_dict(ckpt)
    print(f'  encoder: d_model={d_model}, n_blocks={n_blocks}, n_features={n_features}, device={DEVICE}')

    train_emb = extract_cnn_embeddings(model, X_train_cnn)
    test_emb = extract_cnn_embeddings(model, X_test_cnn)
    emb_cols = [f'cnn_emb_{i}' for i in range(d_model)]
    return pd.DataFrame(train_emb, columns=emb_cols), pd.DataFrame(test_emb, columns=emb_cols)

def add_cnn_encoder_embeddings(train_df, test_df, layout_df):
    """Use best_model_1dcnn.pt as an encoder and concat slot-level embeddings."""
    if not CNN_MODEL_PATH.exists():
        print(f"⚠ CNN 가중치 없음: {CNN_MODEL_PATH} - CNN 임베딩 없이 진행")
        return train_df, test_df, []
    if not TRAIN_IMPUTED_PATH.exists() or not TEST_IMPUTED_PATH.exists():
        print(f"⚠ imputed 데이터 없음: {PROCESSED_DIR} - CNN 임베딩 없이 진행")
        return train_df, test_df, []

    print("\n=== 1D CNN Encoder 임베딩 추출 ===")
    print(f"  weights: {CNN_MODEL_PATH}")
    train_imp = pd.read_csv(TRAIN_IMPUTED_PATH)
    test_imp = pd.read_csv(TEST_IMPUTED_PATH)
    train_emb_df, test_emb_df = load_cnn_encoder_embeddings(train_imp, test_imp, layout_df)

    emb_cols = train_emb_df.columns.tolist()
    train_emb_df = train_emb_df.reset_index(drop=True).astype('float32')
    test_emb_df = test_emb_df.reset_index(drop=True).astype('float32')

    if len(train_emb_df) != len(train_df) or len(test_emb_df) != len(test_df):
        raise ValueError(
            f"CNN embedding 길이 불일치: train {len(train_emb_df)} vs {len(train_df)}, "
            f"test {len(test_emb_df)} vs {len(test_df)}"
        )

    train_df = pd.concat([train_df.reset_index(drop=True), train_emb_df], axis=1)
    test_df = pd.concat([test_df.reset_index(drop=True), test_emb_df], axis=1)
    save_artifact('train_emb_flat_1dcnn.npy', train_emb_df.values)
    save_artifact('test_emb_flat_1dcnn.npy', test_emb_df.values)
    print(f"  CNN embedding 추가: {len(emb_cols)}차원")
    return train_df, test_df, emb_cols

print("=== 시작 ===")
mem_check("초기")
print(f"CatBoost: {HAS_CATBOOST}, XGBoost: {HAS_XGBOOST}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"PROCESSED_DIR: {PROCESSED_DIR}")
print(f"ARTIFACT_DIR: {ARTIFACT_DIR}")

# %%
# === 데이터 로드 (메모리 효율) ===
sample = pd.read_csv(TRAIN_PATH, nrows=100)
numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
dtypes = {col: 'float32' for col in numeric_cols}
dtypes['ID'] = 'object'
dtypes['scenario_id'] = 'object'
dtypes['layout_id'] = 'object'

print("로딩...")
train = pd.read_csv(TRAIN_PATH, dtype=dtypes)
test = pd.read_csv(TEST_PATH, dtype=dtypes)
layout = pd.read_csv(LAYOUT_PATH)
for col in layout.columns:
    if layout[col].dtype == 'float64':
        layout[col] = layout[col].astype('float32')
layout_for_cnn = layout.copy()

target_col = 'avg_delay_minutes_next_30m'

# Layout join
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')
del layout
gc.collect()

# slot_idx
train = train.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
test = test.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
train['slot_idx'] = train.groupby('scenario_id').cumcount().astype('int8')
test['slot_idx'] = test.groupby('scenario_id').cumcount().astype('int8')

print(f"train: {train.shape}, test: {test.shape}")
mem_check("데이터 로드 후")

# Layout 분석
train_layouts = set(train['layout_id'].unique())
test_layouts = set(test['layout_id'].unique())
unseen_layouts = test_layouts - train_layouts
print(f"\n신규 layout (test에만 있음): {len(unseen_layouts)}개")

# %% [markdown]
# ## 핵심 피처 빌드 (간단 버전)

# %%
def build_features(df):
    """검증된 핵심 피처"""
    
    # === 비율 ===
    df['total_fleet'] = (df['robot_active'] + df['robot_idle'] + df['robot_charging']).astype('float32')
    df['idle_ratio'] = (df['robot_idle'] / (df['total_fleet'] + 1e-6)).astype('float32')
    df['charging_ratio'] = (df['robot_charging'] / (df['total_fleet'] + 1e-6)).astype('float32')
    df['active_ratio'] = (df['robot_active'] / (df['total_fleet'] + 1e-6)).astype('float32')
    
    # === Binary ===
    df['has_collision'] = (df['near_collision_15m'] > 0).astype('int8')
    df['has_fault'] = (df['fault_count_15m'] > 0).astype('int8')
    df['pack_saturated'] = (df['pack_utilization'] > 0.95).astype('int8')
    
    # === 상호작용 ===
    df['pack_x_slot'] = (df['pack_utilization'] * df['slot_idx']).astype('float32')
    df['pack_x_battery'] = (df['pack_utilization'] * df['low_battery_ratio']).astype('float32')
    df['pack_x_charging'] = (df['pack_utilization'] * df['charging_ratio']).astype('float32')
    df['cong_x_inflow'] = (df['congestion_score'] * df['order_inflow_15m']).astype('float32')
    df['effective_load'] = (df['order_inflow_15m'] * (1 + 0.5 * df['urgent_order_ratio'])).astype('float32')
    
    # === 누적 ===
    df['cum_orders'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum().astype('float32')
    df['cum_faults'] = df.groupby('scenario_id')['fault_count_15m'].cumsum().astype('float32')
    
    # === Lag ===
    for f in ['order_inflow_15m', 'pack_utilization', 'low_battery_ratio']:
        df[f'{f}_lag1'] = df.groupby('scenario_id')[f].shift(1).astype('float32')
    
    # === Layout 비율 ===
    df['robots_per_pack'] = (df['robot_total'] / (df['pack_station_count'] + 1)).astype('float32')
    df['robots_per_charger'] = (df['robot_total'] / (df['charger_count'] + 1)).astype('float32')
    df['area_per_robot'] = (df['floor_area_sqm'] / (df['robot_total'] + 1)).astype('float32')
    
    # === 메커니즘 (효과 입증) ===
    df['utilization_imbalance'] = (df['pack_utilization'] - df['idle_ratio']).astype('float32')
    df['max_utilization'] = df[['pack_utilization', 'charging_ratio']].max(axis=1).astype('float32')
    
    # === 결측 플래그 ===
    for col in ['battery_mean', 'low_battery_ratio', 'pack_utilization', 'congestion_score']:
        df[f'{col}_isna'] = df[col].isna().astype('int8')
    
    return df

print("Train 피처...")
train = build_features(train)
print("Test 피처...")
test = build_features(test)

# 시나리오 메타
key_features = ['order_inflow_15m', 'pack_utilization', 'low_battery_ratio',
                'congestion_score', 'idle_ratio', 'charging_ratio']

for f in key_features:
    sc_agg = train.groupby('scenario_id')[f].agg(['mean', 'std', 'max', 'min'])
    sc_agg.columns = [f'{f}_sc_{stat}' for stat in sc_agg.columns]
    for col in sc_agg.columns:
        sc_agg[col] = sc_agg[col].astype('float32')
    train = train.merge(sc_agg, on='scenario_id', how='left')
    
    sc_agg_test = test.groupby('scenario_id')[f].agg(['mean', 'std', 'max', 'min'])
    sc_agg_test.columns = [f'{f}_sc_{stat}' for stat in sc_agg_test.columns]
    for col in sc_agg_test.columns:
        sc_agg_test[col] = sc_agg_test[col].astype('float32')
    test = test.merge(sc_agg_test, on='scenario_id', how='left')

print(f"피처 후: train {train.shape}, test {test.shape}")
mem_check("피처 후")
gc.collect()

# === 1D CNN pretrained encoder embedding concat ===
# Boosting 모델은 원본 open/*.csv 기반 Phase1 피처를 그대로 쓰고,
# CNN encoder 입력만 학습 당시와 맞추기 위해 processed/*_imputed.csv를 사용합니다.
train, test, cnn_emb_cols = add_cnn_encoder_embeddings(train, test, layout_for_cnn)
print(f"CNN 임베딩 concat 후: train {train.shape}, test {test.shape}")
mem_check("CNN 임베딩 후")
del layout_for_cnn
gc.collect()

# %% [markdown]
# ## 학습 준비 + 기존 결과 .npy 로드 (있으면)

# %%
# === 학습 준비 ===
common_cols = [c for c in train.columns if c in test.columns]
meta_cols = ['ID', 'scenario_id', target_col]
feature_cols = [c for c in common_cols if c not in meta_cols]
print(f"피처 수 (TE, Cluster 추가 전): {len(feature_cols)}")

train['layout_id'] = train['layout_id'].astype('category')
train['layout_type'] = train['layout_type'].astype('category')
test['layout_id'] = test['layout_id'].astype('category')
test['layout_type'] = test['layout_type'].astype('category')

X_train = train[feature_cols]
y_train_arr = train[target_col].values.astype('float32')
y_log = np.log1p(y_train_arr).astype('float32')
groups = train['scenario_id'].values
X_test = test[feature_cols]

gkf = GroupKFold(n_splits=5)

# === 기존 결과 로드 (있으면) ===
print("\n기존 .npy 파일 확인...")

oof_files = {
    'lgb': ('oof_lgb_cnnenc.npy', 'test_lgb_cnnenc.npy'),
    'cb': ('oof_cb_avg_cnnenc.npy', 'test_cb_avg_cnnenc.npy'),
    'xgb': ('oof_xgb_cnnenc.npy', 'test_xgb_cnnenc.npy'),
}

base_predictions = {}
for name, (oof_file, test_file) in oof_files.items():
    oof_path = find_input_artifact(oof_file)
    test_path = find_input_artifact(test_file)
    if oof_path.exists() and test_path.exists():
        oof = np.load(oof_path).astype('float32')
        test_pred = np.load(test_path).astype('float32')
        if len(oof) == len(X_train) and len(test_pred) == len(X_test):
            base_predictions[name] = (oof, test_pred)
            mae = mean_absolute_error(y_train_arr, oof)
            print(f"  {name}: 로드 성공, OOF MAE = {mae:.4f} ({oof_path.name}, {test_path.name})")
        else:
            print(f"  {name}: 길이 불일치 - 무시")
    else:
        if name == 'xgb' and not HAS_XGBOOST:
            print(f"  {name}: 파일 없음 - XGBoost 미설치로 스킵 ({oof_path}, {test_path})")
        else:
            print(f"  {name}: 파일 없음 - 새로 학습 필요 ({oof_path}, {test_path})")

print(f"\n로드된 base 모델: {list(base_predictions.keys())}")

# %% [markdown]
# ## (선택) 기존 결과가 없으면 Base 모델 다시 학습이미 .npy 있으면 스킵하세요.

# %%
# === CatBoost 학습 (oof_cb_avg.npy 없으면) ===
if 'cb' not in base_predictions:
    print("CatBoost 학습 (메인 모델)")
    
    cb_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.05,
        'depth': 8,
        'iterations': 3000,
        'cat_features': ['layout_id', 'layout_type'],
        'early_stopping_rounds': 100,
        'verbose': 0,
        'thread_count': -1,
    }
    
    seeds = [42, 0, 7]
    oof_cb_sum = np.zeros(len(X_train), dtype='float32')
    test_cb_sum = np.zeros(len(X_test), dtype='float32')
    
    for seed in seeds:
        print(f"  Seed {seed}")
        cb_p = {**cb_params, 'random_seed': seed}
        oof_log = np.zeros(len(X_train), dtype='float32')
        test_log = np.zeros(len(X_test), dtype='float32')
        
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_log, groups=groups)):
            X_tr = X_train.iloc[tr_idx].copy()
            X_val = X_train.iloc[val_idx].copy()
            for col in ['layout_id', 'layout_type']:
                X_tr[col] = X_tr[col].astype(str)
                X_val[col] = X_val[col].astype(str)
            
            model = CatBoostRegressor(**cb_p)
            model.fit(X_tr, y_log[tr_idx], eval_set=(X_val, y_log[val_idx]),
                      use_best_model=True, verbose=0)
            oof_log[val_idx] = model.predict(X_val).astype('float32')
            
            X_test_cb = X_test.copy()
            for col in ['layout_id', 'layout_type']:
                X_test_cb[col] = X_test_cb[col].astype(str)
            test_log += (model.predict(X_test_cb) / 5).astype('float32')
            
            del model, X_tr, X_val, X_test_cb
            gc.collect()
        
        oof_cb_sum += np.clip(np.expm1(oof_log), 0, None) / 3
        test_cb_sum += np.clip(np.expm1(test_log), 0, None) / 3
    
    save_artifact('oof_cb_avg_cnnenc.npy', oof_cb_sum)
    save_artifact('test_cb_avg_cnnenc.npy', test_cb_sum)
    base_predictions['cb'] = (oof_cb_sum, test_cb_sum)
    print(f"  CatBoost MAE: {mean_absolute_error(y_train_arr, oof_cb_sum):.4f}")

# === LightGBM 학습 (없으면) ===
if 'lgb' not in base_predictions:
    print("\nLightGBM 학습")
    params_lgb = {
        'objective': 'regression_l1', 'metric': 'mae',
        'learning_rate': 0.05, 'num_leaves': 63,
        'min_child_samples': 100, 'feature_fraction': 0.7,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'random_state': 42, 'verbose': -1, 'n_jobs': -1,
    }
    
    oof_lgb_log = np.zeros(len(X_train), dtype='float32')
    test_lgb_log = np.zeros(len(X_test), dtype='float32')
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_log, groups=groups)):
        train_data = lgb.Dataset(X_train.iloc[tr_idx], label=y_log[tr_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_log[val_idx], reference=train_data)
        model = lgb.train(params_lgb, train_data, num_boost_round=3000,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_lgb_log[val_idx] = model.predict(X_train.iloc[val_idx]).astype('float32')
        test_lgb_log += (model.predict(X_test) / 5).astype('float32')
        del model, train_data, val_data
        gc.collect()
    
    oof_lgb = np.clip(np.expm1(oof_lgb_log), 0, None)
    test_lgb = np.clip(np.expm1(test_lgb_log), 0, None)
    save_artifact('oof_lgb_cnnenc.npy', oof_lgb)
    save_artifact('test_lgb_cnnenc.npy', test_lgb)
    base_predictions['lgb'] = (oof_lgb, test_lgb)
    print(f"  LightGBM MAE: {mean_absolute_error(y_train_arr, oof_lgb):.4f}")

# === XGBoost 학습 (없으면) ===
if 'xgb' not in base_predictions and HAS_XGBOOST:
    print("\nXGBoost 학습")

    def prepare_xgb_features(train_df, test_df):
        train_df = train_df.copy()
        test_df = test_df.copy()
        for col in train_df.columns:
            if str(train_df[col].dtype) == 'category' or train_df[col].dtype == object:
                combined = pd.concat([train_df[col], test_df[col]], ignore_index=True).astype('category')
                train_df[col] = combined.iloc[:len(train_df)].cat.codes.astype('int32').values
                test_df[col] = combined.iloc[len(train_df):].cat.codes.astype('int32').values
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        return train_df, test_df

    X_train_xgb, X_test_xgb = prepare_xgb_features(X_train, X_test)

    xgb_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'learning_rate': 0.05,
        'max_depth': 7,
        'n_estimators': 3000,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 100,
    }

    oof_xgb_log = np.zeros(len(X_train_xgb), dtype='float32')
    test_xgb_log = np.zeros(len(X_test_xgb), dtype='float32')

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_xgb, y_log, groups=groups)):
        print(f"  XGBoost Fold {fold+1}/5")
        model = XGBRegressor(**{**xgb_params, 'random_state': 42 + fold})
        model.fit(
            X_train_xgb.iloc[tr_idx], y_log[tr_idx],
            eval_set=[(X_train_xgb.iloc[val_idx], y_log[val_idx])],
            verbose=False,
        )
        oof_xgb_log[val_idx] = model.predict(X_train_xgb.iloc[val_idx]).astype('float32')
        test_xgb_log += (model.predict(X_test_xgb) / 5).astype('float32')
        del model
        gc.collect()

    oof_xgb = np.clip(np.expm1(oof_xgb_log), 0, None).astype('float32')
    test_xgb = np.clip(np.expm1(test_xgb_log), 0, None).astype('float32')
    save_artifact('oof_xgb_cnnenc.npy', oof_xgb)
    save_artifact('test_xgb_cnnenc.npy', test_xgb)
    base_predictions['xgb'] = (oof_xgb, test_xgb)
    print(f"  XGBoost MAE: {mean_absolute_error(y_train_arr, oof_xgb):.4f}")
    del X_train_xgb, X_test_xgb
    gc.collect()
elif 'xgb' not in base_predictions:
    print("\nXGBoost 미설치 - xgb base 모델 스킵")

mem_check("Base 모델 후")

# %% [markdown]
# ## Step 1: Target Encoding (★ 첫 번째 시도)KFold 안전 방식으로 target encoding 적용.layout_id, layout_type, scenario 메타에 적용.

# %%
# === KFold-safe Target Encoding ===

def kfold_target_encoding(df_train, df_test, group_col, target_col_arr, 
                            groups_arr, n_splits=5, smoothing=10):
    """K-fold로 안전한 target encoding"""
    
    oof_te = np.zeros(len(df_train), dtype='float32')
    gkf_te = GroupKFold(n_splits=n_splits)
    
    for tr_idx, val_idx in gkf_te.split(df_train, target_col_arr, groups=groups_arr):
        # 각 그룹의 평균 + smoothing
        tr_target = pd.Series(target_col_arr[tr_idx])
        tr_group = df_train.iloc[tr_idx][group_col].astype(str)
        global_mean = tr_target.mean()
        
        agg = pd.DataFrame({'g': tr_group.values, 't': tr_target.values}).groupby('g')['t'].agg(['mean', 'count'])
        agg['te'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
        
        te_map = agg['te'].to_dict()
        val_groups = df_train.iloc[val_idx][group_col].astype(str).values
        oof_te[val_idx] = pd.Series(val_groups).map(te_map).fillna(global_mean).values.astype('float32')
    
    # Test (전체 train 기준)
    global_mean = float(target_col_arr.mean())
    full_groups = df_train[group_col].astype(str)
    full_target = pd.Series(target_col_arr)
    agg = pd.DataFrame({'g': full_groups.values, 't': full_target.values}).groupby('g')['t'].agg(['mean', 'count'])
    agg['te'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    
    test_te = df_test[group_col].astype(str).map(agg['te'].to_dict()).fillna(global_mean).values.astype('float32')
    
    return oof_te, test_te

# === Target Encoding 적용 ===
print("Target Encoding 적용...")

te_results = {}
for col in ['layout_id', 'layout_type']:
    oof_te, test_te = kfold_target_encoding(
        train, test, col, y_train_arr, groups, n_splits=5, smoothing=20
    )
    train[f'{col}_te'] = oof_te
    test[f'{col}_te'] = test_te
    print(f"  {col}_te 추가")

mem_check("TE 후")

# %% [markdown]
# ### Step 1.1: TE 추가 후 CatBoost 재학습

# %%
# === TE 추가 후 새 X_train, X_test ===
common_cols = [c for c in train.columns if c in test.columns]
feature_cols_v1 = [c for c in common_cols if c not in meta_cols]
print(f"피처 수 (TE 추가): {len(feature_cols_v1)}")

X_train_v1 = train[feature_cols_v1]
X_test_v1 = test[feature_cols_v1]

# CatBoost 1 시드만 (시간 절약, 효과만 빠르게 확인)
cb_params_v1 = {
    'loss_function': 'MAE', 'eval_metric': 'MAE',
    'learning_rate': 0.05, 'depth': 8, 'iterations': 3000,
    'cat_features': ['layout_id', 'layout_type'],
    'early_stopping_rounds': 100, 'verbose': 0,
    'thread_count': -1, 'random_seed': 42,
}

oof_cb_v1_log = np.zeros(len(X_train_v1), dtype='float32')
test_cb_v1_log = np.zeros(len(X_test_v1), dtype='float32')

print("\n=== CatBoost (TE 추가) ===")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_v1, y_log, groups=groups)):
    print(f"  Fold {fold+1}/5")
    X_tr = X_train_v1.iloc[tr_idx].copy()
    X_val = X_train_v1.iloc[val_idx].copy()
    for col in ['layout_id', 'layout_type']:
        X_tr[col] = X_tr[col].astype(str)
        X_val[col] = X_val[col].astype(str)
    
    model = CatBoostRegressor(**cb_params_v1)
    model.fit(X_tr, y_log[tr_idx], eval_set=(X_val, y_log[val_idx]),
              use_best_model=True, verbose=0)
    oof_cb_v1_log[val_idx] = model.predict(X_val).astype('float32')
    
    X_test_cb = X_test_v1.copy()
    for col in ['layout_id', 'layout_type']:
        X_test_cb[col] = X_test_cb[col].astype(str)
    test_cb_v1_log += (model.predict(X_test_cb) / 5).astype('float32')
    
    del model, X_tr, X_val, X_test_cb
    gc.collect()

oof_cb_v1 = np.clip(np.expm1(oof_cb_v1_log), 0, None).astype('float32')
test_cb_v1 = np.clip(np.expm1(test_cb_v1_log), 0, None).astype('float32')

mae_cb_v1 = mean_absolute_error(y_train_arr, oof_cb_v1)
mae_cb_orig = mean_absolute_error(y_train_arr, base_predictions['cb'][0])

print(f"\n=== Step 1 결과 (TE 효과) ===")
print(f"  CatBoost (기존):     {mae_cb_orig:.4f}")
print(f"  CatBoost (TE 추가):  {mae_cb_v1:.4f}")
print(f"  변화: {mae_cb_v1 - mae_cb_orig:+.4f}")

# 즉시 저장
save_artifact('oof_cb_te_cnnenc.npy', oof_cb_v1)
save_artifact('test_cb_te_cnnenc.npy', test_cb_v1)
mem_check("Step 1 완료")

# %% [markdown]
# ## Step 2: Layout Clustering (★ 두 번째 시도)신규 layout 일반화 강화. Layout 속성으로 KMeans 클러스터링.

# %%
# === Layout Clustering ===
print("Layout Clustering 적용...")

# Layout 속성 (numeric만)
layout_attrs = ['pack_station_count', 'charger_count', 'floor_area_sqm',
                'aisle_width_avg', 'intersection_count', 'one_way_ratio',
                'ceiling_height_m', 'building_age_years',
                'layout_compactness', 'zone_dispersion',
                'fire_sprinkler_count', 'emergency_exit_count']
layout_attrs = [c for c in layout_attrs if c in train.columns]
print(f"  Layout 속성: {len(layout_attrs)}개")

# 모든 layout 수집 (train + test)
all_layouts_df = pd.concat([
    train[['layout_id'] + layout_attrs].drop_duplicates(),
    test[['layout_id'] + layout_attrs].drop_duplicates()
]).drop_duplicates('layout_id').reset_index(drop=True)
print(f"  총 unique layout: {len(all_layouts_df)}")

# 표준화
scaler_l = StandardScaler()
X_layout = scaler_l.fit_transform(all_layouts_df[layout_attrs].fillna(0))

# 여러 K로 클러스터링
for n_clusters in [5, 10, 20]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_col = f'layout_cluster_{n_clusters}'
    all_layouts_df[cluster_col] = kmeans.fit_predict(X_layout).astype('int8')
    
    cluster_map = dict(zip(all_layouts_df['layout_id'], all_layouts_df[cluster_col]))
    train[cluster_col] = train['layout_id'].astype(str).map(cluster_map).astype('int8')
    test[cluster_col] = test['layout_id'].astype(str).map(cluster_map).astype('int8')

# 클러스터에도 target encoding
for n_clusters in [5, 10, 20]:
    cluster_col = f'layout_cluster_{n_clusters}'
    oof_te, test_te = kfold_target_encoding(
        train, test, cluster_col, y_train_arr, groups, n_splits=5, smoothing=30
    )
    train[f'{cluster_col}_te'] = oof_te
    test[f'{cluster_col}_te'] = test_te

print(f"\n클러스터 + TE 추가 완료")
mem_check("Cluster 후")

# %% [markdown]
# ### Step 2.1: Cluster 추가 후 CatBoost 재학습

# %%
# 새 피처 셋
common_cols = [c for c in train.columns if c in test.columns]
feature_cols_v2 = [c for c in common_cols if c not in meta_cols]
print(f"피처 수 (TE + Cluster): {len(feature_cols_v2)}")

X_train_v2 = train[feature_cols_v2]
X_test_v2 = test[feature_cols_v2]

oof_cb_v2_log = np.zeros(len(X_train_v2), dtype='float32')
test_cb_v2_log = np.zeros(len(X_test_v2), dtype='float32')

print("\n=== CatBoost (TE + Cluster) ===")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_v2, y_log, groups=groups)):
    print(f"  Fold {fold+1}/5")
    X_tr = X_train_v2.iloc[tr_idx].copy()
    X_val = X_train_v2.iloc[val_idx].copy()
    for col in ['layout_id', 'layout_type']:
        X_tr[col] = X_tr[col].astype(str)
        X_val[col] = X_val[col].astype(str)
    
    model = CatBoostRegressor(**cb_params_v1)
    model.fit(X_tr, y_log[tr_idx], eval_set=(X_val, y_log[val_idx]),
              use_best_model=True, verbose=0)
    oof_cb_v2_log[val_idx] = model.predict(X_val).astype('float32')
    
    X_test_cb = X_test_v2.copy()
    for col in ['layout_id', 'layout_type']:
        X_test_cb[col] = X_test_cb[col].astype(str)
    test_cb_v2_log += (model.predict(X_test_cb) / 5).astype('float32')
    
    del model, X_tr, X_val, X_test_cb
    gc.collect()

oof_cb_v2 = np.clip(np.expm1(oof_cb_v2_log), 0, None).astype('float32')
test_cb_v2 = np.clip(np.expm1(test_cb_v2_log), 0, None).astype('float32')

mae_cb_v2 = mean_absolute_error(y_train_arr, oof_cb_v2)

print(f"\n=== Step 2 결과 (TE + Cluster 효과) ===")
print(f"  CatBoost (기존):              {mae_cb_orig:.4f}")
print(f"  CatBoost (TE):                {mae_cb_v1:.4f}")
print(f"  CatBoost (TE + Cluster):      {mae_cb_v2:.4f}")
print(f"  Cluster 추가 변화: {mae_cb_v2 - mae_cb_v1:+.4f}")

save_artifact('oof_cb_te_cluster_cnnenc.npy', oof_cb_v2)
save_artifact('test_cb_te_cluster_cnnenc.npy', test_cb_v2)
mem_check("Step 2 완료")

# %% [markdown]
# ## Step 3: Pseudo-labeling (★ 세 번째 시도)현재 best 예측 중 신뢰 높은 test (5~40분 평균 영역)를 pseudo label로 학습 데이터에 추가.

# %%
# === 일단 현재 최고 모델로 test 예측 만들기 (단순 stacking) ===
print("현재까지의 best로 test 예측 만들기...")

# Step 2의 CatBoost가 가장 새로운 best
current_best_test = test_cb_v2
current_best_oof = oof_cb_v2

print(f"  현재 best OOF: {mean_absolute_error(y_train_arr, current_best_oof):.4f}")
print(f"  Test 예측 통계: mean={current_best_test.mean():.2f}, median={np.median(current_best_test):.2f}")

# === 신뢰 영역 선택 (5~40분, 모델이 잘 맞추는 영역) ===
confident_mask = (current_best_test >= 5) & (current_best_test <= 40)
print(f"\n신뢰 영역 (5~40분) test 행: {confident_mask.sum()}/{len(current_best_test)} ({confident_mask.mean()*100:.1f}%)")

# === Pseudo 데이터셋 구성 ===
X_pseudo = pd.concat([X_train_v2, X_test_v2[confident_mask]], ignore_index=True)
y_pseudo = np.concatenate([y_train_arr, current_best_test[confident_mask]]).astype('float32')
y_pseudo_log = np.log1p(y_pseudo).astype('float32')

# 시나리오 ID 합침 (group 유지)
pseudo_test_groups = test['scenario_id'].values[confident_mask]
groups_pseudo = np.concatenate([groups, pseudo_test_groups])

print(f"\nPseudo dataset: {len(X_pseudo)} (train {len(X_train_v2)} + test_pseudo {confident_mask.sum()})")
mem_check("Pseudo data 후")

# %% [markdown]
# ### Step 3.1: Pseudo data로 CatBoost 학습

# %%
# === Pseudo CatBoost ===
oof_cb_pseudo_log = np.zeros(len(X_train_v2), dtype='float32')
test_cb_pseudo_log = np.zeros(len(X_test_v2), dtype='float32')

print("=== CatBoost (Pseudo-labeling) ===")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_pseudo, y_pseudo_log, groups=groups_pseudo)):
    # val_idx에서 원본 train만 추출 (pseudo는 검증 안 함)
    val_idx_orig = val_idx[val_idx < len(X_train_v2)]
    if len(val_idx_orig) == 0: 
        continue
    
    print(f"  Fold {fold+1}/5 (train {len(tr_idx)}, val {len(val_idx_orig)})")
    
    X_tr = X_pseudo.iloc[tr_idx].copy()
    X_val = X_pseudo.iloc[val_idx_orig].copy()
    for col in ['layout_id', 'layout_type']:
        X_tr[col] = X_tr[col].astype(str)
        X_val[col] = X_val[col].astype(str)
    
    model = CatBoostRegressor(**cb_params_v1)
    model.fit(X_tr, y_pseudo_log[tr_idx], 
              eval_set=(X_val, y_pseudo_log[val_idx_orig]),
              use_best_model=True, verbose=0)
    
    oof_cb_pseudo_log[val_idx_orig] = model.predict(X_val).astype('float32')
    
    X_test_cb = X_test_v2.copy()
    for col in ['layout_id', 'layout_type']:
        X_test_cb[col] = X_test_cb[col].astype(str)
    test_cb_pseudo_log += (model.predict(X_test_cb) / 5).astype('float32')
    
    del model, X_tr, X_val, X_test_cb
    gc.collect()

oof_cb_pseudo = np.clip(np.expm1(oof_cb_pseudo_log), 0, None).astype('float32')
test_cb_pseudo = np.clip(np.expm1(test_cb_pseudo_log), 0, None).astype('float32')

mae_cb_pseudo = mean_absolute_error(y_train_arr, oof_cb_pseudo)

print(f"\n=== Step 3 결과 (Pseudo-labeling) ===")
print(f"  CatBoost (기존):              {mae_cb_orig:.4f}")
print(f"  CatBoost (TE):                {mae_cb_v1:.4f}")
print(f"  CatBoost (TE + Cluster):      {mae_cb_v2:.4f}")
print(f"  CatBoost (+ Pseudo):          {mae_cb_pseudo:.4f}")
print(f"  Pseudo 추가 변화: {mae_cb_pseudo - mae_cb_v2:+.4f}")

save_artifact('oof_cb_pseudo_cnnenc.npy', oof_cb_pseudo)
save_artifact('test_cb_pseudo_cnnenc.npy', test_cb_pseudo)
del X_pseudo
gc.collect()
mem_check("Step 3 완료")

# %% [markdown]
# ## Step 4: 최종 Stacking + 제출모든 단계 결과 + 기존 base 모델로 최종 stacking.

# %%
# === 최종 Stacking 입력 구성 ===
print("=== 모든 모델 결과 정리 ===")
print(f"기존 base 모델:")
for name, (oof, _) in base_predictions.items():
    print(f"  {name}: OOF MAE = {mean_absolute_error(y_train_arr, oof):.4f}")
print(f"\n새 시도:")
print(f"  CB+TE:           {mae_cb_v1:.4f}")
print(f"  CB+TE+Cluster:   {mae_cb_v2:.4f}")
print(f"  CB+Pseudo:       {mae_cb_pseudo:.4f}")

# Stacking 입력
meta_oof = []
meta_test = []
model_names_final = []

# 기존 base
for name, (oof, test_pred) in base_predictions.items():
    meta_oof.append(oof)
    meta_test.append(test_pred)
    model_names_final.append(name)

# 새 시도
meta_oof.extend([oof_cb_v1, oof_cb_v2, oof_cb_pseudo])
meta_test.extend([test_cb_v1, test_cb_v2, test_cb_pseudo])
model_names_final.extend(['cb_te', 'cb_te_cluster', 'cb_pseudo'])

X_meta_train_f = np.column_stack(meta_oof).astype('float32')
X_meta_test_f = np.column_stack(meta_test).astype('float32')

# 시나리오 메타 핵심
key_meta_cols = ['low_battery_ratio_sc_mean', 'pack_utilization_sc_mean',
                  'congestion_score_sc_mean', 'slot_idx', 'pack_x_battery']
key_meta_cols = [c for c in key_meta_cols if c in X_train.columns]

X_meta_train_f = np.column_stack([X_meta_train_f, X_train[key_meta_cols].fillna(0).values.astype('float32')])
X_meta_test_f = np.column_stack([X_meta_test_f, X_test[key_meta_cols].fillna(0).values.astype('float32')])

print(f"\n최종 메타 입력: {X_meta_train_f.shape}")
print(f"포함 모델: {model_names_final}")

# === 메타 학습 ===
meta_params = {
    'objective': 'regression_l1', 'metric': 'mae',
    'learning_rate': 0.03, 'num_leaves': 15,
    'min_child_samples': 200,
    'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'verbose': -1, 'random_state': 42,
}

stacking_final_oof = np.zeros(len(y_train_arr), dtype='float32')
stacking_final_test = np.zeros(len(X_test), dtype='float32')

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_meta_train_f, y_log, groups=groups)):
    print(f"  Stacking Fold {fold+1}/5")
    train_data = lgb.Dataset(X_meta_train_f[tr_idx], label=y_log[tr_idx])
    val_data = lgb.Dataset(X_meta_train_f[val_idx], label=y_log[val_idx], reference=train_data)
    meta_model = lgb.train(meta_params, train_data, num_boost_round=2000,
                            valid_sets=[val_data],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    stacking_final_oof[val_idx] = meta_model.predict(X_meta_train_f[val_idx]).astype('float32')
    stacking_final_test += (meta_model.predict(X_meta_test_f) / 5).astype('float32')
    del meta_model, train_data, val_data
    gc.collect()

stacking_final_raw = np.clip(np.expm1(stacking_final_oof), 0, None).astype('float32')
stacking_final_test_raw = np.clip(np.expm1(stacking_final_test), 0, None).astype('float32')

final_mae = mean_absolute_error(y_train_arr, stacking_final_raw)

print(f"\n=== 최종 Stacking 결과 ===")
print(f"  v6 (검증, 제출 10.00):        8.6293")
print(f"  Phase 1 최종:                  {final_mae:.4f}")
print(f"  v6 대비:                       {final_mae - 8.6293:+.4f}")
print(f"  갭 1.39 적용 시 제출 예상:    {final_mae + 1.39:.2f}")

mem_check("Stacking 완료")

# %% [markdown]
# ## 제출 파일 생성

# %%
# === 제출 파일 ===
if final_mae < 8.6293:
    print(f"개선됨! 제출 파일 생성")
    
    final_test_pred = np.clip(stacking_final_test_raw, 0, None)
    submission = pd.DataFrame({
        'ID': test['ID'].values,
        target_col: final_test_pred
    })
    
    if SAMPLE_SUBMISSION_PATH.exists():
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        submission = sample[['ID']].merge(submission, on='ID', how='left')
        print("  sample_submission 순서로 정렬됨")
    
    assert len(submission) == 50000
    assert submission[target_col].notna().all()
    assert (submission[target_col] >= 0).all()
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\n{SUBMISSION_PATH} 저장")
    print(f"  OOF MAE: {final_mae:.4f}")
    print(f"  제출 예상: ~{final_mae + 1.39:.2f}")
    print(f"  v6 (10.00) 대비: {(final_mae + 1.39) - 10.00:+.2f}")
    print(f"\n첫 5행:")
    print(submission.head())
else:
    print(f"개선 없음 ({final_mae - 8.6293:+.4f})")
    print("v6 그대로 유지하는 게 안전")

# %% [markdown]
# ## 단계별 효과 요약 + 다음 단계 가이드

# %%
print("="*70)
print("Phase 1 단계별 효과 요약")
print("="*70)

steps = [
    ('CatBoost 기존', mae_cb_orig, '-'),
    ('Step 1: + Target Encoding', mae_cb_v1, mae_cb_v1 - mae_cb_orig),
    ('Step 2: + Layout Cluster', mae_cb_v2, mae_cb_v2 - mae_cb_v1),
    ('Step 3: + Pseudo-labeling', mae_cb_pseudo, mae_cb_pseudo - mae_cb_v2),
    ('Step 4: 최종 Stacking', final_mae, final_mae - mae_cb_pseudo),
]

print(f"\n{'단계':30s} {'MAE':>8s} {'변화':>8s}")
print("-"*50)
for name, mae, change in steps:
    if isinstance(change, str):
        print(f"{name:30s} {mae:>8.4f} {change:>8s}")
    else:
        marker = ' ↓' if change < 0 else (' ↑' if change > 0 else ' -')
        print(f"{name:30s} {mae:>8.4f} {change:>+8.4f}{marker}")

print(f"\n[총 변화]")
print(f"  v6 OOF (8.6293) → Phase 1 ({final_mae:.4f})")
print(f"  변화: {final_mae - 8.6293:+.4f}")

print(f"\n[다음 단계 가이드]")
if final_mae < 8.55:
    print(f"  ★ 큰 개선! Phase 2로 진행")
    print(f"  → Bi-LSTM 추가 또는 더 많은 base 모델")
    print(f"  → 입상권 진입 가능성 (Top 30)")
elif final_mae < 8.60:
    print(f"  ★ 의미 있는 개선")
    print(f"  → 제출해서 갭 확인")
    print(f"  → 갭이 비슷하면 1D CNN 또는 Bi-LSTM 시도")
elif final_mae < 8.62:
    print(f"  작은 개선")
    print(f"  → 제출 권장 (운에 맡기기)")
else:
    print(f"  의미 있는 개선 없음")
    print(f"  → v6 유지가 안전")
    print(f"  → 다른 방향 시도 (시퀀스 모델, 외부 정보)")
