# %% [markdown]
# # phase2_full_pipeline.py
#
# Auto-converted from phase2_full_pipeline.ipynb.
# This version consumes Phase 1 CNN-encoder artifacts and skips Phase 2 CNN retraining.
# Paths are configured for this repository layout:
#   open/*.csv -> input data
#   phase2_artifacts/*.npy -> generated Phase 2 artifacts
#   submission_phase2_cnnenc.csv -> final submission

# %% [markdown]
# # Phase 2: CNN encoder Phase1 결과 기반 후처리## 전체 흐름```Step 0: 환경 + 데이터 + 핵심 피처Step 1: phase1_cnn_encoder.py 결과 *_cnnenc.npy 로드Step 2: 가중 평균 grid searchStep 3: Phase2 CNN 재학습 스킵Step 4: KNN Layout TEStep 5: Iterative Pseudo-labelingStep 6: 메가 StackingStep 7: 최종 비교 + submission_phase2_cnnenc.csv 생성```## 사전 조건phase1_cnn_encoder.py에서 저장된 .npy 파일들:- `oof_cb_avg_cnnenc.npy`, `test_cb_avg_cnnenc.npy`- `oof_lgb_cnnenc.npy`, `test_lgb_cnnenc.npy`- `oof_xgb_cnnenc.npy`, `test_xgb_cnnenc.npy` (있으면)- `oof_cb_te_cnnenc.npy`, `test_cb_te_cnnenc.npy`- `oof_cb_te_cluster_cnnenc.npy`, `test_cb_te_cluster_cnnenc.npy`- `oof_cb_pseudo_cnnenc.npy`, `test_cb_pseudo_cnnenc.npy`

# %% [markdown]
# ## Step 0: 환경 + 데이터 + 피처

# %%
import pandas as pd
import numpy as np
import gc
import os
import psutil
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
ARTIFACT_DIR = PROJECT_ROOT / 'phase2_artifacts'
SUBMISSION_PATH = PROJECT_ROOT / 'submission_phase2_cnnenc.csv'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
LAYOUT_PATH = DATA_DIR / 'layout_info.csv'
SAMPLE_SUBMISSION_PATH = DATA_DIR / 'sample_submission.csv'

def find_input_artifact(filename):
    """Phase 1 artifacts may already exist in project root; prefer Phase 2 dir if present."""
    candidates = [ARTIFACT_DIR / filename, PROJECT_ROOT / filename]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

def save_artifact(filename, array):
    path = ARTIFACT_DIR / filename
    np.save(path, array)
    return path

USE_PHASE1_CNN_ENCODER_ARTIFACTS = True
RUN_PHASE2_CNN_RETRAIN = False
OUTPUT_SUFFIX = '_cnnenc'

if HAS_TORCH:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch: {torch.__version__}, Device: {device}")
print(f"CatBoost: {HAS_CATBOOST}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"ARTIFACT_DIR: {ARTIFACT_DIR}")
mem_check("초기")

# %%
# === 데이터 로드 ===
sample = pd.read_csv(TRAIN_PATH, nrows=100)
numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
dtypes = {col: 'float32' for col in numeric_cols}
dtypes['ID'] = 'object'
dtypes['scenario_id'] = 'object'
dtypes['layout_id'] = 'object'

train = pd.read_csv(TRAIN_PATH, dtype=dtypes)
test = pd.read_csv(TEST_PATH, dtype=dtypes)
layout = pd.read_csv(LAYOUT_PATH)
for col in layout.columns:
    if layout[col].dtype == 'float64':
        layout[col] = layout[col].astype('float32')

target_col = 'avg_delay_minutes_next_30m'

train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')
del layout
gc.collect()

train = train.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
test = test.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
train['slot_idx'] = train.groupby('scenario_id').cumcount().astype('int8')
test['slot_idx'] = test.groupby('scenario_id').cumcount().astype('int8')

print(f"train: {train.shape}, test: {test.shape}")
mem_check("데이터 로드 후")

# %%
# === 핵심 피처 ===
def build_features(df):
    df['total_fleet'] = (df['robot_active'] + df['robot_idle'] + df['robot_charging']).astype('float32')
    df['idle_ratio'] = (df['robot_idle'] / (df['total_fleet'] + 1e-6)).astype('float32')
    df['charging_ratio'] = (df['robot_charging'] / (df['total_fleet'] + 1e-6)).astype('float32')
    df['active_ratio'] = (df['robot_active'] / (df['total_fleet'] + 1e-6)).astype('float32')
    
    df['has_collision'] = (df['near_collision_15m'] > 0).astype('int8')
    df['has_fault'] = (df['fault_count_15m'] > 0).astype('int8')
    df['pack_saturated'] = (df['pack_utilization'] > 0.95).astype('int8')
    
    df['pack_x_slot'] = (df['pack_utilization'] * df['slot_idx']).astype('float32')
    df['pack_x_battery'] = (df['pack_utilization'] * df['low_battery_ratio']).astype('float32')
    df['pack_x_charging'] = (df['pack_utilization'] * df['charging_ratio']).astype('float32')
    df['cong_x_inflow'] = (df['congestion_score'] * df['order_inflow_15m']).astype('float32')
    df['effective_load'] = (df['order_inflow_15m'] * (1 + 0.5 * df['urgent_order_ratio'])).astype('float32')
    
    df['cum_orders'] = df.groupby('scenario_id')['order_inflow_15m'].cumsum().astype('float32')
    df['cum_faults'] = df.groupby('scenario_id')['fault_count_15m'].cumsum().astype('float32')
    
    for f in ['order_inflow_15m', 'pack_utilization', 'low_battery_ratio']:
        df[f'{f}_lag1'] = df.groupby('scenario_id')[f].shift(1).astype('float32')
    
    df['robots_per_pack'] = (df['robot_total'] / (df['pack_station_count'] + 1)).astype('float32')
    df['robots_per_charger'] = (df['robot_total'] / (df['charger_count'] + 1)).astype('float32')
    df['area_per_robot'] = (df['floor_area_sqm'] / (df['robot_total'] + 1)).astype('float32')
    
    df['utilization_imbalance'] = (df['pack_utilization'] - df['idle_ratio']).astype('float32')
    df['max_utilization'] = df[['pack_utilization', 'charging_ratio']].max(axis=1).astype('float32')
    
    for col in ['battery_mean', 'low_battery_ratio', 'pack_utilization', 'congestion_score']:
        df[f'{col}_isna'] = df[col].isna().astype('int8')
    
    return df

train = build_features(train)
test = build_features(test)

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

# 학습 준비
common_cols = [c for c in train.columns if c in test.columns]
meta_cols = ['ID', 'scenario_id', target_col]
feature_cols = [c for c in common_cols if c not in meta_cols]

train['layout_id'] = train['layout_id'].astype('category')
train['layout_type'] = train['layout_type'].astype('category')
test['layout_id'] = test['layout_id'].astype('category')
test['layout_type'] = test['layout_type'].astype('category')

X_train = train[feature_cols].reset_index(drop=True)
y_train_arr = train[target_col].values.astype('float32')
y_log = np.log1p(y_train_arr).astype('float32')
groups = train['scenario_id'].values
X_test = test[feature_cols].reset_index(drop=True)

gkf = GroupKFold(n_splits=5)
mem_check("피처 + 준비 완료")

# %% [markdown]
# ## Step 1: Phase 1 결과 로드 (재학습 없음)

# %%
# === 모든 base 모델 .npy 로드 ===
print("=== Base 모델 OOF/Test 로드 ===")
print("Phase1 CNN encoder 산출물(*_cnnenc.npy)을 사용합니다.")

base_models = {}

required_files = [
    ('cb_orig', 'oof_cb_avg_cnnenc.npy', 'test_cb_avg_cnnenc.npy'),
    ('lgb', 'oof_lgb_cnnenc.npy', 'test_lgb_cnnenc.npy'),
    ('xgb', 'oof_xgb_cnnenc.npy', 'test_xgb_cnnenc.npy'),
    ('cb_te', 'oof_cb_te_cnnenc.npy', 'test_cb_te_cnnenc.npy'),
    ('cb_cluster', 'oof_cb_te_cluster_cnnenc.npy', 'test_cb_te_cluster_cnnenc.npy'),
    ('cb_pseudo', 'oof_cb_pseudo_cnnenc.npy', 'test_cb_pseudo_cnnenc.npy'),
]

for name, oof_file, test_file in required_files:
    oof_path = find_input_artifact(oof_file)
    test_path = find_input_artifact(test_file)
    if oof_path.exists() and test_path.exists():
        oof = np.load(oof_path).astype('float32')
        test_pred = np.load(test_path).astype('float32')
        if len(oof) == len(X_train) and len(test_pred) == len(X_test):
            base_models[name] = (oof, test_pred)
            mae = mean_absolute_error(y_train_arr, oof)
            print(f"  ✓ {name}: MAE = {mae:.4f} ({oof_path.name}, {test_path.name})")
        else:
            print(f"  ✗ {name}: 길이 불일치")
    else:
        print(f"  - {name}: 파일 없음 ({oof_path}, {test_path})")

print(f"\n로드된 base 모델: {len(base_models)}")

if len(base_models) < 4:
    print("⚠ Phase 1 결과 부족. Phase 1 노트북 먼저 실행 필요.")
if not base_models:
    raise FileNotFoundError(
        f"Phase 1 .npy 파일을 찾지 못했습니다. {ARTIFACT_DIR} 또는 {PROJECT_ROOT}에 "
        "oof_*.npy/test_*.npy 파일을 먼저 생성해 주세요."
    )
mem_check("Base 로드 완료")

# %% [markdown]
# ## Step 2: 가중 평균 Grid Search (10분)Stacking과 비교해서 더 좋은 거 채택

# %%
# === 가중 평균 Grid Search ===
print("=== 가중 평균 Grid Search ===")
print(f"Phase 1 Stacking 점수: 8.5452 (목표: 이거보다 좋게)")

# 강한 모델 위주로 grid search
oof_dict = {name: oof for name, (oof, _) in base_models.items()}
test_dict = {name: test_p for name, (_, test_p) in base_models.items()}

# 5개 모델 가중치 grid (합 1)
best_mae = float('inf')
best_w = None

# 강한 모델 (cb_pseudo, cb_cluster) 위주로 weight 큰 범위
weights_cb_pseudo = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
weights_others = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

count = 0
total_combinations = 0

# 모델 키 정렬
model_keys = list(oof_dict.keys())
print(f"포함 모델: {model_keys}")

# 5개 weight grid
for w_pseudo in weights_cb_pseudo:
    for w_cluster in weights_others:
        for w_te in weights_others:
            for w_orig in weights_others:
                if 'lgb' in oof_dict:
                    w_lgb = 1.0 - w_pseudo - w_cluster - w_te - w_orig
                    if w_lgb < 0 or w_lgb > 0.5: continue
                    
                    ens = (w_pseudo * oof_dict.get('cb_pseudo', oof_dict[model_keys[0]]) + 
                           w_cluster * oof_dict.get('cb_cluster', oof_dict[model_keys[0]]) + 
                           w_te * oof_dict.get('cb_te', oof_dict[model_keys[0]]) +
                           w_orig * oof_dict.get('cb_orig', oof_dict[model_keys[0]]) +
                           w_lgb * oof_dict.get('lgb', oof_dict[model_keys[0]]))
                else:
                    w_orig_full = 1.0 - w_pseudo - w_cluster - w_te
                    if w_orig_full < 0: continue
                    
                    ens = (w_pseudo * oof_dict.get('cb_pseudo', oof_dict[model_keys[0]]) + 
                           w_cluster * oof_dict.get('cb_cluster', oof_dict[model_keys[0]]) + 
                           w_te * oof_dict.get('cb_te', oof_dict[model_keys[0]]) +
                           w_orig_full * oof_dict.get('cb_orig', oof_dict[model_keys[0]]))
                
                mae = mean_absolute_error(y_train_arr, ens)
                if mae < best_mae:
                    best_mae = mae
                    best_w = {
                        'cb_pseudo': w_pseudo,
                        'cb_cluster': w_cluster,
                        'cb_te': w_te,
                        'cb_orig': w_orig,
                        'lgb': w_lgb if 'lgb' in oof_dict else 0
                    }
                count += 1

print(f"\n조합 평가: {count:,}")
print(f"가중 평균 best MAE: {best_mae:.4f}")
print(f"Best weights: {best_w}")
print(f"\nPhase 1 Stacking (8.5452) 대비: {best_mae - 8.5452:+.4f}")

# 가장 좋은 가중 평균 결과 저장
weighted_oof = np.zeros(len(y_train_arr), dtype='float32')
weighted_test = np.zeros(len(X_test), dtype='float32')
for name, w in best_w.items():
    if name in oof_dict:
        weighted_oof += w * oof_dict[name]
        weighted_test += w * test_dict[name]

save_artifact(f'oof_weighted{OUTPUT_SUFFIX}.npy', weighted_oof)
save_artifact(f'test_weighted{OUTPUT_SUFFIX}.npy', weighted_test)
print(f"\n가중 평균 저장: weighted_oof, weighted_test")
mem_check("Step 2 완료")

# %% [markdown]
# ## Step 3: CNN Encoder + Boosting (1~2시간)시퀀스 정보 추출 → 부스팅 입력 추가

# %%
if not RUN_PHASE2_CNN_RETRAIN:
    print("Phase1 CNN encoder 결과를 사용하므로 Phase2 Step 3 CNN 재학습 스킵")
    SKIP_CNN = True
elif not HAS_TORCH:
    print("PyTorch 없음 - Step 3 스킵")
    SKIP_CNN = True
else:
    SKIP_CNN = False
    
    # === 시퀀스 데이터 준비 ===
    dynamic_cols = [
        'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle',
        'robot_charging', 'battery_mean', 'low_battery_ratio',
        'charge_queue_length', 'avg_charge_wait', 'congestion_score',
        'max_zone_density', 'blocked_path_15m', 'near_collision_15m',
        'fault_count_15m', 'task_reassign_15m', 'pack_utilization',
        'loading_dock_util', 'staging_area_util', 'urgent_order_ratio',
    ]
    dynamic_cols = [c for c in dynamic_cols if c in train.columns]
    n_features = len(dynamic_cols)
    print(f"Dynamic 피처: {n_features}")
    
    def build_seq_with_mask(df, dynamic_cols, target_col=None, scaler=None):
        df_sorted = df.sort_values(['scenario_id', 'slot_idx']).reset_index(drop=True)
        
        # 마스크 (채우기 전)
        mask_arr = df_sorted[dynamic_cols].isna().astype('float32').values
        
        # 시나리오 평균
        df_filled = df_sorted.copy()
        for col in dynamic_cols:
            df_filled[col] = df_filled.groupby('scenario_id')[col].transform(
                lambda x: x.fillna(x.mean())
            )
            if df_filled[col].isna().any():
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        value_arr = df_filled[dynamic_cols].values.astype('float32')
        
        if scaler is None:
            scaler = StandardScaler()
            value_arr = scaler.fit_transform(value_arr).astype('float32')
        else:
            value_arr = scaler.transform(value_arr).astype('float32')
        
        n_scenarios = df_sorted['scenario_id'].nunique()
        value_seq = value_arr.reshape(n_scenarios, 25, len(dynamic_cols))
        mask_seq = mask_arr.reshape(n_scenarios, 25, len(dynamic_cols))
        combined_seq = np.concatenate([value_seq, mask_seq], axis=2).astype('float32')
        
        target_seq = None
        if target_col and target_col in df_sorted.columns:
            target_seq = df_sorted[target_col].values.reshape(n_scenarios, 25).astype('float32')
        
        sorted_ids = df_sorted['ID'].values
        return combined_seq, target_seq, scaler, sorted_ids
    
    print("\nTrain 시퀀스...")
    train_seq, y_seq, scaler, train_sorted_ids = build_seq_with_mask(
        train, dynamic_cols, target_col=target_col
    )
    print("Test 시퀀스...")
    test_seq, _, _, test_sorted_ids = build_seq_with_mask(
        test, dynamic_cols, target_col=None, scaler=scaler
    )
    
    n_features_with_mask = train_seq.shape[2]
    print(f"\nTrain seq: {train_seq.shape}, Test seq: {test_seq.shape}")
    mem_check("시퀀스 준비")

# %%
if not SKIP_CNN:
    # === CNN Encoder ===
    class CNN1DEncoder(nn.Module):
        def __init__(self, n_features_with_mask, embedding_dim=32, dropout=0.2):
            super().__init__()
            hidden = 32
            
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features_with_mask, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden * 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(hidden * 2, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden), nn.ReLU(),
            )
            self.embedding_layer = nn.Conv1d(hidden, embedding_dim, kernel_size=1)
            self.head = nn.Linear(embedding_dim, 1)
        
        def get_embedding(self, x):
            x = x.transpose(1, 2)
            x = self.encoder(x)
            emb = self.embedding_layer(x)
            return emb.transpose(1, 2)
        
        def forward(self, x):
            emb = self.get_embedding(x)
            return emb, self.head(emb).squeeze(-1)
    
    embedding_dim = 32
    
    # === 5-fold 학습 + 임베딩 추출 ===
    n_train_sc = train_seq.shape[0]
    n_test_sc = test_seq.shape[0]
    
    oof_embeddings = np.zeros((n_train_sc, 25, embedding_dim), dtype='float32')
    test_embeddings_sum = np.zeros((n_test_sc, 25, embedding_dim), dtype='float32')
    oof_cnn_pred_log = np.zeros((n_train_sc, 25), dtype='float32')
    test_cnn_pred_log = np.zeros((n_test_sc, 25), dtype='float32')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    cnn_start = time.time()
    
    X_test_t = torch.FloatTensor(test_seq).to(device)
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train_sc))):
        print(f"\nCNN Fold {fold+1}/5")
        fold_start = time.time()
        
        X_tr = torch.FloatTensor(train_seq[tr_idx]).to(device)
        y_tr = torch.FloatTensor(y_seq[tr_idx]).to(device)
        X_val = torch.FloatTensor(train_seq[val_idx]).to(device)
        y_val = torch.FloatTensor(y_seq[val_idx]).to(device)
        
        torch.manual_seed(42 + fold)
        model = CNN1DEncoder(n_features_with_mask, embedding_dim=embedding_dim, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        criterion = nn.L1Loss()
        
        y_tr_log = torch.log1p(torch.clamp(y_tr, min=0))
        y_val_log = torch.log1p(torch.clamp(y_val, min=0))
        
        best_val_mae = float('inf')
        best_state = None
        bad_epochs = 0
        
        for epoch in range(40):
            model.train()
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                bi = idx[i:i+64]
                optimizer.zero_grad()
                _, pred = model(X_tr[bi])
                loss = criterion(pred, y_tr_log[bi])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                _, val_pred = model(X_val)
                val_mae = (torch.expm1(val_pred).clamp(min=0) - y_val).abs().mean().item()
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= 8: break
        
        fold_scores.append(best_val_mae)
        print(f"  Fold {fold+1} val_mae: {best_val_mae:.4f} ({time.time()-fold_start:.0f}초)")
        
        # 임베딩 추출
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            val_emb = model.get_embedding(X_val)
            oof_embeddings[val_idx] = val_emb.cpu().numpy().astype('float32')
            _, val_pred_log = model(X_val)
            oof_cnn_pred_log[val_idx] = val_pred_log.cpu().numpy().astype('float32')
            
            test_emb = model.get_embedding(X_test_t)
            test_embeddings_sum += test_emb.cpu().numpy().astype('float32') / 5
            _, test_pred_log = model(X_test_t)
            test_cnn_pred_log += test_pred_log.cpu().numpy().astype('float32') / 5
        
        del model, X_tr, y_tr, X_val, y_val
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    del X_test_t
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # CNN 예측
    oof_cnn_pred = np.clip(np.expm1(oof_cnn_pred_log.flatten()), 0, None)
    test_cnn_pred = np.clip(np.expm1(test_cnn_pred_log.flatten()), 0, None)
    
    train_orig_ids = train['ID'].values
    test_orig_ids = test['ID'].values
    if not np.array_equal(train_sorted_ids, train_orig_ids):
        pred_map = dict(zip(train_sorted_ids, oof_cnn_pred))
        oof_cnn_pred = np.array([pred_map[id_] for id_ in train_orig_ids], dtype='float32')
    if not np.array_equal(test_sorted_ids, test_orig_ids):
        pred_map = dict(zip(test_sorted_ids, test_cnn_pred))
        test_cnn_pred = np.array([pred_map[id_] for id_ in test_orig_ids], dtype='float32')
    
    cnn_oof_mae = mean_absolute_error(y_train_arr, oof_cnn_pred)
    print(f"\nCNN 단독 OOF MAE: {cnn_oof_mae:.4f}")
    print(f"CNN 학습 시간: {time.time()-cnn_start:.0f}초")
    
    save_artifact(f'oof_cnn{OUTPUT_SUFFIX}.npy', oof_cnn_pred)
    save_artifact(f'test_cnn{OUTPUT_SUFFIX}.npy', test_cnn_pred)
    
    # 임베딩을 부스팅 입력 형태로
    oof_emb_flat = oof_embeddings.reshape(-1, embedding_dim)
    test_emb_flat = test_embeddings_sum.reshape(-1, embedding_dim)
    
    emb_cols = [f'cnn_emb_{i}' for i in range(embedding_dim)]
    
    if not np.array_equal(train_sorted_ids, train_orig_ids):
        train_emb_df = pd.DataFrame(oof_emb_flat, columns=emb_cols)
        train_emb_df['ID'] = train_sorted_ids
        train_emb_df = train_emb_df.set_index('ID').reindex(train_orig_ids).reset_index(drop=True)
        
        test_emb_df = pd.DataFrame(test_emb_flat, columns=emb_cols)
        test_emb_df['ID'] = test_sorted_ids
        test_emb_df = test_emb_df.set_index('ID').reindex(test_orig_ids).reset_index(drop=True)
    else:
        train_emb_df = pd.DataFrame(oof_emb_flat, columns=emb_cols).astype('float32')
        test_emb_df = pd.DataFrame(test_emb_flat, columns=emb_cols).astype('float32')
    
    for col in emb_cols:
        train_emb_df[col] = train_emb_df[col].astype('float32')
        test_emb_df[col] = test_emb_df[col].astype('float32')
    
    save_artifact(f'train_emb_flat{OUTPUT_SUFFIX}.npy', train_emb_df.values)
    save_artifact(f'test_emb_flat{OUTPUT_SUFFIX}.npy', test_emb_df.values)
    
    del train_seq, test_seq, y_seq, oof_embeddings, test_embeddings_sum
    gc.collect()
    mem_check("CNN 완료")

# %% [markdown]
# ### Step 3.1: CatBoost (CNN 임베딩 추가)

# %%
if not SKIP_CNN and HAS_CATBOOST:
    # === CatBoost (임베딩 추가) ===
    X_train_with_emb = pd.concat([X_train, train_emb_df], axis=1)
    X_test_with_emb = pd.concat([X_test, test_emb_df], axis=1)
    
    print(f"임베딩 추가 후 피처: {X_train_with_emb.shape[1]}")
    
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
        'random_seed': 42,
    }
    
    oof_cb_emb_log = np.zeros(len(X_train_with_emb), dtype='float32')
    test_cb_emb_log = np.zeros(len(X_test_with_emb), dtype='float32')
    
    print("\n=== CatBoost (CNN 임베딩 추가) ===")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_with_emb, y_log, groups=groups)):
        print(f"  Fold {fold+1}/5")
        X_tr = X_train_with_emb.iloc[tr_idx].copy()
        X_val = X_train_with_emb.iloc[val_idx].copy()
        for col in ['layout_id', 'layout_type']:
            X_tr[col] = X_tr[col].astype(str)
            X_val[col] = X_val[col].astype(str)
        
        model = CatBoostRegressor(**cb_params)
        model.fit(X_tr, y_log[tr_idx], eval_set=(X_val, y_log[val_idx]),
                  use_best_model=True, verbose=0)
        oof_cb_emb_log[val_idx] = model.predict(X_val).astype('float32')
        
        X_test_cb = X_test_with_emb.copy()
        for col in ['layout_id', 'layout_type']:
            X_test_cb[col] = X_test_cb[col].astype(str)
        test_cb_emb_log += (model.predict(X_test_cb) / 5).astype('float32')
        
        del model, X_tr, X_val, X_test_cb
        gc.collect()
    
    oof_cb_emb = np.clip(np.expm1(oof_cb_emb_log), 0, None).astype('float32')
    test_cb_emb = np.clip(np.expm1(test_cb_emb_log), 0, None).astype('float32')
    
    cb_emb_mae = mean_absolute_error(y_train_arr, oof_cb_emb)
    print(f"\nCatBoost (+ CNN 임베딩) MAE: {cb_emb_mae:.4f}")
    print(f"기존 CatBoost (cb_orig) 대비: {cb_emb_mae - mean_absolute_error(y_train_arr, base_models['cb_orig'][0]):+.4f}")
    
    save_artifact(f'oof_cb_emb{OUTPUT_SUFFIX}.npy', oof_cb_emb)
    save_artifact(f'test_cb_emb{OUTPUT_SUFFIX}.npy', test_cb_emb)
    
    base_models['cb_emb'] = (oof_cb_emb, test_cb_emb)
    base_models['cnn'] = (oof_cnn_pred, test_cnn_pred)
    
    del X_train_with_emb, X_test_with_emb
    gc.collect()
    mem_check("Step 3 완료")

# %% [markdown]
# ## Step 4: KNN Layout Target Encoding (30분)신규 layout 50개에 대한 일반화 강화

# %%
# === KNN Layout Target Encoding ===
print("=== KNN Layout Target Encoding ===")

layout_attrs = ['pack_station_count', 'charger_count', 'floor_area_sqm',
                'aisle_width_avg', 'intersection_count', 'one_way_ratio',
                'ceiling_height_m', 'building_age_years',
                'layout_compactness', 'zone_dispersion']
layout_attrs = [c for c in layout_attrs if c in train.columns]
print(f"Layout 속성: {len(layout_attrs)}개")

# Train layout target 평균
train_layout_target = train.groupby('layout_id', observed=True)[target_col].mean()
print(f"Train unique layout: {len(train_layout_target)}")

# Train layout 속성
train_layout_attrs = train.groupby('layout_id', observed=True)[layout_attrs].first().fillna(0)

# 표준화
scaler_l = StandardScaler()
train_layout_attrs_scaled = scaler_l.fit_transform(train_layout_attrs.values)

# KNN
for k in [3, 5, 10]:
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(train_layout_attrs_scaled)
    
    # 모든 layout
    all_layouts = pd.concat([train, test])[['layout_id'] + layout_attrs].drop_duplicates('layout_id').reset_index(drop=True)
    all_layouts_scaled = scaler_l.transform(all_layouts[layout_attrs].fillna(0).values)
    
    distances, indices = knn.kneighbors(all_layouts_scaled)
    
    knn_targets = []
    for idx_arr in indices:
        layout_ids_neighbors = train_layout_attrs.index[idx_arr]
        avg = train_layout_target.loc[layout_ids_neighbors].mean()
        knn_targets.append(avg)
    
    layout_knn_te_map = dict(zip(all_layouts['layout_id'].astype(str), knn_targets))
    
    train[f'layout_knn_te_k{k}'] = train['layout_id'].astype(str).map(layout_knn_te_map).astype('float32')
    test[f'layout_knn_te_k{k}'] = test['layout_id'].astype(str).map(layout_knn_te_map).astype('float32')
    
    print(f"  K={k}: 추가 완료")

# 새 X_train, X_test
common_cols = [c for c in train.columns if c in test.columns]
feature_cols_v3 = [c for c in common_cols if c not in meta_cols]
X_train_v3 = train[feature_cols_v3].reset_index(drop=True)
X_test_v3 = test[feature_cols_v3].reset_index(drop=True)
print(f"\n피처 수 (KNN TE 추가): {len(feature_cols_v3)}")
mem_check("KNN TE 추가")

# %%
# === CatBoost (KNN Layout TE 추가) ===
if HAS_CATBOOST:
    cb_params_v3 = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.05,
        'depth': 8,
        'iterations': 3000,
        'cat_features': ['layout_id', 'layout_type'],
        'early_stopping_rounds': 100,
        'verbose': 0,
        'thread_count': -1,
        'random_seed': 42,
    }
    
    oof_cb_knn_log = np.zeros(len(X_train_v3), dtype='float32')
    test_cb_knn_log = np.zeros(len(X_test_v3), dtype='float32')
    
    print("=== CatBoost (KNN Layout TE 추가) ===")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_v3, y_log, groups=groups)):
        print(f"  Fold {fold+1}/5")
        X_tr = X_train_v3.iloc[tr_idx].copy()
        X_val = X_train_v3.iloc[val_idx].copy()
        for col in ['layout_id', 'layout_type']:
            X_tr[col] = X_tr[col].astype(str)
            X_val[col] = X_val[col].astype(str)
        
        model = CatBoostRegressor(**cb_params_v3)
        model.fit(X_tr, y_log[tr_idx], eval_set=(X_val, y_log[val_idx]),
                  use_best_model=True, verbose=0)
        oof_cb_knn_log[val_idx] = model.predict(X_val).astype('float32')
        
        X_test_cb = X_test_v3.copy()
        for col in ['layout_id', 'layout_type']:
            X_test_cb[col] = X_test_cb[col].astype(str)
        test_cb_knn_log += (model.predict(X_test_cb) / 5).astype('float32')
        
        del model, X_tr, X_val, X_test_cb
        gc.collect()
    
    oof_cb_knn = np.clip(np.expm1(oof_cb_knn_log), 0, None).astype('float32')
    test_cb_knn = np.clip(np.expm1(test_cb_knn_log), 0, None).astype('float32')
    
    cb_knn_mae = mean_absolute_error(y_train_arr, oof_cb_knn)
    print(f"\nCatBoost (+ KNN Layout TE) MAE: {cb_knn_mae:.4f}")
    
    save_artifact(f'oof_cb_knn{OUTPUT_SUFFIX}.npy', oof_cb_knn)
    save_artifact(f'test_cb_knn{OUTPUT_SUFFIX}.npy', test_cb_knn)
    
    base_models['cb_knn'] = (oof_cb_knn, test_cb_knn)
    mem_check("Step 4 완료")

# %% [markdown]
# ## Step 5: Iterative Pseudo-labeling (1~2시간)Pseudo가 가장 큰 효과 보였으니 반복으로 강화

# %%
# === Iterative Pseudo-labeling ===
print("=== Iterative Pseudo-labeling ===")

# 현재 best test 예측 (가장 좋은 stacking 또는 가중 평균)
# Phase 1 stacking이 8.5452였으니 그거 사용
# 더 좋은 게 있으면 그거 사용

# 일단 cb_pseudo로 시작 (Phase 1의 best base)
if 'cb_pseudo' not in base_models:
    raise KeyError("cb_pseudo 모델이 없습니다. phase1_cnn_encoder.py를 끝까지 실행해 oof_cb_pseudo_cnnenc.npy/test_cb_pseudo_cnnenc.npy를 생성해 주세요.")
current_test = base_models['cb_pseudo'][1].copy()
current_oof = base_models['cb_pseudo'][0].copy()

# Iter 1, 2 시도
n_iters = 2

for iter_n in range(n_iters):
    print(f"\n=== Iteration {iter_n + 1}/{n_iters} ===")
    
    # 신뢰 영역 (점점 더 좁게)
    conf_lower = 5
    conf_upper = 35 if iter_n == 0 else 30
    confident_mask = (current_test >= conf_lower) & (current_test <= conf_upper)
    print(f"  신뢰 영역 ({conf_lower}~{conf_upper}분): {confident_mask.sum()}/{len(current_test)}")
    
    # Pseudo dataset
    X_pseudo = pd.concat([X_train_v3, X_test_v3[confident_mask]], ignore_index=True)
    y_pseudo = np.concatenate([y_train_arr, current_test[confident_mask]]).astype('float32')
    y_pseudo_log = np.log1p(y_pseudo).astype('float32')
    pseudo_test_groups = test['scenario_id'].values[confident_mask]
    groups_pseudo = np.concatenate([groups, pseudo_test_groups])
    
    # CatBoost 학습
    oof_iter_log = np.zeros(len(X_train_v3), dtype='float32')
    test_iter_log = np.zeros(len(X_test_v3), dtype='float32')
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_pseudo, y_pseudo_log, groups=groups_pseudo)):
        val_idx_orig = val_idx[val_idx < len(X_train_v3)]
        if len(val_idx_orig) == 0: continue
        
        print(f"    Fold {fold+1}/5")
        X_tr = X_pseudo.iloc[tr_idx].copy()
        X_val = X_pseudo.iloc[val_idx_orig].copy()
        for col in ['layout_id', 'layout_type']:
            X_tr[col] = X_tr[col].astype(str)
            X_val[col] = X_val[col].astype(str)
        
        model = CatBoostRegressor(**cb_params_v3)
        model.fit(X_tr, y_pseudo_log[tr_idx], 
                  eval_set=(X_val, y_pseudo_log[val_idx_orig]),
                  use_best_model=True, verbose=0)
        oof_iter_log[val_idx_orig] = model.predict(X_val).astype('float32')
        
        X_test_cb = X_test_v3.copy()
        for col in ['layout_id', 'layout_type']:
            X_test_cb[col] = X_test_cb[col].astype(str)
        test_iter_log += (model.predict(X_test_cb) / 5).astype('float32')
        
        del model, X_tr, X_val, X_test_cb
        gc.collect()
    
    new_oof = np.clip(np.expm1(oof_iter_log), 0, None).astype('float32')
    new_test = np.clip(np.expm1(test_iter_log), 0, None).astype('float32')
    
    new_mae = mean_absolute_error(y_train_arr, new_oof)
    print(f"  Iter {iter_n+1} OOF MAE: {new_mae:.4f}")
    
    # 다음 iter용
    current_test = new_test
    current_oof = new_oof
    
    # 저장
    save_artifact(f'oof_pseudo_iter{iter_n+1}{OUTPUT_SUFFIX}.npy', new_oof)
    save_artifact(f'test_pseudo_iter{iter_n+1}{OUTPUT_SUFFIX}.npy', new_test)
    
    # base_models에 추가
    base_models[f'cb_pseudo_iter{iter_n+1}'] = (new_oof, new_test)
    
    del X_pseudo
    gc.collect()
    mem_check(f"Iter {iter_n+1} 완료")

# %% [markdown]
# ## Step 6: 메가 Stacking (모든 base 결합)

# %%
# === 모든 base 모델 정리 ===
print("=== 모든 base 모델 ===")
for name, (oof, _) in base_models.items():
    mae = mean_absolute_error(y_train_arr, oof)
    print(f"  {name}: {mae:.4f}")

# === 메가 Stacking ===
print("\n=== 메가 Stacking ===")

X_meta_mega = np.column_stack([oof for oof, _ in base_models.values()]).astype('float32')
X_meta_mega_test = np.column_stack([test_p for _, test_p in base_models.values()]).astype('float32')

# 시나리오 메타 추가
key_meta = ['low_battery_ratio_sc_mean', 'pack_utilization_sc_mean',
             'congestion_score_sc_mean', 'slot_idx', 'pack_x_battery']
key_meta = [c for c in key_meta if c in X_train.columns]

X_meta_mega = np.column_stack([X_meta_mega, X_train[key_meta].fillna(0).values.astype('float32')])
X_meta_mega_test = np.column_stack([X_meta_mega_test, X_test[key_meta].fillna(0).values.astype('float32')])

print(f"메가 메타 입력: {X_meta_mega.shape}")
print(f"포함 base: {list(base_models.keys())}")

# 메타 학습
meta_params = {
    'objective': 'regression_l1', 'metric': 'mae',
    'learning_rate': 0.03, 'num_leaves': 15,
    'min_child_samples': 200,
    'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'verbose': -1, 'random_state': 42,
}

mega_oof = np.zeros(len(y_train_arr), dtype='float32')
mega_test = np.zeros(len(X_test), dtype='float32')

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_meta_mega, y_log, groups=groups)):
    print(f"  Fold {fold+1}/5")
    train_data = lgb.Dataset(X_meta_mega[tr_idx], label=y_log[tr_idx])
    val_data = lgb.Dataset(X_meta_mega[val_idx], label=y_log[val_idx], reference=train_data)
    meta_model = lgb.train(meta_params, train_data, num_boost_round=2000,
                            valid_sets=[val_data],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    mega_oof[val_idx] = meta_model.predict(X_meta_mega[val_idx]).astype('float32')
    mega_test += (meta_model.predict(X_meta_mega_test) / 5).astype('float32')
    del meta_model
    gc.collect()

mega_oof_raw = np.clip(np.expm1(mega_oof), 0, None).astype('float32')
mega_test_raw = np.clip(np.expm1(mega_test), 0, None).astype('float32')

mega_mae = mean_absolute_error(y_train_arr, mega_oof_raw)

print(f"\n=== 메가 Stacking 결과 ===")
print(f"  Phase 1 stacking:  8.5452")
print(f"  메가 Stacking:     {mega_mae:.4f}")
print(f"  변화:              {mega_mae - 8.5452:+.4f}")
print(f"  갭 1.39 적용:      {mega_mae + 1.39:.2f}")

save_artifact(f'oof_mega{OUTPUT_SUFFIX}.npy', mega_oof_raw)
save_artifact(f'test_mega{OUTPUT_SUFFIX}.npy', mega_test_raw)
mem_check("Step 6 완료")

# %% [markdown]
# ## Step 7: 최종 비교 + 제출

# %%
# === 모든 결과 비교 ===
print("="*70)
print("Phase 2 단계별 효과 요약")
print("="*70)

results = {
    'v6 (검증, 제출 10.00)': 8.6293,
    'Phase 1 Stacking': 8.5452,
    'Step 2: 가중 평균': best_mae,
}

if not SKIP_CNN and HAS_CATBOOST and 'cb_emb' in base_models:
    results['Step 3: CB + CNN 임베딩'] = mean_absolute_error(y_train_arr, base_models['cb_emb'][0])

if 'cb_knn' in base_models:
    results['Step 4: CB + KNN Layout TE'] = mean_absolute_error(y_train_arr, base_models['cb_knn'][0])

for i in range(1, 3):
    if f'cb_pseudo_iter{i}' in base_models:
        results[f'Step 5: Pseudo Iter {i}'] = mean_absolute_error(y_train_arr, base_models[f'cb_pseudo_iter{i}'][0])

results['Step 6: 메가 Stacking'] = mega_mae

prev_mae = None
print(f"\n{'단계':40s} {'OOF MAE':>10s} {'변화':>10s}")
print("-"*65)
for name, mae in results.items():
    if prev_mae is None:
        print(f"{name:40s} {mae:>10.4f} {'-':>10s}")
    else:
        change = mae - prev_mae
        marker = ' ↓' if change < 0 else (' ↑' if change > 0 else ' -')
        print(f"{name:40s} {mae:>10.4f} {change:>+10.4f}{marker}")
    prev_mae = mae

# === 가장 좋은 결과 선택 ===
best_method = min(results.items(), key=lambda x: x[1])
print(f"\n=== 최고 OOF: {best_method[0]} = {best_method[1]:.4f} ===")

# 메가 Stacking이 가장 좋으면 사용
final_method = "메가 Stacking"
final_oof = mega_oof_raw
final_test = mega_test_raw
final_mae = mega_mae

# 가중 평균이 메가보다 좋으면 가중 평균
if best_mae < mega_mae:
    final_method = "가중 평균"
    final_oof = weighted_oof
    final_test = weighted_test
    final_mae = best_mae

print(f"\n=== 최종 선택: {final_method} (OOF {final_mae:.4f}) ===")
print(f"제출 예상: ~{final_mae + 1.39:.2f}")
print(f"v6 (10.00) 대비: {(final_mae + 1.39) - 10.00:+.2f}")

# %%
# === 제출 파일 생성 ===
final_test_pred = np.clip(final_test, 0, None)

submission = pd.DataFrame({
    'ID': test['ID'].values,
    target_col: final_test_pred
})

if SAMPLE_SUBMISSION_PATH.exists():
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission = sample[['ID']].merge(submission, on='ID', how='left')
    print("sample_submission 순서 정렬됨")

assert len(submission) == 50000
assert submission[target_col].notna().all()
assert (submission[target_col] >= 0).all()

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\n=== {SUBMISSION_PATH} 저장 완료 ===")
print(f"  방법: {final_method}")
print(f"  OOF MAE: {final_mae:.4f}")
print(f"  제출 예상: ~{final_mae + 1.39:.2f}")
print(f"\n예측 통계:")
print(f"  min:    {final_test_pred.min():.2f}")
print(f"  max:    {final_test_pred.max():.2f}")
print(f"  mean:   {final_test_pred.mean():.2f}")
print(f"  median: {np.median(final_test_pred):.2f}")

print(f"\n첫 5행:")
print(submission.head())

# === 입상권 가능성 평가 ===
print(f"\n=== 입상권 평가 ===")
expected_lb = final_mae + 1.39
if expected_lb < 9.80:
    print(f"  ★★★ Top 30 진입 매우 가능")
    print(f"  ★ Top 10 진입 가능성")
elif expected_lb < 9.90:
    print(f"  ★★ Top 30 진입 가능")
    print(f"  Top 10 도전 가능")
elif expected_lb < 10.00:
    print(f"  ★ Top 50 진입 가능")
elif expected_lb < 10.05:
    print(f"  현재(84등) 비슷 또는 약간 개선")
else:
    print(f"  큰 개선 어려움")

# === 다음 단계 가이드 ===
print(f"\n=== 추후 옵션 ===")
print(f"1. {SUBMISSION_PATH.name} 제출 → 실제 LB 확인")
print(f"2. LB 좋으면 → 더 시도 (Bi-LSTM, Multi-task 등)")
print(f"3. LB 별로면 → v6 또는 Phase 1 결과 제출")

# %% [markdown]
# ## (선택) 잔차 진단

# %%
# === 잔차 진단 ===
print("=== Phase 2 잔차 분석 ===")

diag = pd.DataFrame({
    'y_true': y_train_arr,
    'y_pred': final_oof,
})
diag['abs_err'] = np.abs(diag['y_true'] - diag['y_pred'])
diag['target_bin'] = pd.cut(diag['y_true'], 
                              bins=[-1, 5, 15, 30, 60, 120, 1000],
                              labels=['0-5', '5-15', '15-30', '30-60', '60-120', '120+'])

bin_stats = diag.groupby('target_bin', observed=True).agg(
    count=('y_true', 'count'),
    mae=('abs_err', 'mean'),
    p90=('abs_err', lambda x: x.quantile(0.9))
).round(2)

print("\n타겟 구간별 MAE:")
print(bin_stats)

# 슬롯별
print("\n슬롯별 평균 잔차:")
slot_residual = pd.DataFrame({
    'slot_idx': train['slot_idx'].values,
    'residual': diag['abs_err'].values
}).groupby('slot_idx')['residual'].mean()
for i in range(0, 25, 5):
    chunk = slot_residual.iloc[i:i+5]
    print(f"  slot {i:2d}-{min(i+4,24):2d}: avg {chunk.mean():.2f}")

print("\n=== 끝 ===")
