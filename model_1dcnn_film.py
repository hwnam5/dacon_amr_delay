"""
1D CNN + FiLM 모델 — 시나리오 단위 시계열 예측

  입력: (B, 25, n_features)  ← 시나리오 1개 = 25 time slots × n_features
  출력: (B, 25)              ← 각 슬롯의 avg_delay_minutes_next_30m 예측

  FiLM: layout 정보(layout_type + 13개 수치)를 conditioning vector로 인코딩
        → 각 CNN1DBlock 후에 γ(layout)·x + β(layout) 아핀 변환 적용
        → 창고 구조에 따라 시계열 패턴의 스케일·편향을 동적으로 조정
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

TARGET = "avg_delay_minutes_next_30m"

LAYOUT_TYPE_MAP = {"narrow": 0, "grid": 1, "hybrid": 2, "hub_spoke": 3}

LAYOUT_NUM_COLS = [
    "aisle_width_avg", "intersection_count", "one_way_ratio",
    "pack_station_count", "charger_count", "layout_compactness",
    "zone_dispersion", "robot_total", "building_age_years",
    "floor_area_sqm", "ceiling_height_m", "fire_sprinkler_count",
    "emergency_exit_count",
]

GROUPS = {
    "A": ["order_inflow_15m", "unique_sku_15m", "avg_items_per_order",
          "urgent_order_ratio", "heavy_item_ratio", "cold_chain_ratio",
          "sku_concentration", "return_order_ratio", "bulk_order_ratio",
          "order_wave_count", "pick_list_length_avg", "express_lane_util"],
    "B": ["robot_active", "robot_idle", "robot_charging", "robot_utilization",
          "avg_trip_distance", "task_reassign_15m", "avg_idle_duration_min",
          "agv_task_success_rate", "path_optimization_score",
          "fleet_age_months_avg", "robot_firmware_update_days", "robot_calibration_score"],
    "C": ["battery_mean", "battery_std", "low_battery_ratio",
          "charge_queue_length", "avg_charge_wait", "charge_efficiency_pct",
          "battery_cycle_count_avg"],
    "D": ["congestion_score", "max_zone_density", "blocked_path_15m",
          "near_collision_15m", "aisle_traffic_score", "intersection_wait_time_avg"],
    "E": ["pack_utilization", "replenishment_overlap", "staging_area_util",
          "pallet_wrap_time_min", "loading_dock_util", "outbound_truck_wait_min",
          "sort_accuracy_pct", "quality_check_rate", "packaging_material_cost"],
    "F": ["fault_count_15m", "avg_recovery_time", "manual_override_ratio"],
    "G": ["staff_on_floor", "forklift_active_count", "worker_avg_tenure_months",
          "safety_score_monthly", "shift_handover_delay_min"],
    "H": ["wms_response_time_ms", "network_latency_ms", "wifi_signal_db",
          "scanner_error_rate", "barcode_read_success_rate", "label_print_queue",
          "ups_battery_pct", "daily_forecast_accuracy", "inventory_turnover_rate"],
    "I": ["warehouse_temp_avg", "humidity_pct", "co2_level_ppm", "air_quality_idx",
          "hvac_power_kw", "external_temp_c", "wind_speed_kmh", "precipitation_mm",
          "lighting_level_lux", "lighting_zone_variance", "ambient_noise_db",
          "floor_vibration_idx", "cold_storage_temp_c", "zone_temp_variance"],
    "J": ["storage_density_pct", "vertical_utilization", "racking_height_avg_m",
          "cross_dock_ratio", "shift_hour", "day_of_week", "prev_shift_volume",
          "kpi_otd_pct", "backorder_ratio", "dock_to_stock_hours",
          "maintenance_schedule_score", "conveyor_speed_mps", "avg_package_weight_kg"],
}

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("processed")
OPEN_DIR = Path("open")
SLOTS    = 25


# ─────────────────────────────────────────────
# 피처 엔지니어링
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(layout_df, on="layout_id", how="left")
    df = df.sort_values(["scenario_id", "ID"]).reset_index(drop=True)

    df["slot_idx"]        = df.groupby("scenario_id").cumcount().astype(np.float32)
    df["layout_type_enc"] = df["layout_type"].map(LAYOUT_TYPE_MAP).astype(np.float32)

    df["pack_saturated"]      = (df["pack_utilization"] > 0.95).astype(np.float32)
    df["pack_idle_anomaly"]   = ((df["pack_utilization"] < 0.1) & (df["slot_idx"] > 5)).astype(np.float32)
    df["low_battery_alert"]   = (df["low_battery_ratio"] > 0.034).astype(np.float32)
    df["battery_critical"]    = (df["battery_mean"] < 58).astype(np.float32)
    df["density_alert"]       = (df["max_zone_density"] > 0.05).astype(np.float32)

    df["has_collision"] = (df["near_collision_15m"] > 0).astype(np.float32)
    df["has_block"]     = (df["blocked_path_15m"] > 0).astype(np.float32)
    df["has_fault"]     = (df["fault_count_15m"] > 0).astype(np.float32)
    df["has_reassign"]  = (df["task_reassign_15m"] > 0).astype(np.float32)

    total_fleet          = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    df["idle_ratio"]     = df["robot_idle"]     / (total_fleet + 1e-6)
    df["charging_ratio"] = df["robot_charging"] / (total_fleet + 1e-6)
    df["active_ratio"]   = df["robot_active"]   / (total_fleet + 1e-6)

    df["pack_x_slot"]     = df["pack_utilization"] * df["slot_idx"]
    df["order_x_slot"]    = df["order_inflow_15m"] * df["slot_idx"]
    df["inflow_x_urgent"] = df["order_inflow_15m"] * df["urgent_order_ratio"]
    df["effective_load"]  = df["order_inflow_15m"] * (1 + 0.5 * df["urgent_order_ratio"])

    df["pack_x_battery"]   = df["pack_utilization"] * df["low_battery_ratio"]
    df["pack_x_charging"]  = df["pack_utilization"] * df["charging_ratio"]
    df["inflow_x_battery"] = df["order_inflow_15m"] * df["low_battery_ratio"]
    df["cong_x_inflow"]    = df["congestion_score"] * df["order_inflow_15m"]

    df["cum_orders"]     = df.groupby("scenario_id")["order_inflow_15m"].cumsum()
    df["cum_collisions"] = df.groupby("scenario_id")["near_collision_15m"].cumsum()
    df["cum_faults"]     = df.groupby("scenario_id")["fault_count_15m"].cumsum()
    df["cum_unique_sku"] = df.groupby("scenario_id")["unique_sku_15m"].cumsum()

    for col in ["battery_mean", "low_battery_ratio", "congestion_score",
                "pack_utilization", "order_inflow_15m"]:
        df[f"{col}_isna"] = df[col].isna().astype(np.float32)

    df["robots_per_pack"]      = df["robot_total"] / (df["pack_station_count"] + 1e-6)
    df["robots_per_charger"]   = df["robot_total"] / (df["charger_count"] + 1e-6)
    df["area_per_robot"]       = df["floor_area_sqm"] / (df["robot_total"] + 1e-6)
    df["pack_per_charger"]     = df["pack_station_count"] / (df["charger_count"] + 1e-6)

    df["is_narrow"]             = (df["layout_type"] == "narrow").astype(np.float32)
    df["narrow_x_battery"]      = df["is_narrow"] * df["low_battery_ratio"]
    df["few_packs"]             = (df["pack_station_count"] < 7).astype(np.float32)
    df["few_packs_x_pack_util"] = df["few_packs"] * df["pack_utilization"]

    return df


def get_feature_cols() -> list[str]:
    """FiLM 버전: layout 수치는 FiLM encoder로 별도 처리 → 메인 피처에서 제외"""
    all_group_cols = [c for cols in GROUPS.values() for c in cols]
    engineered = [
        "slot_idx", "layout_type_enc",
        "pack_saturated", "pack_idle_anomaly", "low_battery_alert",
        "battery_critical", "density_alert",
        "has_collision", "has_block", "has_fault", "has_reassign",
        "idle_ratio", "charging_ratio", "active_ratio",
        "pack_x_slot", "order_x_slot", "inflow_x_urgent", "effective_load",
        "pack_x_battery", "pack_x_charging", "inflow_x_battery", "cong_x_inflow",
        "cum_orders", "cum_collisions", "cum_faults", "cum_unique_sku",
        "battery_mean_isna", "low_battery_ratio_isna", "congestion_score_isna",
        "pack_utilization_isna", "order_inflow_15m_isna",
        "robots_per_pack", "robots_per_charger", "area_per_robot", "pack_per_charger",
        "is_narrow", "narrow_x_battery", "few_packs", "few_packs_x_pack_util",
    ]
    return all_group_cols + engineered  # layout 수치 13개 제외 (FiLM encoder 담당)


# ─────────────────────────────────────────────
# 시나리오 단위 reshape
# ─────────────────────────────────────────────
def to_scenarios(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    layout_scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
):
    """
    (N_rows, n_features) → (N_scenarios, 25, n_features)
    layout_type, layout_nums는 시나리오당 1개 (정적 정보)

    Returns: X, layout_type, layout_nums, y, scenario_ids, scaler, layout_scaler
    """
    n_rows = len(df)
    n_scen = n_rows // SLOTS
    assert n_rows % SLOTS == 0, f"행 수({n_rows})가 {SLOTS}의 배수가 아님"

    # 메인 피처
    raw = df[feature_cols].values.astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0)

    if fit_scaler:
        scaler = StandardScaler()
        raw = scaler.fit_transform(raw)
    elif scaler is not None:
        raw = scaler.transform(raw)

    X = raw.reshape(n_scen, SLOTS, len(feature_cols))

    # layout 정보 (시나리오당 첫 슬롯 기준, 모든 슬롯이 동일)
    layout_type_arr = df["layout_type"].map(LAYOUT_TYPE_MAP).values.astype(np.int64)
    layout_type     = layout_type_arr.reshape(n_scen, SLOTS)[:, 0]  # (N_scen,)

    layout_num_raw  = df[LAYOUT_NUM_COLS].values.astype(np.float32)
    layout_num_raw  = np.nan_to_num(layout_num_raw, nan=0.0)
    if fit_scaler:
        layout_scaler = StandardScaler()
        layout_num_raw = layout_scaler.fit_transform(layout_num_raw)
    elif layout_scaler is not None:
        layout_num_raw = layout_scaler.transform(layout_num_raw)

    layout_nums = layout_num_raw.reshape(n_scen, SLOTS, len(LAYOUT_NUM_COLS))[:, 0, :]  # (N_scen, 13)

    # 타깃
    y = None
    if TARGET in df.columns:
        y = np.log1p(df[TARGET].values.astype(np.float32)).reshape(n_scen, SLOTS)

    scenario_ids = df["scenario_id"].values.reshape(n_scen, SLOTS)[:, 0]
    return X, layout_type, layout_nums, y, scenario_ids, scaler, layout_scaler


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class ScenarioDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        layout_type: np.ndarray,
        layout_nums: np.ndarray,
        y: np.ndarray | None = None,
    ):
        self.X           = torch.tensor(X,           dtype=torch.float32)
        self.layout_type = torch.tensor(layout_type, dtype=torch.long)
        self.layout_nums = torch.tensor(layout_nums, dtype=torch.float32)
        self.y           = torch.tensor(y,           dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.layout_type[idx], self.layout_nums[idx], self.y[idx]
        return self.X[idx], self.layout_type[idx], self.layout_nums[idx]


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────
class LayoutEncoder(nn.Module):
    """layout_type (범주형) + layout_nums (수치형) → d_layout 차원 조건 벡터"""
    def __init__(self, n_layout_types: int, n_num_feats: int, d_layout: int):
        super().__init__()
        type_embed_dim = 8
        self.type_embed = nn.Embedding(n_layout_types, type_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(type_embed_dim + n_num_feats, d_layout),
            nn.LayerNorm(d_layout),
            nn.GELU(),
            nn.Linear(d_layout, d_layout),
        )

    def forward(self, layout_type: torch.Tensor, layout_nums: torch.Tensor) -> torch.Tensor:
        type_emb = self.type_embed(layout_type)                       # (B, 8)
        return self.mlp(torch.cat([type_emb, layout_nums], dim=-1))   # (B, d_layout)


class FiLM(nn.Module):
    """γ(layout_emb) * x + β(layout_emb) — Conv1d 텐서 (B, C, L)에 적용"""
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.gamma = nn.Linear(d_cond, d_model)
        self.beta  = nn.Linear(d_cond, d_model)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, 25)   z: (B, d_cond)
        γ = self.gamma(z).unsqueeze(2)   # (B, d_model, 1) → 시간축 broadcast
        β = self.beta(z).unsqueeze(2)    # (B, d_model, 1)
        return γ * x + β


class CNN1DBlock(nn.Module):
    """Conv1d residual block (kernel_size=3, same padding)"""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class CNN1DFilmModel(nn.Module):
    """
    (B, 25, n_features) + layout → (B, 25)

    proj        : Conv1d(n_feat → d_model, k=1)
    blocks      : CNN1DBlock × n_blocks
    film_layers : FiLM × n_blocks  (블록마다 layout 조건 적용)
    head        : Conv1d(d_model → 1, k=1) + Softplus
    """
    def __init__(
        self,
        n_features: int,
        d_model: int   = 64,
        n_blocks: int  = 3,
        dropout: float = 0.1,
        d_layout: int  = 32,
        n_layout_types: int = len(LAYOUT_TYPE_MAP),
        n_layout_num_feats: int = len(LAYOUT_NUM_COLS),
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            CNN1DBlock(d_model, dropout) for _ in range(n_blocks)
        ])
        self.layout_encoder = LayoutEncoder(n_layout_types, n_layout_num_feats, d_layout)
        self.film_layers = nn.ModuleList([
            FiLM(d_model, d_layout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, 25, n_features)
        layout_type: torch.Tensor,  # (B,)
        layout_nums: torch.Tensor,  # (B, 13)
    ) -> torch.Tensor:              # (B, 25)

        layout_emb = self.layout_encoder(layout_type, layout_nums)  # (B, d_layout)

        x = x.permute(0, 2, 1)   # (B, n_feat, 25)
        x = self.proj(x)          # (B, d_model, 25)

        for block, film in zip(self.blocks, self.film_layers):
            x = block(x)          # (B, d_model, 25)
            x = film(x, layout_emb)  # γ·x + β

        x = self.head(x)          # (B, 1, 25)
        return x.squeeze(1)       # (B, 25)


def print_model_summary(model: CNN1DFilmModel, n_features: int) -> None:
    d_model   = model.proj[0].out_channels
    n_blocks  = len(model.blocks)
    d_layout  = model.film_layers[0].gamma.in_features
    d_half    = model.head[0].out_channels
    type_dim  = model.layout_encoder.type_embed.embedding_dim
    total     = sum(p.numel() for p in model.parameters())

    W = 64
    print("\n" + "=" * W)
    print("  모델 구조 (CNN1DFilmModel)")
    print("=" * W)
    print(f"  입력          : (B, {SLOTS}, {n_features})")
    print(f"  LayoutEncoder : Embedding(4,{type_dim}) + {len(LAYOUT_NUM_COLS)}d → {d_layout}d")
    print(f"  proj          : Conv1d({n_features}→{d_model}, k=1) + BN + GELU")
    print(f"  blocks × {n_blocks}    : CNN1DBlock (Conv1d k=3, residual, BN, GELU)")
    print(f"  FiLM   × {n_blocks}    : γ·x + β  Linear({d_layout}→{d_model}), 블록마다 적용")
    print(f"  head          : Conv1d({d_model}→{d_half}→1, k=1) + Softplus")
    print(f"  출력          : (B, {SLOTS})")
    print(f"  총 파라미터   : {total:,}")
    print("=" * W + "\n")


# ─────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    pbar = tqdm(loader, desc=f"[Epoch {epoch:3d}] Train", leave=False)
    for X, lt, ln, y in pbar:
        X, lt, ln, y = X.to(device), lt.to(device), ln.to(device), y.to(device)
        pred = model(X, lt, ln)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae  += torch.mean(torch.abs(torch.expm1(pred) - torch.expm1(y))).item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    n = len(loader)
    return total_loss / n, total_mae / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    pbar = tqdm(loader, desc=f"[Epoch {epoch:3d}]  Val ", leave=False)
    for X, lt, ln, y in pbar:
        X, lt, ln, y = X.to(device), lt.to(device), ln.to(device), y.to(device)
        pred = model(X, lt, ln)
        total_loss += criterion(pred, y).item()
        total_mae  += torch.mean(torch.abs(torch.expm1(pred) - torch.expm1(y))).item()
    n = len(loader)
    return total_loss / n, total_mae / n


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"device: {DEVICE}")

    # ── 데이터 로드 ──────────────────────────────────────
    train_df  = pd.read_csv(DATA_DIR / "train_imputed.csv")
    test_df   = pd.read_csv(DATA_DIR / "test_imputed.csv")
    layout_df = pd.read_csv(OPEN_DIR / "layout_info.csv")
    print(f"train: {train_df.shape}  test: {test_df.shape}")

    # ── 피처 엔지니어링 ───────────────────────────────────
    print("\n[피처 엔지니어링]")
    train_df     = build_features(train_df, layout_df)
    test_df      = build_features(test_df,  layout_df)
    feature_cols = get_feature_cols()
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"  메인 피처 수: {len(feature_cols)}개  (layout 수치 {len(LAYOUT_NUM_COLS)}개는 FiLM encoder 담당)")

    # ── 시나리오 단위 80/20 split ─────────────────────────
    scenarios = train_df["scenario_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(scenarios)
    cut        = int(len(scenarios) * 0.8)
    train_scen = set(scenarios[:cut])

    tr_df  = train_df[ train_df["scenario_id"].isin(train_scen)].reset_index(drop=True)
    val_df = train_df[~train_df["scenario_id"].isin(train_scen)].reset_index(drop=True)
    print(f"  train: {len(train_scen):,} scenarios  val: {len(scenarios)-cut:,} scenarios")

    # ── reshape + scaling ─────────────────────────────────
    print("\n[시나리오 reshape + 스케일링]")
    X_tr,  lt_tr,  ln_tr,  y_tr,  _, scaler, layout_scaler = to_scenarios(tr_df,   feature_cols, fit_scaler=True)
    X_val, lt_val, ln_val, y_val, _, _,      _              = to_scenarios(val_df,  feature_cols, scaler=scaler, layout_scaler=layout_scaler)
    X_te,  lt_te,  ln_te,  _,     _, _,      _              = to_scenarios(test_df, feature_cols, scaler=scaler, layout_scaler=layout_scaler)
    print(f"  X_train: {X_tr.shape}  X_val: {X_val.shape}  X_test: {X_te.shape}")

    # ── DataLoader ────────────────────────────────────────
    BATCH = 128
    train_loader = DataLoader(ScenarioDataset(X_tr,  lt_tr,  ln_tr,  y_tr),  batch_size=BATCH, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(ScenarioDataset(X_val, lt_val, ln_val, y_val), batch_size=BATCH, shuffle=False, num_workers=4)
    test_loader  = DataLoader(ScenarioDataset(X_te,  lt_te,  ln_te),         batch_size=BATCH, shuffle=False, num_workers=4)

    # ── 모델 ─────────────────────────────────────────────
    n_features = X_tr.shape[2]
    model = CNN1DFilmModel(
        n_features=n_features, d_model=32, n_blocks=2, dropout=0.1, d_layout=16,
    ).to(DEVICE)
    print_model_summary(model, n_features)

    # ── 옵티마이저 & 스케줄러 ─────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=490)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[10])
    criterion = nn.L1Loss()

    # ── 학습 루프 ─────────────────────────────────────────
    best_val_loss  = float("inf")
    patience       = 20
    no_improve_cnt = 0

    for epoch in range(1, 501):
        train_log, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        val_log,   val_mae   = evaluate(model, val_loader, criterion, DEVICE, epoch)
        scheduler.step()

        improved = val_log < best_val_loss
        if improved:
            best_val_loss  = val_log
            no_improve_cnt = 0
            torch.save(model.state_dict(), "best_model_1dcnn_film.pt")
        else:
            no_improve_cnt += 1

        mark = " *" if improved else ""
        print(f"[Epoch {epoch:3d}]  "
              f"train: log={train_log:.4f} MAE={train_mae:.2f}분  "
              f"val: log={val_log:.4f} MAE={val_mae:.2f}분  "
              f"early={no_improve_cnt}/{patience}{mark}")

        if no_improve_cnt >= patience:
            print(f"\nEarly stopping (epoch {epoch})")
            break

    print(f"\n최고 val log-MAE: {best_val_loss:.4f}")

    # ── 추론 ─────────────────────────────────────────────
    model.load_state_dict(torch.load("best_model_1dcnn_film.pt", map_location=DEVICE))
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            X, lt, ln = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            out = model(X, lt, ln)
            preds.append(np.expm1(out.cpu().numpy()))

    preds      = np.concatenate(preds, axis=0).reshape(-1)
    pred_df    = pd.DataFrame({"ID": test_df["ID"].values, TARGET: preds})
    submission = pd.read_csv(OPEN_DIR / "sample_submission.csv")
    submission = submission[["ID"]].merge(pred_df, on="ID", how="left")

    assert submission[TARGET].isnull().sum() == 0, "예측값 누락 ID 존재"
    submission.to_csv("submission_1dcnn_film.csv", index=False)
    print(f"\nsubmission_1dcnn_film.csv 저장 완료 ({len(submission):,}행)")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
