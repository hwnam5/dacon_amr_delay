"""
1D CNN 모델 — 시나리오 단위 시계열 예측

  입력: (B, 25, n_features)  ← 시나리오 1개 = 25 time slots × n_features
  출력: (B, 25)              ← 각 슬롯의 avg_delay_minutes_next_30m 예측

  피처 (~142개):
    원본 90개 (그룹 A~J) + layout 14개 + slot_idx + 도메인 엔지니어링
    lag/diff·시나리오 메타 제외 → CNN이 시간 의존성 직접 학습
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

    df["slot_idx"]       = df.groupby("scenario_id").cumcount().astype(np.float32)
    df["layout_type_enc"]= df["layout_type"].map(LAYOUT_TYPE_MAP).astype(np.float32)

    # Tier 1: 임계점 dummy
    df["pack_saturated"]     = (df["pack_utilization"] > 0.95).astype(np.float32)
    df["pack_idle_anomaly"]  = ((df["pack_utilization"] < 0.1) & (df["slot_idx"] > 5)).astype(np.float32)
    df["low_battery_alert"]  = (df["low_battery_ratio"] > 0.034).astype(np.float32)
    df["battery_critical"]   = (df["battery_mean"] < 58).astype(np.float32)
    df["density_alert"]      = (df["max_zone_density"] > 0.05).astype(np.float32)

    # Binary indicators
    df["has_collision"] = (df["near_collision_15m"] > 0).astype(np.float32)
    df["has_block"]     = (df["blocked_path_15m"] > 0).astype(np.float32)
    df["has_fault"]     = (df["fault_count_15m"] > 0).astype(np.float32)
    df["has_reassign"]  = (df["task_reassign_15m"] > 0).astype(np.float32)

    # 비율 변수
    total_fleet          = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    df["idle_ratio"]     = df["robot_idle"]     / (total_fleet + 1e-6)
    df["charging_ratio"] = df["robot_charging"] / (total_fleet + 1e-6)
    df["active_ratio"]   = df["robot_active"]   / (total_fleet + 1e-6)

    # Tier 2: 시간 상호작용
    df["pack_x_slot"]     = df["pack_utilization"] * df["slot_idx"]
    df["order_x_slot"]    = df["order_inflow_15m"] * df["slot_idx"]
    df["inflow_x_urgent"] = df["order_inflow_15m"] * df["urgent_order_ratio"]
    df["effective_load"]  = df["order_inflow_15m"] * (1 + 0.5 * df["urgent_order_ratio"])

    # Cascading 곱항
    df["pack_x_battery"]   = df["pack_utilization"] * df["low_battery_ratio"]
    df["pack_x_charging"]  = df["pack_utilization"] * df["charging_ratio"]
    df["inflow_x_battery"] = df["order_inflow_15m"] * df["low_battery_ratio"]
    df["cong_x_inflow"]    = df["congestion_score"] * df["order_inflow_15m"]

    # 누적(Stock) 변수
    df["cum_orders"]     = df.groupby("scenario_id")["order_inflow_15m"].cumsum()
    df["cum_collisions"] = df.groupby("scenario_id")["near_collision_15m"].cumsum()
    df["cum_faults"]     = df.groupby("scenario_id")["fault_count_15m"].cumsum()
    df["cum_unique_sku"] = df.groupby("scenario_id")["unique_sku_15m"].cumsum()

    # 결측 플래그
    for col in ["battery_mean", "low_battery_ratio", "congestion_score",
                "pack_utilization", "order_inflow_15m"]:
        df[f"{col}_isna"] = df[col].isna().astype(np.float32)

    # Layout 파생 피처
    df["robots_per_pack"]     = df["robot_total"] / (df["pack_station_count"] + 1e-6)
    df["robots_per_charger"]  = df["robot_total"] / (df["charger_count"] + 1e-6)
    df["area_per_robot"]      = df["floor_area_sqm"] / (df["robot_total"] + 1e-6)
    df["pack_per_charger"]    = df["pack_station_count"] / (df["charger_count"] + 1e-6)

    # Layout type 상호작용
    df["is_narrow"]            = (df["layout_type"] == "narrow").astype(np.float32)
    df["narrow_x_battery"]     = df["is_narrow"] * df["low_battery_ratio"]
    df["few_packs"]            = (df["pack_station_count"] < 7).astype(np.float32)
    df["few_packs_x_pack_util"]= df["few_packs"] * df["pack_utilization"]

    return df


def get_feature_cols() -> list[str]:
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
    return all_group_cols + LAYOUT_NUM_COLS + engineered


# ─────────────────────────────────────────────
# 시나리오 단위 reshape
# ─────────────────────────────────────────────
def to_scenarios(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, StandardScaler]:
    """
    (N_rows, n_features) → (N_scenarios, 25, n_features)

    Returns: X, y (log1p or None), scenario_ids, scaler
    """
    n_rows = len(df)
    n_scen = n_rows // SLOTS
    n_feat = len(feature_cols)
    assert n_rows % SLOTS == 0, f"행 수({n_rows})가 {SLOTS}의 배수가 아님"

    raw = df[feature_cols].values.astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0)  # 파생 피처 잔여 NaN 처리

    if fit_scaler:
        scaler = StandardScaler()
        raw = scaler.fit_transform(raw)
    elif scaler is not None:
        raw = scaler.transform(raw)

    X = raw.reshape(n_scen, SLOTS, n_feat)

    y = None
    if TARGET in df.columns:
        y = np.log1p(df[TARGET].values.astype(np.float32)).reshape(n_scen, SLOTS)

    scenario_ids = df["scenario_id"].values.reshape(n_scen, SLOTS)[:, 0]
    return X, y, scenario_ids, scaler


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class ScenarioDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────
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


class CNN1DModel(nn.Module):
    """
    (B, 25, n_features) → (B, 25)

    proj  : Conv1d(n_feat → d_model, k=1)  피처 차원 축소
    blocks: CNN1DBlock × n_blocks           시계열 패턴 학습
    head  : Conv1d(d_model → 1, k=1)       슬롯별 스칼라 출력
    """
    def __init__(
        self,
        n_features: int,
        d_model: int   = 128,
        n_blocks: int  = 4,
        dropout: float = 0.1,
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
        self.head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)   # (B, n_feat, 25)
        x = self.proj(x)          # (B, d_model, 25)
        for block in self.blocks:
            x = block(x)          # (B, d_model, 25)
        x = self.head(x)          # (B, 1, 25)
        return x.squeeze(1)       # (B, 25)


def print_model_summary(model: CNN1DModel, n_features: int) -> None:
    d_model  = model.proj[0].out_channels
    n_blocks = len(model.blocks)
    d_half   = model.head[0].out_channels
    total    = sum(p.numel() for p in model.parameters())

    W = 60
    print("\n" + "=" * W)
    print("  모델 구조 (CNN1DModel)")
    print("=" * W)
    print(f"  입력     : (B, {SLOTS}, {n_features})")
    print(f"  proj     : Conv1d({n_features}→{d_model}, k=1) + BN + GELU")
    print(f"  blocks   : CNN1DBlock × {n_blocks}  (Conv1d k=3, residual, BN, GELU)")
    print(f"  head     : Conv1d({d_model}→{d_half}→1, k=1) + Softplus")
    print(f"  출력     : (B, {SLOTS})")
    print(f"  총 파라미터: {total:,}")
    print("=" * W + "\n")


# ─────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    pbar = tqdm(loader, desc=f"[Epoch {epoch:3d}] Train", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        pred = model(X)                         # (B, 25)
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
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
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
    train_df = build_features(train_df, layout_df)
    test_df  = build_features(test_df,  layout_df)
    feature_cols = get_feature_cols()
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"  피처 수: {len(feature_cols)}개")

    # ── 시나리오 단위 80/20 split ─────────────────────────
    scenarios = train_df["scenario_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(scenarios)
    cut        = int(len(scenarios) * 0.8)
    train_scen = set(scenarios[:cut])

    tr_df  = train_df[ train_df["scenario_id"].isin(train_scen)].reset_index(drop=True)
    val_df = train_df[~train_df["scenario_id"].isin(train_scen)].reset_index(drop=True)
    print(f"  train: {len(train_scen):,} scenarios  val: {len(scenarios)-cut:,} scenarios")

    # ── reshape + scaling ────────────────────────────────
    print("\n[시나리오 reshape + 스케일링]")
    X_train, y_train, _, scaler = to_scenarios(tr_df,  feature_cols, fit_scaler=True)
    X_val,   y_val,   _, _      = to_scenarios(val_df, feature_cols, scaler=scaler)
    X_test,  _,    test_ids, _  = to_scenarios(test_df, feature_cols, scaler=scaler)
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # ── DataLoader ────────────────────────────────────────
    BATCH = 128
    train_loader = DataLoader(ScenarioDataset(X_train, y_train), batch_size=BATCH, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(ScenarioDataset(X_val,   y_val),   batch_size=BATCH, shuffle=False, num_workers=4)
    test_loader  = DataLoader(ScenarioDataset(X_test),           batch_size=BATCH, shuffle=False, num_workers=4)

    # ── 모델 ─────────────────────────────────────────────
    n_features = X_train.shape[2]
    model = CNN1DModel(n_features=n_features, d_model=32, n_blocks=2, dropout=0.1).to(DEVICE)
    print_model_summary(model, n_features)

    # ── 옵티마이저 & 스케줄러 ─────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
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
            torch.save(model.state_dict(), "best_model_1dcnn.pt")
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
    model.load_state_dict(torch.load("best_model_1dcnn.pt", map_location=DEVICE))
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            X = batch.to(DEVICE)
            out = model(X)                        # (B, 25)
            preds.append(np.expm1(out.cpu().numpy()))

    preds = np.concatenate(preds, axis=0).reshape(-1)  # (N_test_rows,)

    # ID 복원: test_df는 이미 scenario_id, ID 순으로 정렬됨
    pred_df    = pd.DataFrame({"ID": test_df["ID"].values, TARGET: preds})
    submission = pd.read_csv(OPEN_DIR / "sample_submission.csv")
    submission = submission[["ID"]].merge(pred_df, on="ID", how="left")

    assert submission[TARGET].isnull().sum() == 0, "예측값 누락 ID 존재"
    submission.to_csv("submission_1dcnn.csv", index=False)
    print(f"\nsubmission_1dcnn.csv 저장 완료 ({len(submission):,}행)")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
