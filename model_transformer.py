"""
[그룹A] → [MLP] → 토큰A ─┐
[그룹B] → [MLP] → 토큰B ─┤
...                       ┼→ [FiLM] → [Transformer] → [FiLM] → [Head]
[그룹J] → [MLP] → 토큰J ─┘     ↑                        ↑
                          [layout_encoder]          [layout_encoder]

Dataset: train.csv + layout_info.csv를 layout_id 기준으로 merge
  → 그룹 A~J 피처: 각 GroupMLP 입력
  → layout 피처:   LayoutEncoder 입력 → FiLM γ, β → 토큰 아핀 변환
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─────────────────────────────────────────────
# 그룹 정의 (PDF A~J)
# ─────────────────────────────────────────────
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
GROUP_COLS  = list(GROUPS.values())
GROUP_SIZES = [len(c) for c in GROUP_COLS]

TARGET = "avg_delay_minutes_next_30m"

# layout_info에서 가져오는 피처
LAYOUT_TYPE_MAP = {"narrow": 0, "grid": 1, "hybrid": 2, "hub_spoke": 3}
LAYOUT_NUM_COLS = [
    "aisle_width_avg", "intersection_count", "one_way_ratio",
    "pack_station_count", "charger_count", "layout_compactness",
    "zone_dispersion", "robot_total", "building_age_years",
    "floor_area_sqm", "ceiling_height_m", "fire_sprinkler_count",
    "emergency_exit_count",
]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class WarehouseDataset(Dataset):
    """
    train/test.csv + layout_info.csv를 layout_id로 merge한 뒤
    - 그룹 A~J 피처 → group_arrays  (GroupMLP 입력)
    - layout 피처    → layout_type, layout_nums  (LayoutEncoder 입력)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        layout_df: pd.DataFrame,
        scalers: dict | None = None,
        is_train: bool = True,
    ):
        # layout_id 기준으로 layout_info merge
        df = df.merge(layout_df, on="layout_id", how="left")

        self.scalers = scalers or {}

        # 그룹별 피처 스케일링
        self.group_arrays = []
        for name, cols in GROUPS.items():
            cols_exist = [c for c in cols if c in df.columns]
            arr = df[cols_exist].values.astype(np.float32)
            if name not in self.scalers:
                sc = StandardScaler()
                arr = sc.fit_transform(arr)
                self.scalers[name] = sc
            else:
                arr = self.scalers[name].transform(arr)
            self.group_arrays.append(arr)

        # layout 수치 피처 스케일링
        layout_num = df[LAYOUT_NUM_COLS].values.astype(np.float32)
        if "layout_num" not in self.scalers:
            sc = StandardScaler()
            layout_num = sc.fit_transform(layout_num)
            self.scalers["layout_num"] = sc
        else:
            layout_num = self.scalers["layout_num"].transform(layout_num)
        self.layout_nums  = layout_num

        # layout_type → 정수
        self.layout_types = df["layout_type"].map(LAYOUT_TYPE_MAP).values.astype(np.int64)

        # target
        self.targets = None
        if is_train and TARGET in df.columns:
            self.targets = np.log1p(df[TARGET].values.astype(np.float32))

        self.ids = df["ID"].values if "ID" in df.columns else None

    def __len__(self) -> int:
        return len(self.layout_types)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "groups":      [torch.tensor(arr[idx]) for arr in self.group_arrays],
            "layout_type": torch.tensor(self.layout_types[idx]),
            "layout_nums": torch.tensor(self.layout_nums[idx]),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx])
        return item


def collate_fn(batch: list[dict]) -> dict:
    groups = [
        torch.stack([b["groups"][i] for b in batch])
        for i in range(len(batch[0]["groups"]))
    ]
    out = {
        "groups":      groups,
        "layout_type": torch.stack([b["layout_type"] for b in batch]),
        "layout_nums": torch.stack([b["layout_nums"]  for b in batch]),
    }
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch])
    return out


# ─────────────────────────────────────────────
# 모델 블록
# ─────────────────────────────────────────────
class GroupMLP(nn.Module):
    """그룹 피처 → 토큰 (d_model 차원)"""
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayoutEncoder(nn.Module):
    """
    merge된 layout 피처 → layout 임베딩
    layout_type (범주형 embed) + layout_nums (수치형) → MLP → d_layout
    """
    def __init__(self, n_layout_types: int, n_num_feats: int, d_layout: int):
        super().__init__()
        type_embed_dim = 8
        in_dim = type_embed_dim + n_num_feats   # 8 + 13 = 21
        self.type_embed = nn.Embedding(n_layout_types, type_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_layout),
            nn.LayerNorm(d_layout),
            nn.GELU(),
            nn.Linear(d_layout, d_layout),
        )

    def forward(self, layout_type: torch.Tensor, layout_nums: torch.Tensor) -> torch.Tensor:
        # layout_type: (B,)   layout_nums: (B, n_num_feats)
        type_emb = self.type_embed(layout_type)                        # (B, 8)
        return self.mlp(torch.cat([type_emb, layout_nums], dim=-1))    # (B, d_layout)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: output = γ(z) * x + β(z)"""
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.gamma = nn.Linear(d_cond, d_model)
        self.beta  = nn.Linear(d_cond, d_model)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tokens, d_model)   z: (B, d_cond)
        γ = self.gamma(z).unsqueeze(1)   # (B, 1, d_model)
        β = self.beta(z).unsqueeze(1)    # (B, 1, d_model)
        return γ * x + β


class WarehouseDelayModel(nn.Module):
    def __init__(
        self,
        group_sizes: list[int] = GROUP_SIZES,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        d_layout: int = 32,
        n_layout_types: int = len(LAYOUT_TYPE_MAP),
        n_layout_num_feats: int = len(LAYOUT_NUM_COLS),
        use_film: bool = False,
    ):
        super().__init__()
        self.use_film = use_film

        # 그룹별 MLP
        self.group_mlps = nn.ModuleList([
            GroupMLP(size, d_model) for size in group_sizes
        ])

        # layout 피처 → layout 임베딩 + FiLM (use_film=True일 때만 생성)
        if use_film:
            self.layout_encoder = LayoutEncoder(n_layout_types, n_layout_num_feats, d_layout)
            self.film_layers = nn.ModuleList([
                FiLM(d_model, d_layout) for _ in range(n_layers)
            ])

        # CLS 토큰 (학습 가능 파라미터)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer 레이어
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # Head: CLS 토큰 → regression
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        group_inputs: list[torch.Tensor],  # [n_groups] × (B, group_size)
        layout_type: torch.Tensor,         # (B,)
        layout_nums: torch.Tensor,         # (B, n_layout_num_feats)
    ) -> torch.Tensor:                     # (B,)

        # 그룹 피처 → 토큰
        tokens = torch.stack(
            [mlp(x) for mlp, x in zip(self.group_mlps, group_inputs)],
            dim=1,
        )  # (B, n_groups, d_model)

        # CLS 토큰 prepend
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, 1+n_groups, d_model)

        if self.use_film:
            layout_emb = self.layout_encoder(layout_type, layout_nums)
            for layer, film in zip(self.transformer_layers, self.film_layers):
                tokens = layer(tokens)
                tokens = film(tokens, layout_emb)
        else:
            for layer in self.transformer_layers:
                tokens = layer(tokens)

        out = tokens[:, 0]                 # CLS 토큰만 추출 (B, d_model)
        return self.head(out).squeeze(-1)  # (B,)

    def encode(
        self,
        group_inputs: list[torch.Tensor],
        layout_type: torch.Tensor,
        layout_nums: torch.Tensor,
    ) -> torch.Tensor:                     # (B, d_model)
        """Head 직전 CLS 임베딩 반환 (LightGBM 등 downstream 용)"""
        tokens = torch.stack(
            [mlp(x) for mlp, x in zip(self.group_mlps, group_inputs)], dim=1
        )
        B   = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        if self.use_film:
            layout_emb = self.layout_encoder(layout_type, layout_nums)
            for layer, film in zip(self.transformer_layers, self.film_layers):
                tokens = layer(tokens)
                tokens = film(tokens, layout_emb)
        else:
            for layer in self.transformer_layers:
                tokens = layer(tokens)
        return tokens[:, 0]               # (B, d_model)


# ─────────────────────────────────────────────
# 모델 구조 출력
# ─────────────────────────────────────────────
def print_model_summary(model: WarehouseDelayModel) -> None:
    d_model   = model.head[1].in_features
    d_model_h = model.head[1].out_features
    n_heads   = model.transformer_layers[0].self_attn.num_heads
    n_layers  = len(model.transformer_layers)
    ffn_dim   = model.transformer_layers[0].linear1.out_features
    dropout   = model.transformer_layers[0].dropout.p

    W = 64
    print("\n" + "=" * W)
    film_tag = "FiLM ON" if model.use_film else "FiLM OFF"
    print(f"  모델 구조 (WarehouseDelayModel  [{film_tag}])")
    print("=" * W)

    print(f"\n[GroupMLP × {len(model.group_mlps)}개]  →  각 그룹 피처를 {d_model}차원 토큰으로 변환")
    for name, mlp in zip(GROUPS.keys(), model.group_mlps):
        in_d = mlp.net[0].in_features
        print(f"  그룹 {name} ({in_d:2d}d): {in_d}→{d_model}")

    if model.use_film:
        d_layout = model.film_layers[0].gamma.in_features
        type_dim = model.layout_encoder.type_embed.embedding_dim
        n_num    = model.layout_encoder.mlp[0].in_features - type_dim
        n_types  = model.layout_encoder.type_embed.num_embeddings
        lm       = model.layout_encoder.mlp
        print(f"\n[LayoutEncoder]  →  layout 조건 벡터 ({d_layout}d)")
        print(f"  Embedding({n_types}, {type_dim})  +  layout_nums ({n_num}d)")
        print(f"  {lm[0].in_features}→{lm[0].out_features}→{lm[3].out_features}")
        tf_label = f"[TransformerLayer + FiLM]  ×{n_layers}층"
        film_detail = f"  각 레이어: TransformerEncoderLayer(Pre-LN) → FiLM(γ,β: Linear({d_layout}→{d_model}))"
    else:
        print(f"\n[LayoutEncoder / FiLM]  사용 안 함")
        tf_label = f"[TransformerLayer]  ×{n_layers}층  (FiLM 없음)"
        film_detail = f"  각 레이어: TransformerEncoderLayer(Pre-LN)"

    print(f"\n[CLS Token]  nn.Parameter(1, 1, {d_model})")

    print(f"\n{tf_label}")
    print(f"  d_model={d_model},  n_heads={n_heads},  ffn={ffn_dim},  dropout={dropout}")
    print(f"  시퀀스 길이: 1(CLS) + {len(model.group_mlps)}(그룹) = {1+len(model.group_mlps)}")
    print(film_detail)

    print(f"\n[Head]  CLS 토큰 → 스칼라 예측")
    print(f"  LayerNorm({d_model}) → Linear({d_model}→{d_model_h}) → GELU "
          f"→ Dropout → Linear({d_model_h}→1) → Softplus")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n총 파라미터:     {total:>10,}")
    print(f"학습 파라미터:   {trainable:>10,}")
    print("=" * W + "\n")


# ─────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss     = 0.0
    total_mae_orig = 0.0
    mae_fn = nn.L1Loss()
    pbar = tqdm(loader, desc=f"[Epoch {epoch:3d}] Train", leave=False)
    for batch in pbar:
        groups      = [g.to(device) for g in batch["groups"]]
        layout_type = batch["layout_type"].to(device)
        layout_nums = batch["layout_nums"].to(device)
        target      = batch["target"].to(device)

        pred = model(groups, layout_type, layout_nums)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss     += loss.item()
        total_mae_orig += mae_fn(torch.expm1(pred), torch.expm1(target)).item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return total_loss / n, total_mae_orig / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss     = 0.0
    total_mae_orig = 0.0
    mae_fn = nn.L1Loss()
    pbar = tqdm(loader, desc=f"[Epoch {epoch:3d}]  Val ", leave=False)
    for batch in pbar:
        groups      = [g.to(device) for g in batch["groups"]]
        layout_type = batch["layout_type"].to(device)
        layout_nums = batch["layout_nums"].to(device)
        target      = batch["target"].to(device)
        pred        = model(groups, layout_type, layout_nums)
        loss        = criterion(pred, target)
        total_loss     += loss.item()
        total_mae_orig += mae_fn(torch.expm1(pred), torch.expm1(target)).item()
        pbar.set_postfix(mae=f"{loss.item():.4f}")

    n = len(loader)
    return total_loss / n, total_mae_orig / n


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_dir  = Path("processed")
    train_df  = pd.read_csv(data_dir / "train_imputed.csv")
    test_df   = pd.read_csv(data_dir / "test_imputed.csv")
    layout_df = pd.read_csv(Path("open") / "layout_info.csv")

    # scenario 기준 80/20 split (같은 scenario가 train/val에 섞이지 않도록)
    scenarios = train_df["scenario_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(scenarios)
    cut          = int(len(scenarios) * 0.8)
    train_scen   = set(scenarios[:cut])
    val_df       = train_df[~train_df["scenario_id"].isin(train_scen)].reset_index(drop=True)
    train_df     = train_df[ train_df["scenario_id"].isin(train_scen)].reset_index(drop=True)
    print(f"train: {len(train_df):,}행 ({len(train_scen):,} scenarios)  "
          f"val: {len(val_df):,}행 ({len(scenarios)-len(train_scen):,} scenarios)")

    train_ds = WarehouseDataset(train_df, layout_df, scalers=None,             is_train=True)
    val_ds   = WarehouseDataset(val_df,   layout_df, scalers=train_ds.scalers, is_train=True)
    test_ds  = WarehouseDataset(test_df,  layout_df, scalers=train_ds.scalers, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,  collate_fn=collate_fn, num_workers=8)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, collate_fn=collate_fn, num_workers=8)

    model = WarehouseDelayModel().to(device)
    print_model_summary(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=490)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])
    criterion = nn.L1Loss()

    best_val_loss  = float("inf")
    patience       = 10
    no_improve_cnt = 0

    for epoch in range(1, 501):
        train_log, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_log,   val_mae   = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        improved = val_log < best_val_loss
        if improved:
            best_val_loss  = val_log
            no_improve_cnt = 0
            torch.save(model.state_dict(), "best_model.pt")
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

    print(f"\n최고 val MAE: {best_val_loss:.4f}")

    # 추론
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            groups      = [g.to(device) for g in batch["groups"]]
            layout_type = batch["layout_type"].to(device)
            layout_nums = batch["layout_nums"].to(device)
            preds.extend(model(groups, layout_type, layout_nums).cpu().numpy())

    pred_df    = pd.DataFrame({"ID": test_ds.ids, TARGET: np.expm1(np.array(preds))})
    submission = pd.read_csv(Path("open") / "sample_submission.csv")
    submission = submission[["ID"]].merge(pred_df, on="ID", how="left")

    assert submission[TARGET].isnull().sum() == 0, "예측값 누락 ID 존재"
    submission.to_csv("submission.csv", index=False)
    print(f"submission.csv 저장 완료 ({len(submission):,}행)")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
