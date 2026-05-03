"""
2~3단계: 딥러닝 임베딩 추출 + LightGBM 5-Fold 앙상블

흐름:
  best_model.pt 로드
  → encode() 로 CLS 임베딩 추출  (결측치 처리된 데이터 사용)
  → 원본 피처(결측치 유지)와 concat  (LightGBM이 NaN 직접 처리)
  → GroupKFold(5) 앙상블 학습
  → submission_lgbm.csv 저장
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from model import (
    WarehouseDelayModel, WarehouseDataset, collate_fn, TARGET,
    GROUPS, LAYOUT_NUM_COLS, LAYOUT_TYPE_MAP,
)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("processed")
OPEN_DIR = Path("open")
MODEL_PT = Path("best_model.pt")
BATCH    = 512
N_FOLDS  = 5


@torch.no_grad()
def extract_embeddings(model: WarehouseDelayModel, loader, device) -> np.ndarray:
    model.eval()
    embs = []
    for batch in tqdm(loader, desc="  embedding", leave=False):
        groups      = [g.to(device) for g in batch["groups"]]
        layout_type = batch["layout_type"].to(device)
        layout_nums = batch["layout_nums"].to(device)
        embs.append(model.encode(groups, layout_type, layout_nums).cpu().numpy())
    return np.vstack(embs)


def get_raw_features(df: pd.DataFrame, layout_df: pd.DataFrame, flag_cols: list[str] | None = None):
    """원본 피처 (결측치 유지) + 결측 플래그 — LightGBM이 NaN 직접 처리

    flag_cols: 결측 플래그를 만들 컬럼 목록. None이면 train 기준으로 자동 감지.
    Returns (features, flag_cols)
    """
    merged = df.merge(layout_df, on="layout_id", how="left")
    all_group_cols = [c for cols in GROUPS.values() for c in cols]

    if flag_cols is None:
        flag_cols = [c for c in all_group_cols if merged[c].isnull().any()]

    group_feats = merged[all_group_cols].values.astype(np.float32)
    layout_num  = merged[LAYOUT_NUM_COLS].values.astype(np.float32)
    layout_type = merged["layout_type"].map(LAYOUT_TYPE_MAP).values.reshape(-1, 1).astype(np.float32)
    flags       = merged[flag_cols].isnull().values.astype(np.float32)

    return np.hstack([group_feats, layout_num, layout_type, flags]), flag_cols


def main():
    print("=" * 60)
    print(f"Stage 2~3: 임베딩 추출 + LightGBM ({N_FOLDS}-Fold 앙상블)")
    print("=" * 60)

    # ── 데이터 로딩 ───────────────────────────────────────
    train_imp = pd.read_csv(DATA_DIR / "train_imputed.csv")  # 임베딩 추출용
    test_imp  = pd.read_csv(DATA_DIR / "test_imputed.csv")
    train_raw = pd.read_csv(OPEN_DIR / "train.csv")          # LightGBM 원본 피처용
    test_raw  = pd.read_csv(OPEN_DIR / "test.csv")
    layout_df = pd.read_csv(OPEN_DIR / "layout_info.csv")
    print(f"train: {len(train_imp):,}행  test: {len(test_imp):,}행")

    # ── 모델 로드 (체크포인트에서 하이퍼파라미터 자동 추론) ──
    ckpt     = torch.load(MODEL_PT, map_location=DEVICE)
    d_model  = ckpt["cls_token"].shape[2]
    n_layers = sum(1 for k in ckpt if k.startswith("transformer_layers.") and k.endswith(".norm1.weight"))
    use_film = "film_layers.0.gamma.weight" in ckpt
    d_layout = ckpt["film_layers.0.gamma.weight"].shape[1] if use_film else 32
    model = WarehouseDelayModel(d_model=d_model, d_layout=d_layout, n_layers=n_layers, use_film=use_film).to(DEVICE)
    model.load_state_dict(ckpt)

    n_heads = model.transformer_layers[0].self_attn.num_heads
    ffn_dim = model.transformer_layers[0].linear1.out_features
    total_p = sum(p.numel() for p in model.parameters())
    W = 60
    print("\n" + "=" * W)
    film_tag = "FiLM ON" if use_film else "FiLM OFF"
    print(f"  인코더 구조  ({MODEL_PT})  [{film_tag}]")
    print("=" * W)
    print(f"  d_model   : {d_model}")
    print(f"  Transformer: {n_layers}층  n_heads={n_heads}  ffn={ffn_dim}")
    print(f"  GroupMLP   : {len(model.group_mlps)}개  (input→{d_model})")
    if use_film:
        type_dim = model.layout_encoder.type_embed.embedding_dim
        n_num    = model.layout_encoder.mlp[0].in_features - type_dim
        print(f"  LayoutEnc  : Embedding(4,{type_dim}) + {n_num}d → {d_layout}d")
        print(f"  FiLM       : {n_layers}개 (레이어마다)")
    else:
        print(f"  LayoutEnc / FiLM: 없음")
    print(f"  CLS 토큰   : nn.Parameter(1, 1, {d_model})")
    print(f"  총 파라미터: {total_p:,}")
    print("=" * W)

    # ── 임베딩 추출 (imputed 데이터 → 인코더) ────────────
    print("\n[임베딩 추출]  결측치 처리된 데이터 → 인코더 통과")
    train_ds = WarehouseDataset(train_imp, layout_df, scalers=None,             is_train=True)
    test_ds  = WarehouseDataset(test_imp,  layout_df, scalers=train_ds.scalers, is_train=False)

    kw = dict(batch_size=BATCH, collate_fn=collate_fn, num_workers=4, shuffle=False)
    train_emb = extract_embeddings(model, DataLoader(train_ds, **kw), DEVICE)
    test_emb  = extract_embeddings(model, DataLoader(test_ds,  **kw), DEVICE)
    print(f"  shape: train={train_emb.shape}  test={test_emb.shape}")

    # ── 원본 피처 (결측치 유지) + concat ──────────────────
    print("\n[원본 피처]  결측치 유지 + 결측 플래그 — LightGBM이 NaN 직접 처리")
    train_orig, flag_cols = get_raw_features(train_raw, layout_df, flag_cols=None)
    test_orig,  _         = get_raw_features(test_raw,  layout_df, flag_cols=flag_cols)
    print(f"  결측 플래그 컬럼: {len(flag_cols)}개")

    X_all  = np.hstack([train_emb, train_orig])
    X_test = np.hstack([test_emb,  test_orig])
    y_all  = train_ds.targets  # log1p
    print(f"  최종 피처 차원: {X_all.shape[1]} "
          f"(임베딩 {train_emb.shape[1]} + 원본 {train_orig.shape[1]})")

    # ── GroupKFold 5-Fold 앙상블 ──────────────────────────
    print(f"\n[LightGBM {N_FOLDS}-Fold 앙상블]  GroupKFold by scenario_id")
    groups   = train_imp["scenario_id"].values
    gkf      = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(X_all))
    test_pred_sum = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups), 1):
        print(f"\n  ── Fold {fold}/{N_FOLDS} ──")
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]

        lgbm = LGBMRegressor(
            objective="mae",
            n_estimators=3000,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.5,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42 + fold,
            verbose=-1,
        )
        lgbm.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=200),
                log_evaluation(period=200),
            ],
        )

        val_pred_fold        = lgbm.predict(X_val)
        oof_pred[val_idx]    = val_pred_fold
        test_pred_sum       += lgbm.predict(X_test) / N_FOLDS

        fold_mae = np.mean(np.abs(np.expm1(val_pred_fold) - np.expm1(y_val)))
        print(f"  Fold {fold}  best_iter={lgbm.best_iteration_}  val MAE={fold_mae:.4f}분")

    oof_mae = np.mean(np.abs(np.expm1(oof_pred) - np.expm1(y_all)))
    print(f"\nOOF MAE (전체): {oof_mae:.4f}분")

    # ── 추론 및 저장 ──────────────────────────────────────
    final_pred = np.expm1(test_pred_sum)
    pred_df    = pd.DataFrame({"ID": test_ds.ids, TARGET: final_pred})
    submission = pd.read_csv(OPEN_DIR / "sample_submission.csv")
    submission = submission[["ID"]].merge(pred_df, on="ID", how="left")

    assert submission[TARGET].isnull().sum() == 0, "예측값 누락 ID 존재"
    out_path = "submission_lgbm.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}  ({len(submission):,}행)")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
