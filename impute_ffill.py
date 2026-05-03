"""
시나리오 내 Forward Fill → Backward Fill 결측치 대체

흐름:
  scenario_id 그룹 내에서 ffill → bfill 순서로 적용
  (같은 시나리오 안의 인접 슬롯 값으로 채움)
  그래도 남은 결측치는 전체 컬럼 중앙값으로 마무리

  결과: processed/train_imputed.csv, processed/test_imputed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("open")
OUT_DIR  = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

TARGET = "avg_delay_minutes_next_30m"
SKIP_COLS = {"ID", "scenario_id", "layout_id", TARGET}


def print_missing(df: pd.DataFrame, label: str) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\n[결측 현황 - {label}]")
    if missing.empty:
        print("  결측 없음")
    else:
        pct = (missing / len(df) * 100).round(2)
        print(pd.DataFrame({"count": missing, "pct(%)": pct}).to_string())


def impute_ffill_bfill(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """scenario_id 그룹 내 ffill → bfill"""
    df = df.sort_values(["scenario_id", "ID"]).reset_index(drop=True)
    df[feature_cols] = (
        df.groupby("scenario_id")[feature_cols]
        .ffill()
        .bfill()
    )
    return df


def main():
    print("=" * 60)
    print("Forward Fill → Backward Fill (시나리오 내)")
    print("=" * 60)

    print("\n데이터 로딩...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  train: {train.shape}, test: {test.shape}")

    # 수치형 컬럼 중 skip 제외
    num_train = set(train.select_dtypes(include="number").columns) - SKIP_COLS
    num_test  = set(test.select_dtypes(include="number").columns)  - SKIP_COLS
    feature_cols = sorted(num_train & num_test)
    print(f"\n대상 컬럼: {len(feature_cols)}개")

    print_missing(train, "train (before)")
    print_missing(test,  "test  (before)")

    # ── ffill → bfill (시나리오 내) ───────────────────────
    print("\ntrain ffill/bfill...")
    train = impute_ffill_bfill(train, feature_cols)

    print("test  ffill/bfill...")
    test  = impute_ffill_bfill(test,  feature_cols)

    # ── 시나리오 전체가 결측인 경우: 컬럼 중앙값으로 마무리 ─
    remaining_train = [c for c in feature_cols if train[c].isnull().any()]
    remaining_test  = [c for c in feature_cols if test[c].isnull().any()]

    if remaining_train or remaining_test:
        print(f"\n잔여 결측 컬럼: train {len(remaining_train)}개 / test {len(remaining_test)}개")
        print("  → 전체 중앙값으로 대체")
        for col in set(remaining_train + remaining_test):
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col]  = test[col].fillna(median_val)

    print_missing(train, "train (after)")
    print_missing(test,  "test  (after)")

    train.to_csv(OUT_DIR / "train_imputed.csv", index=False)
    test.to_csv(OUT_DIR  / "test_imputed.csv",  index=False)
    print(f"\n저장 완료: {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
