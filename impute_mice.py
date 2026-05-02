"""
MICE (IterativeImputer + LightGBM) 결측치 대체
- 전체 수치형 컬럼에 단일 IterativeImputer 적용 (진짜 MICE)
- TARGET, ID 제외
- train으로 fit → train/test 모두 transform
- 결과: processed/train_imputed.csv, processed/test_imputed.csv
"""

import warnings
import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor

DATA_DIR = Path("open")
OUT_DIR  = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

MAX_ITER     = 10
N_ESTIMATORS = 200
TARGET       = "avg_delay_minutes_next_30m"


def print_missing(df: pd.DataFrame, label: str) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\n[결측 현황 - {label}]")
    if missing.empty:
        print("  결측 없음")
    else:
        pct = (missing / len(df) * 100).round(2)
        print(pd.DataFrame({"count": missing, "pct(%)": pct}).to_string())


def main():
    print("=" * 60)
    print("MICE Imputation (LightGBM) — 단일 IterativeImputer")
    print("=" * 60)

    print("\n데이터 로딩...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  train: {train.shape}, test: {test.shape}")

    print_missing(train, "train (before)")
    print_missing(test,  "test  (before)")

    # 수치형 컬럼: TARGET 및 비수치형 제외, train/test 공통 컬럼만 사용
    num_train = set(train.select_dtypes(include="number").columns) - {TARGET}
    num_test  = set(test.select_dtypes(include="number").columns)
    feat_cols = sorted(num_train & num_test)   # train/test 교집합 (TARGET 제외됨)

    print(f"\n대상 컬럼: {len(feat_cols)}개 (TARGET·비수치형 제외)")

    # 결측 있는 컬럼만 안내
    missing_train = [c for c in feat_cols if train[c].isnull().any()]
    missing_test  = [c for c in feat_cols if test[c].isnull().any()]
    print(f"결측 컬럼: train {len(missing_train)}개 / test {len(missing_test)}개")

    imputer = IterativeImputer(
        estimator=LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            n_jobs=-1,
            verbose=-1,
            random_state=42,
        ),
        max_iter=MAX_ITER,
        random_state=42,
        verbose=2,
    )

    print("\nIterativeImputer fit (train)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_arr = imputer.fit_transform(train[feat_cols].values)

    print("\nIterativeImputer transform (test)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_arr = imputer.transform(test[feat_cols].values)

    train[feat_cols] = train_arr
    test[feat_cols]  = test_arr

    print_missing(train, "train (after)")
    print_missing(test,  "test  (after)")

    train.to_csv(OUT_DIR / "train_imputed.csv", index=False)
    test.to_csv(OUT_DIR  / "test_imputed.csv",  index=False)
    print(f"\n저장 완료: {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
