"""
Smart Storage EDA Script
- train / test / layout_info 각각 EDA 수행
- 결과물은 eda_output/{train, test, layout_info}/ 폴더에 저장
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 100

DATA_DIR = Path("open")


# ─────────────────────────────────────────────
# stdout → 터미널 + 파일 동시 출력
# ─────────────────────────────────────────────
class Tee:
    def __init__(self, filepath):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# ─────────────────────────────────────────────
# 1. 데이터 기본 파악
# ─────────────────────────────────────────────
def section1_basic(df: pd.DataFrame, out: Path) -> None:
    print("\n" + "="*60)
    print("1. 데이터 기본 파악")
    print("="*60)

    print(f"\n[Shape] {df.shape[0]:,} rows × {df.shape[1]:,} cols")

    print("\n[dtypes]")
    for dtype, cnt in df.dtypes.value_counts().items():
        print(f"  {str(dtype):<12}: {cnt}개")

    print("\n[Head]")
    print(df.head(3).to_string())

    print("\n[Tail]")
    print(df.tail(3).to_string())

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct(%)": missing_pct,
    }).query("missing_count > 0").sort_values("missing_pct(%)", ascending=False)

    print(f"\n[결측값] 결측 있는 컬럼: {len(missing_df)}개")
    print(missing_df.to_string())

    if len(missing_df) > 0:
        fig, ax = plt.subplots(figsize=(12, max(4, len(missing_df) * 0.4)))
        missing_top = missing_df.head(40)
        ax.barh(missing_top.index, missing_top["missing_pct(%)"], color="tomato")
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values by Column")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(out / "1_missing_values.png")
        plt.close()

    dup_count = df.duplicated().sum()
    print(f"\n[중복 행] {dup_count:,}개")


# ─────────────────────────────────────────────
# 2. 기술 통계
# ─────────────────────────────────────────────
def section2_stats(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("2. 기술 통계")
    print("="*60)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        print(f"\n수치형 컬럼 ({len(num_cols)}개):")
        desc = df[num_cols].describe().T
        desc["skew"] = df[num_cols].skew().round(3)
        desc["kurt"] = df[num_cols].kurt().round(3)
        print(desc.to_string())

    if cat_cols:
        print(f"\n범주형 컬럼 ({len(cat_cols)}개):")
        for col in cat_cols:
            print(f"\n  [{col}] 고유값: {df[col].nunique()}개")
            print(df[col].value_counts().head(10).to_string())


# ─────────────────────────────────────────────
# 3. 분포 시각화
# ─────────────────────────────────────────────
def section3_distribution(df: pd.DataFrame, out: Path) -> None:
    print("\n" + "="*60)
    print("3. 분포 시각화")
    print("="*60)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        n_cols = 4
        n_rows = (len(num_cols) + n_cols - 1) // n_cols

        # 히스토그램 + KDE
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        for i, col in enumerate(num_cols):
            ax = axes[i]
            data = df[col].dropna()
            ax.hist(data, bins=40, color="steelblue", alpha=0.7, density=True)
            try:
                data.plot.kde(ax=ax, color="darkblue", linewidth=1.5)
            except Exception:
                pass
            ax.set_title(col, fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Numeric Distributions (Histogram + KDE)", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out / "3_hist_kde.png", bbox_inches="tight")
        plt.close()

        # 박스플롯
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        for i, col in enumerate(num_cols):
            ax = axes[i]
            ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor="lightblue"))
            ax.set_title(col, fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Boxplots (Outlier Detection)", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(out / "3_boxplot.png", bbox_inches="tight")
        plt.close()

        # IQR 이상치 요약
        print("\n[이상치 요약] IQR 기준:")
        rows = []
        for col in num_cols:
            data = df[col].dropna()
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            n_out = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
            pct = n_out / len(data) * 100
            if pct > 0:
                rows.append({"col": col, "n_outlier": n_out, "pct(%)": round(pct, 2)})
        if rows:
            print(pd.DataFrame(rows).sort_values("pct(%)", ascending=False).to_string(index=False))

    if cat_cols:
        n_cat = len(cat_cols)
        fig, axes = plt.subplots(1, n_cat, figsize=(n_cat * 5, 4))
        axes = [axes] if n_cat == 1 else axes
        for ax, col in zip(axes, cat_cols):
            df[col].value_counts().head(20).plot.bar(ax=ax, color="coral")
            ax.set_title(col, fontsize=9)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        plt.tight_layout()
        plt.savefig(out / "3_categorical.png", bbox_inches="tight")
        plt.close()

    print(f"  → 그래프 저장: {out}/")


# ─────────────────────────────────────────────
# 4. 관계 분석
# ─────────────────────────────────────────────
def section4_relationship(df: pd.DataFrame, out: Path, target_col: str | None = None) -> None:
    print("\n" + "="*60)
    print("4. 관계 분석")
    print("="*60)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        print("  수치형 컬럼이 2개 미만 — 관계 분석 생략")
        return

    corr = df[num_cols].corr()

    # 상관관계 히트맵
    fig, ax = plt.subplots(figsize=(max(12, len(num_cols) * 0.5),
                                    max(10, len(num_cols) * 0.5)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                annot=len(num_cols) <= 20, fmt=".2f",
                linewidths=0.3, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.6}, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "4_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # 타깃 상관관계
    if target_col and target_col in df.columns:
        target_corr = corr[target_col].drop(target_col, errors="ignore") \
                                       .abs().sort_values(ascending=False)
        print(f"\n[타깃 '{target_col}'과 상관관계 Top 20]")
        print(target_corr.head(20).to_string())

        # 산점도 (Top 8)
        top_feats = target_corr.head(8).index.tolist()
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()
        for i, feat in enumerate(top_feats):
            ax = axes[i]
            sample = df[[feat, target_col]].dropna().sample(
                min(3000, len(df)), random_state=42)
            ax.scatter(sample[feat], sample[target_col],
                       alpha=0.3, s=10, color="steelblue")
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel(target_col, fontsize=8)
            ax.set_title(f"r={corr.loc[feat, target_col]:.3f}", fontsize=9)
        plt.suptitle(f"Top Features vs {target_col}", fontsize=12)
        plt.tight_layout()
        plt.savefig(out / "4_scatter_target.png", bbox_inches="tight")
        plt.close()

        # 타깃 상관 막대그래프
        top20 = corr[target_col].drop(target_col, errors="ignore") \
                                  .sort_values(key=abs, ascending=False).head(20)
        colors = ["tomato" if v < 0 else "steelblue" for v in top20]
        fig, ax = plt.subplots(figsize=(10, 6))
        top20.plot.barh(ax=ax, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Top 20 Correlations with '{target_col}'")
        plt.tight_layout()
        plt.savefig(out / "4_target_corr_bar.png", bbox_inches="tight")
        plt.close()

        pair_cols = target_corr.head(5).index.tolist() + [target_col]
    else:
        pair_cols = num_cols[:6]

    # pairplot
    sample_pp = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)
    g = sns.pairplot(sample_pp, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10})
    g.figure.suptitle("Pairplot", y=1.01, fontsize=12)
    g.figure.savefig(out / "4_pairplot.png", bbox_inches="tight")
    plt.close()

    print(f"  → 그래프 저장: {out}/")


# ─────────────────────────────────────────────
# 5. 시계열 / 그룹 분석
# ─────────────────────────────────────────────
def section5_group_time(
    df: pd.DataFrame,
    out: Path,
    target_col: str | None = None,
    time_col: str | None = None,
    group_col: str | None = None,
) -> None:
    print("\n" + "="*60)
    print("5. 시계열/그룹 분석")
    print("="*60)

    if not target_col or target_col not in df.columns:
        print("  타깃 컬럼 없음 — 그룹/시계열 분석 생략")
        return

    # 시계열 추세
    if time_col and time_col in df.columns:
        trend = df.groupby(time_col)[target_col].agg(["mean", "median", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(trend[time_col], trend["mean"], label="mean", marker="o")
        ax.plot(trend[time_col], trend["median"], label="median", marker="s", linestyle="--")
        ax.fill_between(trend[time_col],
                        trend["mean"] - trend["std"],
                        trend["mean"] + trend["std"],
                        alpha=0.2, label="±1 std")
        ax.set_xlabel(time_col)
        ax.set_ylabel(target_col)
        ax.set_title(f"{target_col} Trend by {time_col}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / "5_time_trend.png")
        plt.close()
        print(f"\n[{time_col}별 추세]")
        print(trend.to_string(index=False))

    # 그룹 박스플롯
    if group_col and group_col in df.columns:
        group_stats = df.groupby(group_col)[target_col].agg(
            ["count", "mean", "std", "median"]).round(3).sort_values("mean", ascending=False)
        print(f"\n[그룹별 '{target_col}'] by {group_col}:")
        print(group_stats.head(20).to_string())

        top_groups = group_stats.head(20).index.tolist()
        group_data = [df[df[group_col] == g][target_col].dropna().values
                      for g in top_groups]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.boxplot(group_data, labels=top_groups, patch_artist=True,
                   boxprops=dict(facecolor="lightgreen"))
        ax.set_xlabel(group_col)
        ax.set_ylabel(target_col)
        ax.set_title(f"{target_col} Distribution by {group_col} (Top 20 by mean)")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        plt.tight_layout()
        plt.savefig(out / "5_group_boxplot.png")
        plt.close()

    # 요일별 분석
    if "day_of_week" in df.columns:
        dow_stats = df.groupby("day_of_week")[target_col].agg(["mean", "std", "count"])
        print(f"\n[요일별 '{target_col}']")
        print(dow_stats.to_string())
        fig, ax = plt.subplots(figsize=(8, 4))
        dow_stats["mean"].plot.bar(ax=ax, color="mediumpurple", yerr=dow_stats["std"])
        ax.set_title(f"Mean {target_col} by Day of Week")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig(out / "5_dow_bar.png")
        plt.close()

    print(f"  → 그래프 저장: {out}/")


# ─────────────────────────────────────────────
# EDA 실행 (데이터셋별 설정 적용)
# ─────────────────────────────────────────────
def run_eda(df: pd.DataFrame, name: str, target_col=None, time_col=None, group_col=None):
    out = Path("eda_output") / name
    out.mkdir(parents=True, exist_ok=True)

    tee = Tee(out / "eda_report.txt")
    try:
        print(f"\n{'#'*60}")
        print(f"# EDA: {name}  ({df.shape[0]:,} rows × {df.shape[1]:,} cols)")
        print(f"{'#'*60}")

        section1_basic(df, out)
        section2_stats(df)
        section3_distribution(df, out)
        section4_relationship(df, out, target_col=target_col)
        section5_group_time(df, out, target_col=target_col,
                            time_col=time_col, group_col=group_col)

        print(f"\n{'='*60}")
        print(f"완료. 결과물: {out.resolve()}/")
        print("="*60)
    finally:
        tee.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=== Smart Storage EDA ===")

    # train
    print("\n[1/5] train.csv 로딩...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    run_eda(train, name="train",
            target_col="avg_delay_minutes_next_30m",
            time_col="shift_hour",
            group_col="layout_id")

    # test
    print("\n[2/5] test.csv 로딩...")
    test = pd.read_csv(DATA_DIR / "test.csv")
    run_eda(test, name="test",
            target_col=None,
            time_col="shift_hour",
            group_col="layout_id")

    # layout_info
    print("\n[3/5] layout_info.csv 로딩...")
    layout = pd.read_csv(DATA_DIR / "layout_info.csv")
    run_eda(layout, name="layout_info",
            target_col=None,
            time_col=None,
            group_col="layout_type")

    # train_imputed
    imputed_dir = Path("processed")
    print("\n[4/5] train_imputed.csv 로딩...")
    train_imp = pd.read_csv(imputed_dir / "train_imputed.csv")
    run_eda(train_imp, name="train_imputed",
            target_col="avg_delay_minutes_next_30m",
            time_col="shift_hour",
            group_col="layout_id")

    # test_imputed
    print("\n[5/5] test_imputed.csv 로딩...")
    test_imp = pd.read_csv(imputed_dir / "test_imputed.csv")
    run_eda(test_imp, name="test_imputed",
            target_col=None,
            time_col="shift_hour",
            group_col="layout_id")


if __name__ == "__main__":
    main()
