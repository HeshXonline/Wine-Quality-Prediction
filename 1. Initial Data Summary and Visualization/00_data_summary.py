import os
import numpy as np
import pandas as pd
from eda_utils import load_data, ensure_dir


def main():
    df = load_data()
    summary_dir = ensure_dir("Summary")
    summary_path = os.path.join(summary_dir, "data_summary_report.txt")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DEEP DATA SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1) DATASET SHAPE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")

        f.write("2) DATA TYPES\n")
        f.write("-" * 40 + "\n")
        f.write(df.dtypes.to_string() + "\n\n")

        f.write("3) MISSING VALUES\n")
        f.write("-" * 40 + "\n")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({"missing": missing, "missing_pct": missing_pct})
        f.write(missing_df.to_string() + "\n\n")

        f.write("4) DUPLICATE ROWS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")

        f.write("5) TARGET DISTRIBUTION (QUALITY)\n")
        f.write("-" * 40 + "\n")
        q_counts = df["quality"].value_counts().sort_index()
        q_pct = (q_counts / len(df) * 100).round(2)
        q_df = pd.DataFrame({"count": q_counts, "pct": q_pct})
        f.write(q_df.to_string() + "\n\n")

        f.write("6) TARGET DISTRIBUTION (QUALITY_CLASS)\n")
        f.write("-" * 40 + "\n")
        qc_counts = df["quality_class"].value_counts()
        qc_pct = (qc_counts / len(df) * 100).round(2)
        qc_df = pd.DataFrame({"count": qc_counts, "pct": qc_pct})
        f.write(qc_df.to_string() + "\n\n")

        f.write("7) DESCRIPTIVE STATISTICS\n")
        f.write("-" * 40 + "\n")
        desc = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        desc["skewness"] = df[numeric_cols].skew()
        desc["kurtosis"] = df[numeric_cols].kurtosis()
        f.write(desc.to_string() + "\n\n")

        f.write("8) ZERO COUNTS PER FEATURE\n")
        f.write("-" * 40 + "\n")
        zeros = (df[numeric_cols] == 0).sum().sort_values(ascending=False)
        f.write(zeros.to_string() + "\n\n")

        f.write("9) IQR OUTLIER COUNTS\n")
        f.write("-" * 40 + "\n")
        q1 = df[numeric_cols].quantile(0.25)
        q3 = df[numeric_cols].quantile(0.75)
        iqr = q3 - q1
        outlier_counts = ((df[numeric_cols] < (q1 - 1.5 * iqr)) | (df[numeric_cols] > (q3 + 1.5 * iqr))).sum()
        outlier_pct = (outlier_counts / len(df) * 100).round(2)
        outlier_df = pd.DataFrame({"outliers": outlier_counts, "outlier_pct": outlier_pct})
        f.write(outlier_df.to_string() + "\n\n")

        f.write("10) CORRELATION WITH QUALITY (PEARSON)\n")
        f.write("-" * 40 + "\n")
        corr_pearson = df[numeric_cols].corr()["quality"].sort_values(ascending=False)
        f.write(corr_pearson.to_string() + "\n\n")

        f.write("11) CORRELATION WITH QUALITY (SPEARMAN)\n")
        f.write("-" * 40 + "\n")
        corr_spearman = df[numeric_cols].corr(method="spearman")["quality"].sort_values(ascending=False)
        f.write(corr_spearman.to_string() + "\n\n")

        f.write("12) TOP FEATURE PAIRS BY ABS CORRELATION\n")
        f.write("-" * 40 + "\n")
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        pairs = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"})
            .sort_values("abs_corr", ascending=False)
        )
        f.write(pairs.head(20).to_string(index=False) + "\n\n")

    print(f"Summary report saved to {summary_path}")


if __name__ == "__main__":
    main()
