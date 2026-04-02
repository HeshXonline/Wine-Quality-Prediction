import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda_utils import load_data, ensure_dir, save_fig, setup_plotting


def main():
    setup_plotting()
    df = load_data()
    out_dir = ensure_dir("Outlier_Analysis")

    features = [c for c in df.columns if c not in ["quality", "quality_class"]]

    for col in features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col], color="lightcoral")
        plt.title(f"Outliers: {col}")
        plt.xlabel(col)
        save_fig(os.path.join(out_dir, f"outlier_box_{col}.png"))

    q1 = df[features].quantile(0.25)
    q3 = df[features].quantile(0.75)
    iqr = q3 - q1
    outlier_counts = ((df[features] < (q1 - 1.5 * iqr)) | (df[features] > (q3 + 1.5 * iqr))).sum().sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=outlier_counts.values, y=outlier_counts.index, color="indianred")
    plt.title("Outlier Counts by Feature (IQR Method)")
    plt.xlabel("Outlier Count")
    plt.ylabel("Feature")
    save_fig(os.path.join(out_dir, "outlier_counts.png"))

    print(f"Outlier analysis plots saved to {out_dir}")


if __name__ == "__main__":
    main()
