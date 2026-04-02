import os
import matplotlib.pyplot as plt
import seaborn as sns
from eda_utils import load_data, ensure_dir, save_fig, setup_plotting


def main():
    setup_plotting()
    df = load_data()
    corr_dir = ensure_dir("Correlation_Analysis")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    plt.figure(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Pearson Correlation Heatmap")
    save_fig(os.path.join(corr_dir, "correlation_heatmap_pearson.png"))

    plt.figure(figsize=(12, 10))
    corr_spearman = df[numeric_cols].corr(method="spearman")
    sns.heatmap(corr_spearman, annot=True, cmap="vlag", fmt=".2f", linewidths=0.5)
    plt.title("Spearman Correlation Heatmap")
    save_fig(os.path.join(corr_dir, "correlation_heatmap_spearman.png"))

    corr_target = corr_matrix["quality"].drop("quality").sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=corr_target.values, y=corr_target.index, color="slateblue")
    plt.title("Pearson Correlation with Quality")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    save_fig(os.path.join(corr_dir, "correlation_with_quality.png"))

    print(f"Correlation analysis plots saved to {corr_dir}")


if __name__ == "__main__":
    main()
