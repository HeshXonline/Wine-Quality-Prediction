import os
import matplotlib.pyplot as plt
import seaborn as sns
from eda_utils import load_data, ensure_dir, save_fig, setup_plotting


def main():
    setup_plotting()
    df = load_data()
    bi_dir = ensure_dir("Bivariate_Relationships")

    features = [c for c in df.columns if c not in ["quality", "quality_class"]]

    for col in features:
        plt.figure(figsize=(8, 5))
        sns.stripplot(x="quality", y=col, data=df, jitter=0.25, alpha=0.5, color="steelblue")
        plt.title(f"{col} vs Quality (Stripplot)")
        plt.xlabel("Quality")
        plt.ylabel(col)
        save_fig(os.path.join(bi_dir, f"strip_{col}_vs_quality.png"))

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="quality", y=col, data=df, hue="quality", palette="Set2", legend=False)
        plt.title(f"{col} vs Quality (Boxplot)")
        plt.xlabel("Quality")
        plt.ylabel(col)
        save_fig(os.path.join(bi_dir, f"box_{col}_vs_quality.png"))

        plt.figure(figsize=(8, 5))
        sns.boxplot(
            x="quality_class",
            y=col,
            data=df,
            order=["low", "medium", "high"],
            hue="quality_class",
            palette="Set3",
            legend=False,
        )
        plt.title(f"{col} vs Quality Class (Boxplot)")
        plt.xlabel("Quality Class")
        plt.ylabel(col)
        save_fig(os.path.join(bi_dir, f"box_{col}_vs_quality_class.png"))

    print(f"Bivariate plots saved to {bi_dir}")


if __name__ == "__main__":
    main()
