import os
import matplotlib.pyplot as plt
import seaborn as sns
from eda_utils import load_data, ensure_dir, save_fig, setup_plotting


def main():
    setup_plotting()
    df = load_data()
    target_dir = ensure_dir("Target_Analysis")

    plt.figure(figsize=(8, 5))
    sns.countplot(x="quality", data=df, color="steelblue")
    plt.title("Quality Distribution (Counts)")
    plt.xlabel("Quality")
    plt.ylabel("Count")
    save_fig(os.path.join(target_dir, "quality_counts.png"))

    quality_pct = df["quality"].value_counts(normalize=True).sort_index() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=quality_pct.index, y=quality_pct.values, color="seagreen")
    plt.title("Quality Distribution (Percent)")
    plt.xlabel("Quality")
    plt.ylabel("Percent")
    save_fig(os.path.join(target_dir, "quality_percent.png"))

    plt.figure(figsize=(8, 5))
    sns.countplot(x="quality_class", data=df, order=["low", "medium", "high"], color="mediumpurple")
    plt.title("Quality Class Distribution (Counts)")
    plt.xlabel("Quality Class")
    plt.ylabel("Count")
    save_fig(os.path.join(target_dir, "quality_class_counts.png"))

    class_pct = df["quality_class"].value_counts(normalize=True).reindex(["low", "medium", "high"]) * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_pct.index, y=class_pct.values, color="teal")
    plt.title("Quality Class Distribution (Percent)")
    plt.xlabel("Quality Class")
    plt.ylabel("Percent")
    save_fig(os.path.join(target_dir, "quality_class_percent.png"))

    print(f"Target analysis plots saved to {target_dir}")


if __name__ == "__main__":
    main()
