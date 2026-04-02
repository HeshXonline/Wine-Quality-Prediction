import os
import matplotlib.pyplot as plt
import seaborn as sns
from eda_utils import load_data, ensure_dir, save_fig, setup_plotting


def main():
    setup_plotting()
    df = load_data()
    uni_dir = ensure_dir("Univariate_Distributions")

    numeric_cols = [c for c in df.columns if c not in ["quality_class"]]

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30, color="royalblue")
        plt.title(f"Histogram + KDE: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_fig(os.path.join(uni_dir, f"hist_kde_{col}.png"))

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col], color="lightcoral")
        plt.title(f"Boxplot: {col}")
        plt.xlabel(col)
        save_fig(os.path.join(uni_dir, f"box_{col}.png"))

        plt.figure(figsize=(8, 4))
        sns.violinplot(x=df[col], color="lightslategray")
        plt.title(f"Violin: {col}")
        plt.xlabel(col)
        save_fig(os.path.join(uni_dir, f"violin_{col}.png"))

        plt.figure(figsize=(8, 5))
        sns.ecdfplot(df[col], color="darkorange")
        plt.title(f"ECDF: {col}")
        plt.xlabel(col)
        plt.ylabel("ECDF")
        save_fig(os.path.join(uni_dir, f"ecdf_{col}.png"))

    print(f"Univariate distribution plots saved to {uni_dir}")


if __name__ == "__main__":
    main()
