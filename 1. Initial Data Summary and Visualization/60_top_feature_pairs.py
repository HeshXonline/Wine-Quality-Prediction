import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda_utils import load_data, ensure_dir, save_fig, setup_plotting


def main():
    setup_plotting()
    df = load_data()
    pair_dir = ensure_dir("Top_Feature_Pairs")

    features = [c for c in df.columns if c not in ["quality", "quality_class"]]
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"})
        .sort_values("abs_corr", ascending=False)
    )

    top_pairs = pairs.head(10)

    for _, row in top_pairs.iterrows():
        a = row["feature_a"]
        b = row["feature_b"]
        corr_val = row["abs_corr"]
        plt.figure(figsize=(7, 5))
        sns.regplot(x=df[a], y=df[b], scatter_kws={"alpha": 0.4, "s": 18}, line_kws={"color": "red"})
        plt.title(f"{a} vs {b} (|corr|={corr_val:.2f})")
        plt.xlabel(a)
        plt.ylabel(b)
        filename = f"scatter_{a}_vs_{b}.png".replace(" ", "_")
        save_fig(os.path.join(pair_dir, filename))

    print(f"Top feature pair plots saved to {pair_dir}")


if __name__ == "__main__":
    main()
