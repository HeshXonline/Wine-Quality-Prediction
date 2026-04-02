import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "Analysis_Graphs")


def setup_plotting():
    sns.set_theme(style="whitegrid", palette="muted")


def ensure_dir(name):
    path = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def load_data():
    wine_quality = fetch_ucirepo(id=186)
    df = wine_quality.data.features.copy()
    df["quality"] = wine_quality.data.targets["quality"]

    def encode_quality(q):
        if q <= 4:
            return "low"
        if q <= 6:
            return "medium"
        return "high"

    df["quality_class"] = df["quality"].apply(encode_quality)
    return df


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
