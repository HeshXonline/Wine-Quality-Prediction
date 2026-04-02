import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Set aesthetic parameters in seaborn
sns.set_theme(style="whitegrid", palette="muted")

# 1. Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
subfolders = ["Univariate Analysis", "Bivariate Analysis", "Correlation Analysis", "Outlier Analysis"]

for folder in subfolders:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# 2. Fetch dataset
print("Loading dataset from ucimlrepo...")
wine_quality = fetch_ucirepo(id=186)
df = wine_quality.data.features.copy()
df['quality'] = wine_quality.data.targets['quality']

# Summary Statistics & Deep Analysis
summary_file = os.path.join(BASE_DIR, "data_summary.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("DEEP DATASET ANALYSIS & SUMMARY REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    # 1. Basic Info
    f.write("1. DATASET DIMENSIONS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Observations (Rows): {df.shape[0]}\n")
    f.write(f"Total Features (Columns): {df.shape[1]}\n\n")
    
    # 2. Missing Values
    f.write("2. MISSING VALUES CHECK\n")
    f.write("-" * 40 + "\n")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        f.write("No missing values found in the dataset.\n\n")
    else:
        f.write(missing[missing > 0].to_string() + "\n\n")
        
    # 3. Target Distribution
    f.write("3. TARGET DISTRIBUTION ('quality')\n")
    f.write("-" * 40 + "\n")
    val_counts = df['quality'].value_counts(normalize=True) * 100
    for val, pct in val_counts.items():
        f.write(f"Quality {val}: {pct:.2f}% ({df['quality'].value_counts()[val]} samples)\n")
    f.write("\n")
    
    # 4. Deep Descriptive Statistics (Skewness, Kurtosis)
    f.write("4. DEEP DESCRIPTIVE STATISTICS\n")
    f.write("-" * 40 + "\n")
    desc = df.describe().T
    desc['skewness'] = df.skew()
    desc['kurtosis'] = df.kurtosis()
    f.write(desc[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].to_string() + "\n\n")
    
    f.write("Note on Skewness: > 1 or < -1 indicates a highly skewed distribution.\n")
    f.write("Note on Kurtosis: > 3 indicates heavy tails (more outliers).\n\n")

    # 5. Outlier Detection via IQR
    f.write("5. OUTLIER ASSESSMENT (IQR METHOD)\n")
    f.write("-" * 40 + "\n")
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_counts = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    outlier_df = pd.DataFrame({'Total Outliers': outlier_counts, 'Percentage': (outlier_counts / len(df) * 100).round(2)})
    f.write(outlier_df.to_string() + "\n\n")
    
    # 6. Correlation Analysis with Target
    f.write("6. FEATURE CORRELATIONS WITH TARGET ('quality')\n")
    f.write("-" * 40 + "\n")
    corr_target = df.corr()['quality'].sort_values(ascending=False).drop('quality')
    f.write(corr_target.to_string() + "\n\n")
    
    # 7. Multi-collinearity Check
    f.write("7. HIGHLY CORRELATED FEATURE PAIRS ( > |0.6| )\n")
    f.write("-" * 40 + "\n")
    corr_matrix = df.corr().abs()
    import numpy as np
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(c1, c2, corr_matrix.loc[c1, c2]) for c1 in upper.columns for c2 in upper.index if upper.loc[c2, c1] > 0.6]
    if high_corr:
        for c1, c2, val in sorted(high_corr, key=lambda x: x[2], reverse=True):
            f.write(f"{c2} & {c1}: {val:.3f}\n")
    else:
        f.write("No feature pairs with correlation > 0.6 found.\n")
    f.write("\n")

print(f"Deep summary statistics saved to {summary_file}")

# 3. Univariate Analysis (Histograms and KDEs)
print("Generating Univariate Analysis plots...")
uni_dir = os.path.join(BASE_DIR, "Univariate Analysis")
for col in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30, color='royalblue')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(uni_dir, f'dist_{col}.png'), dpi=300)
    plt.close()

# 4. Bivariate Analysis (Boxplots vs Target)
print("Generating Bivariate Analysis plots...")
bivar_dir = os.path.join(BASE_DIR, "Bivariate Analysis")
features = [c for c in df.columns if c != 'quality']
for col in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='quality', y=col, data=df, palette='Set2')
    plt.title(f'{col} vs Quality', fontsize=14)
    plt.xlabel('Quality')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(bivar_dir, f'box_{col}_vs_quality.png'), dpi=300)
    plt.close()

# 5. Correlation Analysis
print("Generating Correlation Analysis heatmap...")
corr_dir = os.path.join(BASE_DIR, "Correlation Analysis")
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Wine Quality Parameters', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(corr_dir, 'correlation_heatmap.png'), dpi=300)
plt.close()

# 6. Outlier Analysis (Global Boxplots without grouping)
print("Generating Outlier Analysis plots...")
outlier_dir = os.path.join(BASE_DIR, "Outlier Analysis")
for col in df.columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Outlier Detection for {col}', fontsize=14)
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(outlier_dir, f'outlier_{col}.png'), dpi=300)
    plt.close()

print("All EDA scripts executed and plots saved successfully!")
