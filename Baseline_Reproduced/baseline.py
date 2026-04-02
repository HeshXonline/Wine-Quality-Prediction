# ============================================================
# Optuna-Free Baseline Reproduction (Hardcoded Best Params)
# ============================================================

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. HARDCODED OPTIMAL PARAMETERS (From Optuna Output)
# ============================================================
BEST_PARAMS = {
    'Decision Tree': {
        'max_depth': 18, 
        'min_samples_split': 3, 
        'min_samples_leaf': 1, 
        'criterion': 'gini'
    },
    'Random Forest': {
        'n_estimators': 128, 
        'max_depth': 29, 
        'min_samples_split': 3, 
        'min_samples_leaf': 1, 
        'max_features': 'sqrt'
    },
    'SVM': {
        'C': 52.51153534188632, 
        'gamma': 'scale', 
        'kernel': 'rbf'
    },
    'KNN': {
        'n_neighbors': 4, 
        'weights': 'distance', 
        'metric': 'manhattan'
    },
    'Gradient Boosting': {
        'n_estimators': 495, 
        'learning_rate': 0.0404186339977614, 
        'max_depth': 9, 
        'subsample': 0.6522563836084332, 
        'colsample_bytree': 0.6472570559840887, 
        'min_child_weight': 1, 
        'reg_alpha': 7.070910364320472e-08, 
        'reg_lambda': 0.0009212525453530797, 
        'gamma': 0.03532237654907653
    }
}


# ============================================================
# 2. SETUP — Data preparation
# ============================================================
print("=" * 60)
print("Loading and preparing data...")
print("=" * 60)

wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

def encode_quality(q):
    if q <= 4:   return 0
    elif q <= 6: return 1
    else:        return 2

y_encoded = y['quality'].apply(encode_quality)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.3, random_state=42, stratify=y_encoded
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Train (after SMOTE): {X_train_sm.shape}")
print(f"Test               : {X_test.shape}\n")


# ============================================================
# 3. HELPER — Evaluate a trained model
# ============================================================
def evaluate(model, name):
    print(f"Training {name}...")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    ybin = label_binarize(y_test, classes=[0, 1, 2])
    auc  = roc_auc_score(ybin, y_prob, multi_class='ovr', average='weighted')

    return {'Accuracy': round(acc,2), 'Precision': round(prec,2),
            'Recall': round(rec,2), 'F1-Score': round(f1,2),
            'ROC-AUC': round(auc,2)}


# ============================================================
# 4. INITIALIZE MODELS WITH SAVED PARAMS
# ============================================================
models = {
    'Decision Tree': DecisionTreeClassifier(**BEST_PARAMS['Decision Tree'], random_state=42),
    'Random Forest': RandomForestClassifier(**BEST_PARAMS['Random Forest'], random_state=42, n_jobs=-1),
    'SVM': SVC(**BEST_PARAMS['SVM'], probability=True, random_state=42),
    'KNN': KNeighborsClassifier(**BEST_PARAMS['KNN']),
    'Gradient Boosting': XGBClassifier(**BEST_PARAMS['Gradient Boosting'], random_state=42, eval_metric='mlogloss', verbosity=0, n_jobs=-1)
}

# Store results
all_results = []
for name, model in models.items():
    res = evaluate(model, name)
    all_results.append((name, res))


# ============================================================
# 5. FINAL SUMMARY TABLE
# ============================================================
# These are the metrics reported in the 2025 paper
paper_results = {
    'Decision Tree':     {'Accuracy':0.73,'F1-Score':0.71,'ROC-AUC':0.76},
    'Random Forest':     {'Accuracy':0.85,'F1-Score':0.83,'ROC-AUC':0.89},
    'SVM':               {'Accuracy':0.79,'F1-Score':0.77,'ROC-AUC':0.81},
    'KNN':               {'Accuracy':0.76,'F1-Score':0.74,'ROC-AUC':0.78},
    'Gradient Boosting': {'Accuracy':0.86,'F1-Score':0.84,'ROC-AUC':0.90},
}

print("\n" + "=" * 65)
print("FINAL RESULTS — INSTANT REPRODUCTION vs 2025 PAPER")
print("=" * 65)
print(f"{'Model':<20} {'Metric':<12} {'Ours':>6} {'Paper':>7} {'Diff':>7}  Status")
print("-" * 65)

for name, res in all_results:
    for metric in ['Accuracy', 'F1-Score', 'ROC-AUC']:
        ours  = res[metric]
        pval  = paper_results[name][metric]
        diff  = round(ours - pval, 2)
        # Using a 0.05 threshold to account for slight differences in library versions / exact SMOTE implementation
        status = "OK" if abs(diff) <= 0.05 else "CHECK"
        print(f"{name:<20} {metric:<12} {ours:>6.2f} {pval:>7.2f} {diff:>+7.2f}  {status}")
    print()

print("=" * 65)
print("Instant execution complete! Your baseline is locked in.")
print("=" * 65)