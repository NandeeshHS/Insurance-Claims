"""
Results & Visualizations Generator
====================================
Generates all charts and result images for the results/ folder.
Run this script once after training to populate the results directory.

Author: Nandeesh H S
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TweedieRegressor, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_style("whitegrid")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Output directories
EDA_DIR = Path("results/visualizations/eda")
FI_DIR = Path("results/visualizations/feature_importance")
MC_DIR = Path("results/visualizations/model_comparison")
EDA_DIR.mkdir(parents=True, exist_ok=True)
FI_DIR.mkdir(parents=True, exist_ok=True)
MC_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING RESULTS AND VISUALIZATIONS")
print("=" * 70)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n[1/7] Loading data...")
train = pd.read_csv('data/raw/insurance_train.csv')
test  = pd.read_csv('data/raw/insurance_test.csv')

# Create targets
train['CS']   = (train['X.16'] > 0).astype(int)
train['LC']   = np.where(train['X.16'] > 0, train['X.15'] / train['X.16'], 0)
train['HALC'] = np.where(train['X.16'] > 0, (train['X.15'] / train['X.16']) * train['X.18'], 0)
print(f"  Train: {train.shape}, Test: {test.shape}")

# ── 2. EDA VISUALIZATIONS ─────────────────────────────────────────────────────
print("\n[2/7] Generating EDA visualizations...")

# 2a. Target distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# CS
cs_counts = train['CS'].value_counts()
axes[0].bar(['No Claim', 'Claim'], cs_counts, color=['#3498db', '#e74c3c'],
            edgecolor='black', alpha=0.85)
axes[0].set_title('Claim Status Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
for i, v in enumerate(cs_counts):
    axes[0].text(i, v + 200, f'{v:,}\n({v/len(train)*100:.1f}%)',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# LC (non-zero)
lc_nz = train[train['LC'] > 0]['LC']
axes[1].hist(lc_nz, bins=60, color='#e74c3c', edgecolor='black', alpha=0.75)
axes[1].set_title('Loss Cost Distribution (Non-Zero)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Loss Cost', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].text(0.97, 0.95, f'Mean: {lc_nz.mean():.1f}\nMedian: {lc_nz.median():.1f}\nn={len(lc_nz):,}',
             transform=axes[1].transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6), fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

# HALC (non-zero)
halc_nz = train[train['HALC'] > 0]['HALC']
axes[2].hist(halc_nz, bins=60, color='#2ecc71', edgecolor='black', alpha=0.75)
axes[2].set_title('HALC Distribution (Non-Zero)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('HALC', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].text(0.97, 0.95, f'Mean: {halc_nz.mean():.1f}\nMedian: {halc_nz.median():.1f}\nn={len(halc_nz):,}',
             transform=axes[2].transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6), fontsize=10)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Target Variable Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(EDA_DIR / 'target_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# 2b. Correlation heatmap
key_cols = ['X.8', 'X.9', 'X.10', 'X.11', 'X.12', 'X.14',
            'X.22', 'X.23', 'X.24', 'X.25', 'X.26', 'CS']
key_cols = [c for c in key_cols if c in train.columns]
corr = train[key_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(EDA_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 2c. Zero-inflation illustration
fig, ax = plt.subplots(figsize=(10, 5))
zero_pct = (train['LC'] == 0).mean() * 100
nonzero_pct = 100 - zero_pct
bars = ax.bar(['Zero Claims (No Cost)', 'Non-Zero Claims (With Cost)'],
              [zero_pct, nonzero_pct],
              color=['#95a5a6', '#e74c3c'], edgecolor='black', alpha=0.85, width=0.5)
ax.set_ylabel('Percentage of Records (%)', fontsize=12, fontweight='bold')
ax.set_title('Zero-Inflation in Loss Cost (LC)\nWhy Tweedie Distribution is Required',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
            f'{h:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(EDA_DIR / 'zero_inflation.png', dpi=150, bbox_inches='tight')
plt.close()

print("  EDA visualizations saved.")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\n[3/7] Feature engineering...")

ref_date = pd.to_datetime('2019-12-31')

def engineer(df):
    df = df.copy()
    # Force-parse date columns (handles both string and already-parsed dates)
    for col in ['X.2', 'X.3', 'X.4', 'X.5', 'X.6']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    df['driver_age']       = ((ref_date - df['X.5']).dt.days / 365.25).clip(15, 100).fillna(45).astype(int)
    df['driver_exp']       = ((ref_date - df['X.6']).dt.days / 365.25).clip(0, 80).fillna(20).astype(int)
    df['vehicle_age']      = (2019 - df['X.22']).clip(0, 50).fillna(10).astype(int)
    df['policy_duration']  = ((df['X.4'] - df['X.2']).dt.days / 365.25).clip(0, 10).fillna(1)
    df['power_to_weight']  = (df['X.23'] / (df['X.28'] + 1)).fillna(0.08)
    df['premium_to_value'] = (df['X.14'] / (df['X.25'] + 1)).fillna(0.02)
    df['loyalty']          = (df['X.8'] / (df['X.10'] + 1)).fillna(1)
    df['driver_age_sq']    = df['driver_age'] ** 2
    df['vehicle_age_sq']   = df['vehicle_age'] ** 2
    df.drop(['X.2', 'X.3', 'X.4', 'X.5', 'X.6'], axis=1, inplace=True, errors='ignore')
    return df

train_fe = engineer(train)
test_fe  = engineer(test)

# Engineered features distribution
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
eng_features = ['driver_age', 'driver_exp', 'vehicle_age', 'policy_duration',
                'power_to_weight', 'premium_to_value', 'loyalty', 'driver_age_sq']
titles = ['Driver Age', 'Driver Experience (yrs)', 'Vehicle Age (yrs)',
          'Policy Duration (yrs)', 'Power-to-Weight Ratio',
          'Premium-to-Value Ratio', 'Customer Loyalty Score', 'Driver Age Squared']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
          '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for i, (feat, title, col) in enumerate(zip(eng_features, titles, colors)):
    if feat in train_fe.columns:
        axes[i].hist(train_fe[feat].dropna(), bins=40, color=col, edgecolor='black', alpha=0.75)
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Count', fontsize=9)
        axes[i].grid(axis='y', alpha=0.3)
        mean_val = train_fe[feat].mean()
        axes[i].axvline(mean_val, color='black', linestyle='--', linewidth=1.5,
                        label=f'Mean: {mean_val:.1f}')
        axes[i].legend(fontsize=9)

plt.suptitle('Engineered Features — Distribution Overview', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FI_DIR / 'engineered_features_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 4. PREPROCESSING & FEATURE SELECTION ─────────────────────────────────────
print("\n[4/7] Preprocessing and feature selection...")

target_cols = ['X.1', 'X.15', 'X.16', 'X.17', 'X.18', 'CS', 'LC', 'HALC']
cat_cols    = ['X.7', 'X.13', 'X.19', 'X.20', 'X.21', 'X.27']

feat_cols = [c for c in train_fe.columns if c not in target_cols]
X_train   = pd.get_dummies(train_fe[feat_cols], columns=[c for c in cat_cols if c in feat_cols],
                           drop_first=True, dtype=int)
X_test    = pd.get_dummies(test_fe[[c for c in feat_cols if c in test_fe.columns]],
                           columns=[c for c in cat_cols if c in feat_cols],
                           drop_first=True, dtype=int)
X_test    = X_test.reindex(columns=X_train.columns, fill_value=0)

y_cs   = train_fe['CS'].values
y_lc   = train_fe['LC'].values
y_halc = train_fe['HALC'].values

scaler       = StandardScaler()
X_sc         = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_sc    = pd.DataFrame(scaler.transform(X_test),  columns=X_train.columns)

# Feature selection
rf_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1, max_depth=10)
rf_sel.fit(X_sc, y_cs)
importances = rf_sel.feature_importances_

# Feature importance plot
imp_df = pd.DataFrame({'feature': X_sc.columns, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
plt.barh(range(20), imp_df['importance'], color=colors_fi, edgecolor='black')
plt.yticks(range(20), imp_df['feature'], fontsize=10)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.title('Top 20 Feature Importances\n(Random Forest — used for feature selection)',
          fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FI_DIR / 'feature_importance_rf.png', dpi=150, bbox_inches='tight')
plt.close()

# Select features
selector = SelectFromModel(rf_sel, threshold='median', prefit=True)
X_fin      = pd.DataFrame(selector.transform(X_sc),
                           columns=X_sc.columns[selector.get_support()])
X_test_fin = pd.DataFrame(selector.transform(X_test_sc),
                           columns=X_sc.columns[selector.get_support()])
print(f"  Selected {X_fin.shape[1]} features from {X_sc.shape[1]}")

# ── 5. QUICK MODEL TRAINING (5 trials for speed) ─────────────────────────────
print("\n[5/7] Training models (quick run: 5 Optuna trials each)...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

def tune(objective_fn, direction, n_trials=5):
    study = optuna.create_study(direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study

# XGBoost classifier
def obj_xgb_clf(trial):
    p = dict(max_depth=trial.suggest_int('max_depth', 3, 8),
             learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             n_estimators=trial.suggest_int('n_estimators', 100, 400),
             subsample=trial.suggest_float('subsample', 0.6, 1.0),
             colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
             random_state=RANDOM_SEED, n_jobs=-1, eval_metric='auc')
    return cross_val_score(xgb.XGBClassifier(**p), X_fin, y_cs,
                           cv=cv, scoring='roc_auc', n_jobs=-1).mean()

study_xgb_clf = tune(obj_xgb_clf, 'maximize', n_trials=5)
xgb_clf = xgb.XGBClassifier(**study_xgb_clf.best_params)
xgb_clf.fit(X_fin, y_cs)
print(f"  XGB Classifier AUC: {study_xgb_clf.best_value:.4f}")

# LightGBM classifier
def obj_lgb_clf(trial):
    p = dict(num_leaves=trial.suggest_int('num_leaves', 20, 80),
             learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
             n_estimators=trial.suggest_int('n_estimators', 100, 400),
             subsample=trial.suggest_float('subsample', 0.6, 1.0),
             colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
             random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    return cross_val_score(lgb.LGBMClassifier(**p), X_fin, y_cs,
                           cv=cv, scoring='roc_auc', n_jobs=-1).mean()

study_lgb_clf = tune(obj_lgb_clf, 'maximize', n_trials=5)
lgb_clf = lgb.LGBMClassifier(**study_lgb_clf.best_params)
lgb_clf.fit(X_fin, y_cs)
print(f"  LGB Classifier AUC: {study_lgb_clf.best_value:.4f}")

# Logistic Regression
lr_clf = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, n_jobs=-1)
lr_clf.fit(X_fin, y_cs)
lr_auc = cross_val_score(lr_clf, X_fin, y_cs, cv=cv, scoring='roc_auc').mean()
print(f"  LR Classifier AUC:  {lr_auc:.4f}")

# Ensemble
scores = np.array([study_xgb_clf.best_value, study_lgb_clf.best_value, lr_auc])
w_clf  = scores / scores.sum()
prob_train = (w_clf[0] * xgb_clf.predict_proba(X_fin)[:, 1] +
              w_clf[1] * lgb_clf.predict_proba(X_fin)[:, 1] +
              w_clf[2] * lr_clf.predict_proba(X_fin)[:, 1])
ensemble_auc = roc_auc_score(y_cs, prob_train)
print(f"  Ensemble AUC:       {ensemble_auc:.4f}")

# XGBoost Tweedie (Approach B — all data)
def obj_xgb_tweedie(trial):
    p = dict(objective='reg:tweedie',
             tweedie_variance_power=trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
             max_depth=trial.suggest_int('max_depth', 3, 8),
             learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
             n_estimators=trial.suggest_int('n_estimators', 100, 400),
             subsample=trial.suggest_float('subsample', 0.6, 1.0),
             colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
             random_state=RANDOM_SEED, n_jobs=-1)
    m = xgb.XGBRegressor(**p); m.fit(X_fin, y_lc)
    return mean_squared_error(y_lc, m.predict(X_fin))

study_xgb_b = tune(obj_xgb_tweedie, 'minimize', n_trials=5)
xgb_lc_b = xgb.XGBRegressor(**study_xgb_b.best_params); xgb_lc_b.fit(X_fin, y_lc)
lgb_lc_b = lgb.LGBMRegressor(objective='tweedie', random_state=RANDOM_SEED,
                              n_jobs=-1, verbose=-1); lgb_lc_b.fit(X_fin, y_lc)
glm_lc_b = TweedieRegressor(power=1.5, alpha=0.1, max_iter=300); glm_lc_b.fit(X_fin, y_lc)

mse_x = mean_squared_error(y_lc, xgb_lc_b.predict(X_fin))
mse_l = mean_squared_error(y_lc, lgb_lc_b.predict(X_fin))
mse_g = mean_squared_error(y_lc, glm_lc_b.predict(X_fin))
w_b   = np.array([1/mse_x, 1/mse_l, 1/mse_g]); w_b /= w_b.sum()

lc_pred_b = (w_b[0]*xgb_lc_b.predict(X_fin) +
             w_b[1]*lgb_lc_b.predict(X_fin) +
             w_b[2]*glm_lc_b.predict(X_fin))
mse_b_ens = mean_squared_error(y_lc, lc_pred_b)
print(f"  Approach B LC MSE:  {mse_b_ens:.2f}")

# Approach A (two-stage)
X_cl   = X_fin[y_cs == 1]; y_lc_cl = y_lc[y_cs == 1]
xgb_lc_a = xgb.XGBRegressor(**study_xgb_b.best_params); xgb_lc_a.fit(X_cl, y_lc_cl)
prob_cs_train = xgb_clf.predict_proba(X_fin)[:, 1]
lc_pred_a = prob_cs_train * xgb_lc_a.predict(X_fin)
mse_a_ens = mean_squared_error(y_lc, lc_pred_a)
print(f"  Approach A LC MSE:  {mse_a_ens:.2f}")

# ── 6. GENERATE MODEL COMPARISON VISUALIZATIONS ───────────────────────────────
print("\n[6/7] Generating model comparison visualizations...")

# 6a. ROC Curve
fpr, tpr, _ = roc_curve(y_cs, prob_train)
plt.figure(figsize=(9, 7))
plt.plot(fpr, tpr, color='#3498db', linewidth=2.5,
         label=f'Ensemble ROC (AUC = {ensemble_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#e74c3c', linestyle='--', linewidth=1.5,
         label='Random Classifier (AUC = 0.50)')
plt.fill_between(fpr, tpr, alpha=0.08, color='#3498db')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('ROC Curve — Claim Status Classification\n(Ensemble: XGBoost + LightGBM + Logistic Regression)',
          fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(MC_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 6b. Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_cs, (prob_train >= 0.5).astype(int))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', square=True,
            xticklabels=['No Claim', 'Claim'],
            yticklabels=['No Claim', 'Claim'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix — Ensemble Classifier\n(threshold = 0.5)',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(MC_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 6c. Individual model AUC comparison
fig, ax = plt.subplots(figsize=(10, 6))
model_names = ['XGBoost', 'LightGBM', 'Logistic\nRegression', 'Ensemble\n(Weighted Avg)']
model_aucs  = [study_xgb_clf.best_value, study_lgb_clf.best_value, lr_auc, ensemble_auc]
colors_clf  = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax.bar(model_names, model_aucs, color=colors_clf, edgecolor='black', alpha=0.85)
ax.set_ylim(0.75, max(model_aucs) + 0.02)
ax.set_ylabel('ROC-AUC (3-Fold CV)', fontsize=12, fontweight='bold')
ax.set_title('Classification Model Comparison\nHigher is better', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, model_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(MC_DIR / 'classification_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 6d. Approach A vs Approach B comparison (use known final results)
# Using results from full pipeline run (reported results)
approach_labels = ['Approach A\n(Two-Stage)', 'Approach B\n(Direct Tweedie)']
lc_mses    = [mse_a_ens, mse_b_ens]
lc_rmses   = [np.sqrt(mse_a_ens), np.sqrt(mse_b_ens)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_ab = ['#e74c3c', '#2ecc71']

# MSE
bars = axes[0].bar(approach_labels, lc_mses, color=colors_ab, edgecolor='black', alpha=0.85)
axes[0].set_ylabel('MSE (lower is better)', fontsize=12, fontweight='bold')
axes[0].set_title('Loss Cost — MSE Comparison', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for bar, val in zip(bars, lc_mses):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val * 0.95,
                 f'{val:,.0f}', ha='center', va='top', fontweight='bold', fontsize=12, color='white')
improvement = mse_a_ens / mse_b_ens
axes[0].annotate(f'Approach B is\n{improvement:.0f}x better',
                 xy=(0.5, 0.80), xycoords='axes fraction', ha='center',
                 fontsize=12, fontweight='bold', color='#27ae60',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#27ae60'))

# RMSE
bars2 = axes[1].bar(approach_labels, lc_rmses, color=colors_ab, edgecolor='black', alpha=0.85)
axes[1].set_ylabel('RMSE (lower is better)', fontsize=12, fontweight='bold')
axes[1].set_title('Loss Cost — RMSE Comparison', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, lc_rmses):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val * 0.95,
                 f'{val:,.1f}', ha='center', va='top', fontweight='bold', fontsize=12, color='white')

plt.suptitle('Approach A (Two-Stage) vs Approach B (Direct Tweedie)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(MC_DIR / 'approach_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 6e. Prediction distribution — final test predictions
prob_test = (w_clf[0] * xgb_clf.predict_proba(X_test_fin)[:, 1] +
             w_clf[1] * lgb_clf.predict_proba(X_test_fin)[:, 1] +
             w_clf[2] * lr_clf.predict_proba(X_test_fin)[:, 1])
lc_pred_test = (w_b[0] * xgb_lc_b.predict(X_test_fin) +
                w_b[1] * lgb_lc_b.predict(X_test_fin) +
                w_b[2] * glm_lc_b.predict(X_test_fin))

# XGBoost feature importance
xgb_imp = xgb_clf.feature_importances_
xgb_imp_df = pd.DataFrame({'feature': X_fin.columns, 'importance': xgb_imp})
xgb_imp_df = xgb_imp_df.sort_values('importance', ascending=False)
top_n_fi = min(len(xgb_imp_df), 20)
xgb_imp_df = xgb_imp_df.head(top_n_fi)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
colors_fi2 = plt.cm.plasma(np.linspace(0.3, 0.9, top_n_fi))
axes[0].barh(range(top_n_fi), xgb_imp_df['importance'], color=colors_fi2, edgecolor='black')
axes[0].set_yticks(range(top_n_fi))
axes[0].set_yticklabels(xgb_imp_df['feature'], fontsize=10)
axes[0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
axes[0].set_title(f'Top {top_n_fi} Features — XGBoost Classifier', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# CS prediction distribution by actual class (training data)
prob_no_claim = prob_train[y_cs == 0]
prob_claim    = prob_train[y_cs == 1]
axes[1].hist(prob_no_claim, bins=50, alpha=0.7, label='Actual: No Claim',
             color='#3498db', edgecolor='black')
axes[1].hist(prob_claim, bins=50, alpha=0.7, label='Actual: Claim',
             color='#e74c3c', edgecolor='black')
axes[1].set_xlabel('Predicted Claim Probability', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Score Separation Between Classes\n(Good separation = strong discriminative power)',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FI_DIR / 'xgboost_importance_and_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# 6f. Final summary dashboard
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ROC (small)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fpr, tpr, color='#3498db', linewidth=2, label=f'AUC={ensemble_auc:.4f}')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=1.2, label='Random=0.50')
ax1.set_xlabel('FPR', fontsize=10); ax1.set_ylabel('TPR', fontsize=10)
ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Approach MSE
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(approach_labels, lc_mses, color=colors_ab, edgecolor='black', alpha=0.85)
ax2.set_ylabel('MSE', fontsize=10)
ax2.set_title('Approach A vs B — LC MSE', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(lc_mses):
    ax2.text(i, v * 0.93, f'{v:,.0f}', ha='center', va='top',
             fontweight='bold', color='white', fontsize=10)

# Model AUC comparison
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(['XGB', 'LGB', 'LR', 'Ensemble'], model_aucs,
        color=colors_clf, edgecolor='black', alpha=0.85)
ax3.set_ylim(0.75, max(model_aucs) + 0.02)
ax3.set_ylabel('ROC-AUC', fontsize=10)
ax3.set_title('Classifier Comparison', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, v in enumerate(model_aucs):
    ax3.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# CS prediction dist
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(prob_no_claim, bins=40, alpha=0.7, label='No Claim', color='#3498db')
ax4.hist(prob_claim, bins=40, alpha=0.7, label='Claim', color='#e74c3c')
ax4.set_xlabel('Predicted Probability', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('Score Distribution by Class', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9); ax4.grid(axis='y', alpha=0.3)

# Test CS distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(prob_test, bins=50, color='#9b59b6', edgecolor='black', alpha=0.75)
ax5.axvline(prob_test.mean(), color='black', linestyle='--', linewidth=2,
            label=f'Mean={prob_test.mean():.3f}')
ax5.set_xlabel('Claim Probability', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Test Set: Predicted Claim Probabilities', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9); ax5.grid(axis='y', alpha=0.3)

# Test LC distribution
ax6 = fig.add_subplot(gs[1, 2])
lc_nz_test = lc_pred_test[lc_pred_test > 0.1]
ax6.hist(lc_nz_test, bins=50, color='#e67e22', edgecolor='black', alpha=0.75)
ax6.set_xlabel('Loss Cost', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title('Test Set: Predicted Loss Cost (>0)', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

fig.suptitle('Insurance Claims Prediction — Results Dashboard', fontsize=16, fontweight='bold', y=1.01)
plt.savefig(MC_DIR / 'results_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Model comparison visualizations saved.")

# ── 7. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n[7/7] Saving final prediction distribution chart...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(prob_test, bins=50, color='#3498db', edgecolor='black', alpha=0.75)
axes[0].set_title('CS — Claim Probability', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Probability'); axes[0].set_ylabel('Frequency')
axes[0].grid(axis='y', alpha=0.3)

lc_nz_t = lc_pred_test[lc_pred_test > 0.1]
axes[1].hist(lc_nz_t, bins=50, color='#e74c3c', edgecolor='black', alpha=0.75)
axes[1].set_title('LC — Loss Cost (Non-Zero)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Loss Cost'); axes[1].set_ylabel('Frequency')
axes[1].grid(axis='y', alpha=0.3)

xgb_halc_b = xgb.XGBRegressor(**study_xgb_b.best_params); xgb_halc_b.fit(X_fin, y_halc)
halc_pred_test = xgb_halc_b.predict(X_test_fin)
halc_nz_t = halc_pred_test[halc_pred_test > 0.1]
axes[2].hist(halc_nz_t, bins=50, color='#2ecc71', edgecolor='black', alpha=0.75)
axes[2].set_title('HALC — Adjusted Loss Cost (Non-Zero)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('HALC'); axes[2].set_ylabel('Frequency')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Final Test Predictions — Distribution Overview', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/visualizations/final_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nSaved to:")
print(f"  results/visualizations/eda/                 ({len(list(EDA_DIR.glob('*.png')))} files)")
print(f"  results/visualizations/feature_importance/  ({len(list(FI_DIR.glob('*.png')))} files)")
print(f"  results/visualizations/model_comparison/    ({len(list(MC_DIR.glob('*.png')))} files)")
print(f"\nKey Results:")
print(f"  Ensemble AUC:       {ensemble_auc:.4f}")
print(f"  LC MSE (Approach B): {mse_b_ens:,.1f}  (RMSE={np.sqrt(mse_b_ens):.1f})")
print(f"  LC MSE (Approach A): {mse_a_ens:,.1f}  (RMSE={np.sqrt(mse_a_ens):.1f})")
print(f"  Approach B improvement: {mse_a_ens/mse_b_ens:.1f}x better than Approach A")
