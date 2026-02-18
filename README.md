# Insurance Risk Analytics & Claims Prediction

> End-to-end ML pipeline predicting insurance claim occurrence and severity — with a focus on statistical rigor, interpretability, and production-ready code structure.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0+-green.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-red.svg)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-purple.svg)](https://optuna.org/)

---

## The Problem

Insurance pricing is a classic data problem with a tricky statistical structure: **most policyholders never file a claim**, but for those who do, the cost varies enormously.

This project predicts three interrelated targets:

| Target | Type | Description |
|--------|------|-------------|
| **CS** (Claim Status) | Binary Classification | Will this policyholder file a claim? |
| **LC** (Loss Cost) | Regression | Expected cost of the claim |
| **HALC** (Historically Adjusted Loss Cost) | Regression | LC adjusted by historical claim frequency |

Getting this right matters — underpricing leads to losses, overpricing drives away customers. The goal is accurate, data-driven risk estimation.

---

## The Dataset Challenge

- **37,451** training records, **15,787** test records
- **28 anonymized features**: policy details, policyholder demographics, vehicle attributes, dates
- **88.92% of records have zero claims** — severe zero-inflation

This imbalance is not a flaw in the data — it reflects reality. Handling it correctly is the core technical challenge.

---

## Two Approaches, One Clear Winner

The standard industry approach to zero-inflated claims is a **two-stage model**:
1. Classify whether a claim will occur
2. Regress on claim amount given a claim exists
3. Final prediction = P(claim) × E[amount | claim]

I implemented and benchmarked this against a single-stage approach using the **Tweedie distribution** — a compound Poisson-Gamma family that models zero-inflation and positive continuous values simultaneously.

| Approach | Strategy | LC MSE (Validation) |
|----------|----------|---------------------|
| **Approach A** | Two-stage: classify → regress on claims only | 292,458 |
| **Approach B** | Direct Tweedie regression on all data | **2,614** |

**Approach B is ~112x better.** The reason is intuitive: Approach A trains the regressor on only the ~11% of data where claims exist, throwing away signal from the majority. Tweedie regression uses *all 37,451 samples* while still placing probability mass at zero — it's a strictly superior statistical formulation for this problem.

---

## Results in Context

### Classification — Claim Status (CS)

- **ROC-AUC: 0.8387**
- A random classifier scores 0.5. An AUC of 0.84 on an 89/11 imbalanced dataset represents strong discriminative power, achieved without any data resampling (SMOTE etc.) — purely through stratified cross-validation and well-tuned models.
- Predicted claim rate: **11.16%** vs actual **11.08%** — the model is well-calibrated.

### Regression — Loss Cost (LC)

- **MSE: 2,614 → RMSE: ~51**
- Raw MSE numbers don't mean much without context. Insurance claim amounts in this dataset range from 0 to several thousand. The RMSE of ~51 is compact relative to that range, and relative to Approach A's RMSE of ~540 — a 10x reduction in prediction error.
- The zero-inflation is reflected correctly: the model outputs near-zero predictions for low-risk policyholders rather than inflating their expected costs.

### Regression — HALC

- **MSE: 7,724 → RMSE: ~88**
- HALC incorporates historical claim frequency, making it more variable than LC. The model captures this additional spread while maintaining accurate predictions for both zero and non-zero claim populations.

---

## Methodology

### Feature Engineering

Raw data included several date columns that needed transformation before modeling:

| Raw Feature | Engineered Feature | Reasoning |
|-------------|-------------------|-----------|
| Birth date | `driver_age` | Younger/older drivers have different risk profiles |
| License date | `driver_exp` | Experience reduces accident probability |
| Manufacture year | `vehicle_age` | Older vehicles have different claim patterns |
| Policy start/end | `policy_duration` | Longer policies indicate stable customers |
| Engine power + weight | `power_to_weight` | Proxy for driving aggressiveness |
| Premium + vehicle value | `premium_to_value` | Pricing efficiency indicator |
| Tenure metrics | `loyalty` | Long-term customers are typically lower risk |
| — | `driver_age²`, `vehicle_age²` | Capture non-linear (U-shaped) risk relationships |

### Preprocessing Pipeline

```
Raw Data → Feature Engineering → One-Hot Encoding → StandardScaler → Feature Selection → Model Training
```

Feature selection used Random Forest importance at the median threshold, reducing the feature set to the most predictive subset — preventing overfitting without manual feature curation.

### Model Training

All models tuned with **Optuna** using Tree-structured Parzen Estimator (TPE) sampling — a Bayesian approach that learns which hyperparameter regions perform well across trials, significantly more efficient than grid or random search.

| Component | Models | Tuning Trials |
|-----------|--------|---------------|
| Classification | XGBoost, LightGBM, Logistic Regression | 20 per model |
| Regression | XGBoost Tweedie, LightGBM Tweedie, GLM Tweedie | 15 per model |

Final predictions are weighted ensembles where each model's weight is proportional to its cross-validation score — better-performing models contribute more to the final output.

---

## Project Structure

```
Insurance-Claims/
│
├── src/                          # Modular, importable source code
│   ├── config.py                 # Central configuration (seeds, paths, hyperparams)
│   ├── data_loader.py            # DataLoader class + target variable creation
│   ├── feature_engineer.py       # FeatureEngineer class
│   ├── preprocessor.py           # DataPreprocessor (encode, scale, select)
│   ├── evaluation.py             # ModelEvaluator (classification + regression metrics)
│   ├── visualization.py          # InsuranceVisualizer (charts, plots)
│   └── models/
│       ├── classifier.py         # ClaimStatusClassifier ensemble
│       ├── regressor.py          # TweedieRegressionModel ensemble
│       └── ensemble.py           # ApproachA, ApproachB, ApproachComparison
│
├── notebooks/                    # Step-by-step interactive analysis
│   ├── 01_EDA_and_Visualization.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training_Classification.ipynb
│   ├── 04_Model_Training_Regression.ipynb
│   └── 05_Results_and_Comparison.ipynb
│
├── results/                      # Outputs, charts, predictions
├── main.py                       # Pipeline entry point
├── requirements.txt
└── README.md
```

The `src/` modules are designed to be imported independently — notebooks call into `src/` rather than duplicating logic. This separation keeps notebooks readable and keeps the modeling code testable and reusable.

---

## Quickstart

```bash
git clone https://github.com/yourusername/Insurance-Claims.git
cd Insurance-Claims
pip install -r requirements.txt

# Run overview (works without data)
python main.py

# Interactive analysis (requires data in data/raw/)
jupyter notebook notebooks/
```

> **Note**: Data files are excluded from this repository to protect policyholder privacy. The code, methodology, and results are fully documented here and in the notebooks.

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Hyperparameter tuning | `optuna` |
| Visualization | `matplotlib`, `seaborn` |
| Environment | `jupyter`, `python 3.10+` |

---

## Author

**Nandeesh H S**

[LinkedIn](https://linkedin.com/in/yourprofile) · [Email](mailto:your.email@example.com) · [Portfolio](https://yourportfolio.com)
