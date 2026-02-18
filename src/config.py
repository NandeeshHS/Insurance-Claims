"""
Configuration Module
====================
Central configuration for the Insurance Risk Analytics project.
Contains all hyperparameters, paths, and global settings.

Author: Nandeesh H S
Project: Insurance Risk Analytics & Claims Prediction
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models_saved"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                 RESULTS_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA FILES
# =============================================================================

TRAIN_FILE = RAW_DATA_DIR / "insurance_train.csv"
TEST_FILE = RAW_DATA_DIR / "insurance_test.csv"

TRAIN_PROCESSED = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED = PROCESSED_DATA_DIR / "test_processed.csv"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation
N_FOLDS = 3
CV_STRATEGY = "stratified"  # For classification

# Train-test split
TEST_SIZE = 0.2

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Reference date for age calculations
REFERENCE_DATE = "2019-12-31"

# Date columns
DATE_COLUMNS = ['X.2', 'X.3', 'X.4', 'X.5', 'X.6']

# Categorical columns
CATEGORICAL_FEATURES = ['X.7', 'X.13', 'X.19', 'X.20', 'X.21', 'X.27']

# Target variables
TARGET_COLUMNS = ['CS', 'LC', 'HALC']

# Columns to exclude from features
EXCLUDE_COLUMNS = ['X.1', 'X.15', 'X.16', 'X.17', 'X.18', 'CS', 'LC', 'HALC']

# Feature selection
N_FEATURES_TO_SELECT = 15
FEATURE_SELECTION_METHOD = "rfe"  # Recursive Feature Elimination

# =============================================================================
# HYPERPARAMETER TUNING (OPTUNA)
# =============================================================================

OPTUNA_CONFIG = {
    'classification': {
        'n_trials': 20,
        'timeout': None,
        'direction': 'maximize',  # Maximize AUC
        'sampler': 'TPE'
    },
    'regression': {
        'n_trials': 15,
        'timeout': None,
        'direction': 'minimize',  # Minimize MSE
        'sampler': 'TPE'
    }
}

# Convenience aliases for direct imports by model modules
OPTUNA_N_TRIALS_CLASSIFICATION = OPTUNA_CONFIG['classification']['n_trials']
OPTUNA_N_TRIALS_REGRESSION = OPTUNA_CONFIG['regression']['n_trials']
OPTUNA_TIMEOUT = OPTUNA_CONFIG['classification']['timeout']

# Tweedie variance power bounds
TWEEDIE_VARIANCE_POWER_MIN = 1.1
TWEEDIE_VARIANCE_POWER_MAX = 1.9

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# XGBoost search space
XGBOOST_PARAMS = {
    'max_depth': (3, 8),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 500),
    'min_child_weight': (1, 7),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-8, 10.0),
    'reg_lambda': (1e-8, 10.0),
}

# LightGBM search space
LIGHTGBM_PARAMS = {
    'num_leaves': (20, 100),
    'max_depth': (3, 8),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 500),
    'min_child_samples': (5, 50),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (1e-8, 10.0),
    'reg_lambda': (1e-8, 10.0),
}

# Tweedie regression
TWEEDIE_PARAMS = {
    'variance_power': (1.1, 1.9),  # Power parameter for Tweedie distribution
    'alpha': 0.1,  # Regularization
    'max_iter': 300
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 6)
DPI = 300

COLOR_PALETTE = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'dark': '#34495e'
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

CLASSIFICATION_METRICS = ['roc_auc', 'precision', 'recall', 'f1']
REGRESSION_METRICS = ['mse', 'rmse', 'mae', 'r2']

# =============================================================================
# CONSTANTS
# =============================================================================

# Problem-specific constants
CLAIM_THRESHOLD = 0  # Zero claims indicator
MIN_DRIVER_AGE = 15
MAX_DRIVER_AGE = 100
MIN_VEHICLE_AGE = 0
MAX_VEHICLE_AGE = 50

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_path(model_name: str) -> Path:
    """Get the save path for a trained model."""
    return MODELS_DIR / f"{model_name}.pkl"

def get_results_path(result_type: str, filename: str) -> Path:
    """Get the save path for results/visualizations."""
    return RESULTS_DIR / result_type / filename

# =============================================================================
# DISPLAY CONFIG ON IMPORT
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("INSURANCE RISK ANALYTICS - CONFIGURATION")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"CV Folds: {N_FOLDS}")
    print(f"Feature Selection: Top {N_FEATURES_TO_SELECT} features")
    print(f"Optuna Trials: {OPTUNA_CONFIG['classification']['n_trials']} (classification)")
    print("="*60)
