"""
Model Evaluation Module
=======================
Comprehensive evaluation metrics and validation for insurance claims models.

Author: Nandeesh H S
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class ModelEvaluator:
    """
    Evaluates classification and regression models.

    Provides comprehensive metrics for both claim status prediction
    and loss cost estimation.
    """

    @staticmethod
    def evaluate_classifier(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Evaluate binary classification performance.

        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of classification metrics
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        cm = confusion_matrix(y_true, y_pred_binary)

        # Calculate precision, recall, f1
        report = classification_report(y_true, y_pred_binary, output_dict=True)

        results = {
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': thresholds,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'classification_report': report
        }

        return results

    @staticmethod
    def evaluate_regressor(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str = "LC"
    ) -> Dict[str, float]:
        """
        Evaluate regression performance.

        Args:
            y_true: True values
            y_pred: Predicted values
            target_name: Name of target variable

        Returns:
            Dictionary of regression metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # R2 score (can be negative for poor fits)
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan

        # Mean absolute percentage error (for non-zero values)
        mask = y_true > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        results = {
            f'{target_name}_mse': mse,
            f'{target_name}_rmse': rmse,
            f'{target_name}_mae': mae,
            f'{target_name}_r2': r2,
            f'{target_name}_mape': mape
        }

        return results

    @staticmethod
    def print_classification_summary(metrics: Dict[str, any], title: str = "Classification Results"):
        """
        Print formatted classification metrics.

        Args:
            metrics: Dictionary from evaluate_classifier
            title: Title for the summary
        """
        print("\n" + "="*80)
        print(title.upper())
        print("="*80)
        print(f"\nAUC-ROC:       {metrics['auc']:.4f}")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1-Score:      {metrics['f1_score']:.4f}")

        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  TN: {cm[0,0]:>6,}  |  FP: {cm[0,1]:>6,}")
        print(f"  FN: {cm[1,0]:>6,}  |  TP: {cm[1,1]:>6,}")

    @staticmethod
    def print_regression_summary(metrics: Dict[str, float], title: str = "Regression Results"):
        """
        Print formatted regression metrics.

        Args:
            metrics: Dictionary from evaluate_regressor
            title: Title for the summary
        """
        print("\n" + "="*80)
        print(title.upper())
        print("="*80)

        for key, value in metrics.items():
            if not np.isnan(value):
                if 'mse' in key or 'mae' in key or 'rmse' in key:
                    print(f"{key.upper():<20} {value:>12,.2f}")
                elif 'mape' in key:
                    print(f"{key.upper():<20} {value:>12,.2f}%")
                else:
                    print(f"{key.upper():<20} {value:>12,.4f}")

    @staticmethod
    def create_performance_summary(
        clf_metrics: Dict[str, any],
        lc_metrics: Dict[str, float],
        halc_metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame of all metrics.

        Args:
            clf_metrics: Classification metrics
            lc_metrics: LC regression metrics
            halc_metrics: HALC regression metrics

        Returns:
            Summary DataFrame
        """
        summary_data = {
            'Metric': [
                'Classification AUC',
                'Classification Accuracy',
                'LC MSE',
                'LC RMSE',
                'LC MAE',
                'HALC MSE',
                'HALC RMSE',
                'HALC MAE'
            ],
            'Value': [
                f"{clf_metrics['auc']:.4f}",
                f"{clf_metrics['accuracy']:.4f}",
                f"{lc_metrics['LC_mse']:,.2f}",
                f"{lc_metrics['LC_rmse']:,.2f}",
                f"{lc_metrics['LC_mae']:,.2f}",
                f"{halc_metrics['HALC_mse']:,.2f}",
                f"{halc_metrics['HALC_rmse']:,.2f}",
                f"{halc_metrics['HALC_mae']:,.2f}"
            ]
        }

        return pd.DataFrame(summary_data)

    @staticmethod
    def analyze_prediction_distribution(
        y_pred: np.ndarray,
        target_name: str = "Prediction"
    ) -> Dict[str, float]:
        """
        Analyze the distribution of predictions.

        Args:
            y_pred: Predicted values
            target_name: Name of the target

        Returns:
            Dictionary of distribution statistics
        """
        stats = {
            f'{target_name}_mean': np.mean(y_pred),
            f'{target_name}_median': np.median(y_pred),
            f'{target_name}_std': np.std(y_pred),
            f'{target_name}_min': np.min(y_pred),
            f'{target_name}_max': np.max(y_pred),
            f'{target_name}_zeros': (y_pred == 0).sum(),
            f'{target_name}_zero_pct': (y_pred == 0).mean() * 100
        }

        return stats

    @staticmethod
    def print_distribution_summary(stats: Dict[str, float], title: str = "Distribution Analysis"):
        """
        Print formatted distribution statistics.

        Args:
            stats: Dictionary from analyze_prediction_distribution
            title: Title for the summary
        """
        print("\n" + "="*80)
        print(title.upper())
        print("="*80)

        for key, value in stats.items():
            if 'pct' in key:
                print(f"{key:<30} {value:>12,.2f}%")
            elif isinstance(value, int):
                print(f"{key:<30} {value:>12,}")
            else:
                print(f"{key:<30} {value:>12,.2f}")


def evaluate_complete_model(
    y_cs_true: np.ndarray,
    y_cs_pred: np.ndarray,
    y_lc_true: np.ndarray,
    y_lc_pred: np.ndarray,
    y_halc_true: np.ndarray,
    y_halc_pred: np.ndarray
) -> Dict[str, any]:
    """
    Comprehensive evaluation of all model components.

    Args:
        y_cs_true: True claim status
        y_cs_pred: Predicted claim probabilities
        y_lc_true: True LC values
        y_lc_pred: Predicted LC values
        y_halc_true: True HALC values
        y_halc_pred: Predicted HALC values

    Returns:
        Dictionary containing all evaluation results
    """
    evaluator = ModelEvaluator()

    # Evaluate classification
    clf_metrics = evaluator.evaluate_classifier(y_cs_true, y_cs_pred)

    # Evaluate LC regression
    lc_metrics = evaluator.evaluate_regressor(y_lc_true, y_lc_pred, target_name="LC")

    # Evaluate HALC regression
    halc_metrics = evaluator.evaluate_regressor(y_halc_true, y_halc_pred, target_name="HALC")

    # Print summaries
    evaluator.print_classification_summary(clf_metrics, "Claim Status Prediction")
    evaluator.print_regression_summary(lc_metrics, "Loss Cost Prediction (LC)")
    evaluator.print_regression_summary(halc_metrics, "HALC Prediction")

    return {
        'classification': clf_metrics,
        'lc_regression': lc_metrics,
        'halc_regression': halc_metrics
    }
