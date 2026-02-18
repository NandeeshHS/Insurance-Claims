"""
Visualization Module
===================
Professional plotting utilities for insurance claims analysis.

Author: Nandeesh H S
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class InsuranceVisualizer:
    """
    Creates professional visualizations for insurance claims analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots (if None, plots are displayed only)
        """
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def save_or_show(self, filename: Optional[str] = None, dpi: int = 300):
        """
        Save plot to file or display it.

        Args:
            filename: Filename to save (if None, displays plot)
            dpi: Resolution for saved image
        """
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        title: str = "ROC Curve - Claim Status Prediction",
        filename: Optional[str] = None
    ):
        """
        Plot ROC curve.

        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc: AUC score
            title: Plot title
            filename: Filename to save
        """
        plt.figure(figsize=(10, 8))

        # Plot ROC curve
        plt.plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})')

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], color='#e74c3c', linestyle='--', linewidth=2, label='Random Classifier')

        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)

        self.save_or_show(filename)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix - Claim Status",
        filename: Optional[str] = None
    ):
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix (2x2)
            title: Plot title
            filename: Filename to save
        """
        plt.figure(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=',d',
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=['No Claim', 'Claim'],
            yticklabels=['No Claim', 'Claim'],
            annot_kws={'size': 14, 'weight': 'bold'}
        )

        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)

        self.save_or_show(filename)

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20,
        title: str = "Top Feature Importances",
        filename: Optional[str] = None
    ):
        """
        Plot feature importance bar chart.

        Args:
            feature_names: List of feature names
            importances: Feature importance scores
            top_n: Number of top features to display
            title: Plot title
            filename: Filename to save
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(12, 8))

        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        plt.barh(range(len(top_features)), top_importances, color=colors, edgecolor='black')

        plt.yticks(range(len(top_features)), top_features, fontsize=10)
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)

        self.save_or_show(filename)

    def plot_prediction_distribution(
        self,
        predictions: np.ndarray,
        target_name: str = "Prediction",
        title: Optional[str] = None,
        filename: Optional[str] = None,
        bins: int = 50
    ):
        """
        Plot distribution of predictions.

        Args:
            predictions: Predicted values
            target_name: Name of target variable
            title: Plot title (auto-generated if None)
            filename: Filename to save
            bins: Number of histogram bins
        """
        if title is None:
            title = f"{target_name} Distribution"

        plt.figure(figsize=(12, 6))

        # Remove zeros for better visualization
        non_zero_preds = predictions[predictions > 0]

        # Create histogram
        plt.hist(
            non_zero_preds,
            bins=bins,
            color='#3498db',
            edgecolor='black',
            alpha=0.7
        )

        plt.xlabel(target_name, fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        # Add statistics text
        stats_text = (
            f"Mean: {predictions.mean():.2f}\n"
            f"Median: {np.median(predictions):.2f}\n"
            f"Zeros: {(predictions == 0).sum():,} ({(predictions == 0).mean()*100:.1f}%)"
        )
        plt.text(
            0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        self.save_or_show(filename)

    def plot_approach_comparison(
        self,
        results_a: Dict[str, float],
        results_b: Dict[str, float],
        title: str = "Approach A vs Approach B Comparison",
        filename: Optional[str] = None
    ):
        """
        Compare two modeling approaches.

        Args:
            results_a: Metrics for Approach A
            results_b: Metrics for Approach B
            title: Plot title
            filename: Filename to save
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics = ['cs_auc', 'lc_mse', 'halc_mse']
        titles = ['Claim Status AUC', 'Loss Cost MSE', 'HALC MSE']
        colors = ['#3498db', '#e74c3c']

        for idx, (metric, metric_title) in enumerate(zip(metrics, titles)):
            values = [results_a.get(metric, 0), results_b.get(metric, 0)]
            labels = ['Approach A\n(Two-Stage)', 'Approach B\n(Direct Tweedie)']

            axes[idx].bar(labels, values, color=colors, edgecolor='black', alpha=0.7)
            axes[idx].set_title(metric_title, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=11)
            axes[idx].grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, v in enumerate(values):
                if 'mse' in metric:
                    label_text = f'{v:,.0f}'
                else:
                    label_text = f'{v:.4f}'
                axes[idx].text(i, v, label_text, ha='center', va='bottom', fontweight='bold')

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        self.save_or_show(filename)

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str = "LC",
        filename: Optional[str] = None
    ):
        """
        Plot residual analysis.

        Args:
            y_true: True values
            y_pred: Predicted values
            target_name: Name of target variable
            filename: Filename to save
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=10, color='#3498db')
        axes[0].axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Residual Plot - {target_name}', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Residual histogram
        axes[1].hist(residuals, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residual Value', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Residual Distribution - {target_name}', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self.save_or_show(filename)

    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str = "LC",
        filename: Optional[str] = None
    ):
        """
        Plot actual vs predicted scatter plot.

        Args:
            y_true: True values
            y_pred: Predicted values
            target_name: Name of target variable
            filename: Filename to save
        """
        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20, color='#3498db', edgecolors='none')

        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Prediction')

        plt.xlabel(f'Actual {target_name}', fontsize=12, fontweight='bold')
        plt.ylabel(f'Predicted {target_name}', fontsize=12, fontweight='bold')
        plt.title(f'Actual vs Predicted - {target_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        self.save_or_show(filename)

    def create_model_summary_plot(
        self,
        CS_pred: np.ndarray,
        LC_pred: np.ndarray,
        HALC_pred: np.ndarray,
        filename: Optional[str] = None
    ):
        """
        Create comprehensive summary plot with all predictions.

        Args:
            CS_pred: Claim status predictions
            LC_pred: Loss cost predictions
            HALC_pred: HALC predictions
            filename: Filename to save
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # CS distribution
        axes[0].hist(CS_pred, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Claim Probability', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Claim Status Predictions', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # LC distribution (non-zero)
        LC_nonzero = LC_pred[LC_pred > 0]
        axes[1].hist(LC_nonzero, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Loss Cost', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('LC Predictions (Non-Zero)', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        # HALC distribution (non-zero)
        HALC_nonzero = HALC_pred[HALC_pred > 0]
        axes[2].hist(HALC_nonzero, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('HALC', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[2].set_title('HALC Predictions (Non-Zero)', fontsize=13, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)

        plt.suptitle('Final Model Predictions Overview', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()

        self.save_or_show(filename)
