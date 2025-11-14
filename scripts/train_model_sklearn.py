#!/usr/bin/env python3
"""
Standalone model training and evaluation script for parking meter violations.

This script trains a Random Forest classifier to predict parking ticket locations
based on temporal and spatial features extracted from DC parking violation data.

Uses scikit-learn for compatibility across different systems.

Usage:
    python train_model_sklearn.py [--input_file path/to/tickets.csv] [--output_dir ./models] [--no_plot]
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParkingTicketModelTrainer:
    """Trains and evaluates a parking ticket prediction model."""

    def __init__(self, input_file: str, output_dir: str = "./models", random_state: int = 42):
        """
        Initialize the trainer.

        Args:
            input_file: Path to tickets_extracted.csv
            output_dir: Directory to save trained model and results
            random_state: Random seed for reproducibility
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.results = {}

    def load_data(self) -> pd.DataFrame:
        """Load and validate ticket data."""
        logger.info(f"Loading data from {self.input_file}...")

        if not self.input_file.exists():
            raise FileNotFoundError(f"File not found: {self.input_file}")

        self.df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(self.df)} tickets")

        # Validate required columns
        required_cols = ['x', 'y', 'hour', 'minute', 'day', 'month', 'year']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return self.df

    def prepare_features(self) -> tuple:
        """
        Prepare features for modeling.

        Creates clusters of ticket locations and generates temporal features.
        Returns a feature matrix X and target vector y.
        """
        logger.info("Preparing features...")

        df = self.df.copy()

        # Create date column
        df['date'] = pd.to_datetime(
            df[['year', 'month', 'day']], errors='coerce'
        )

        # Create hour feature (handle missing values)
        df['hour'] = df['hour'].fillna(-1).astype(int)

        # Filter out rows with missing critical data
        df_valid = df[
            (df['x'].notna()) &
            (df['y'].notna()) &
            (df['date'].notna()) &
            (df['hour'] >= 0)
        ].copy()

        logger.info(f"Using {len(df_valid)} tickets with complete temporal data")

        # Extract temporal features
        df_valid['date_ordinal'] = df_valid['date'].dt.date.apply(
            lambda d: d.toordinal()
        )
        df_valid['day_of_month'] = df_valid['date'].dt.day
        df_valid['weekday'] = df_valid['date'].dt.isocalendar().day
        df_valid['latitude'] = df_valid['y']
        df_valid['longitude'] = df_valid['x']

        # Perform DBSCAN clustering to find hotspot locations
        logger.info("Computing DBSCAN clusters for ticket locations...")
        clusters = self._compute_clusters(df_valid)
        df_valid['cluster'] = clusters

        # Create target: 1 if in a cluster, 0 if noise
        df_valid['y_target'] = (df_valid['cluster'] != -1).astype(int)

        # Select features
        feature_cols = ['date_ordinal', 'day_of_month', 'weekday', 'hour', 'longitude', 'latitude']
        X = df_valid[feature_cols].copy()
        y = df_valid['y_target'].copy()

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def _compute_clusters(self, df: pd.DataFrame, eps: float = 0.005, min_samples: int = 3) -> np.ndarray:
        """
        Compute DBSCAN clusters for ticket locations.

        Args:
            df: DataFrame with 'x' and 'y' columns (longitude, latitude)
            eps: DBSCAN epsilon parameter (degrees) - smaller = more clusters
            min_samples: DBSCAN min_samples parameter - smaller = more lenient clustering

        Returns:
            Array of cluster labels
        """
        coords = df[['x', 'y']].values
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        return clusterer.fit_predict(coords)

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Split data into training and testing sets."""
        logger.info("Splitting data into train/test sets...")

        # Check if we can stratify (need at least 2 samples of minority class)
        min_class_count = y.value_counts().min()
        use_stratify = min_class_count >= 2

        if use_stratify:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            logger.warning(f"Cannot stratify: minority class has only {min_class_count} sample(s)")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        logger.info(f"Train set size: {len(self.X_train)}")
        logger.info(f"Test set size: {len(self.X_test)}")
        logger.info(f"Train positive class: {self.y_train.sum() / len(self.y_train):.1%}")
        logger.info(f"Test positive class: {self.y_test.sum() / len(self.y_test):.1%}")

    def train_model(self) -> RandomForestClassifier:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.model.fit(self.X_train, self.y_train)
        logger.info("Model training complete")

        return self.model

    def evaluate_model(self) -> dict:
        """
        Evaluate model performance on test set.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Log results
        logger.info("=" * 50)
        logger.info("TEST SET PERFORMANCE")
        logger.info("=" * 50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info("=" * 50)

        self.results['test_metrics'] = metrics
        return metrics

    def cross_validate_model(self, n_splits: int = 5) -> dict:
        """
        Perform cross-validation on the model.

        Args:
            n_splits: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")

        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc'
        }

        cv_results = cross_validate(
            self.model,
            self.X_train,
            self.y_train,
            cv=n_splits,
            scoring=scoring,
            return_train_score=True
        )

        # Summarize results
        summary = {}
        for metric in scoring.keys():
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'

            test_scores = cv_results[test_key]
            train_scores = cv_results[train_key]

            summary[metric] = {
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
                'test_scores': test_scores.tolist(),
                'train_scores': train_scores.tolist(),
            }

        logger.info("=" * 50)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("=" * 50)
        for metric, scores in summary.items():
            logger.info(
                f"{metric.upper()}: {scores['test_mean']:.4f} (+/- {scores['test_std']:.4f})"
            )
        logger.info("=" * 50)

        self.results['cv_results'] = summary
        return summary

    def feature_importance(self, top_n: int = 10) -> dict:
        """
        Get feature importance from the model.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary with feature importance
        """
        logger.info("Computing feature importance...")

        feature_names = self.X_train.columns
        importance = self.model.feature_importances_

        # Create sorted list
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info("Top features:")
        for idx, row in feature_imp.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        self.results['feature_importance'] = feature_imp.to_dict('records')
        return feature_imp

    def save_model(self, model_path: str = None) -> Path:
        """
        Save trained model to disk.

        Args:
            model_path: Path to save model. Defaults to output_dir/model.pkl

        Returns:
            Path to saved model
        """
        if model_path is None:
            model_path = self.output_dir / "model.pkl"
        else:
            model_path = Path(model_path)

        logger.info(f"Saving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved successfully")
        return model_path

    def save_results(self, results_path: str = None) -> Path:
        """
        Save evaluation results as JSON.

        Args:
            results_path: Path to save results. Defaults to output_dir/results.json

        Returns:
            Path to saved results
        """
        if results_path is None:
            results_path = self.output_dir / "results.json"
        else:
            results_path = Path(results_path)

        logger.info(f"Saving results to {results_path}...")

        # Ensure numpy/pandas types are JSON serializable
        results_for_json = self._make_json_serializable(self.results)

        with open(results_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)

        logger.info(f"Results saved successfully")
        return results_path

    @staticmethod
    def _make_json_serializable(obj):
        """Convert numpy and pandas types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: ParkingTicketModelTrainer._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ParkingTicketModelTrainer._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        else:
            return obj

    def plot_results(self, output_dir: str = None) -> dict:
        """
        Generate visualization plots.

        Args:
            output_dir: Directory to save plots. Defaults to self.output_dir

        Returns:
            Dictionary with plot paths
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping plots.")
            return {}

        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}

        try:
            # Confusion matrix plot
            y_pred = self.model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            cm_path = output_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=100, bbox_inches='tight')
            plt.close()
            plot_paths['confusion_matrix'] = str(cm_path)
            logger.info(f"Saved confusion matrix plot to {cm_path}")
        except Exception as e:
            logger.warning(f"Failed to create confusion matrix plot: {e}")

        try:
            # ROC curve
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            roc_path = output_dir / "roc_curve.png"
            plt.savefig(roc_path, dpi=100, bbox_inches='tight')
            plt.close()
            plot_paths['roc_curve'] = str(roc_path)
            logger.info(f"Saved ROC curve plot to {roc_path}")
        except Exception as e:
            logger.warning(f"Failed to create ROC curve plot: {e}")

        try:
            # Feature importance plot
            feature_imp = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp, y='feature', x='importance')
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')
            imp_path = output_dir / "feature_importance.png"
            plt.savefig(imp_path, dpi=100, bbox_inches='tight')
            plt.close()
            plot_paths['feature_importance'] = str(imp_path)
            logger.info(f"Saved feature importance plot to {imp_path}")
        except Exception as e:
            logger.warning(f"Failed to create feature importance plot: {e}")

        return plot_paths

    def run_full_pipeline(self, cross_val: bool = True, plot: bool = True) -> dict:
        """
        Run the complete training and evaluation pipeline.

        Args:
            cross_val: Whether to perform cross-validation
            plot: Whether to generate plots

        Returns:
            Dictionary with all results and paths
        """
        logger.info("Starting model training pipeline...")
        logger.info("=" * 50)

        # Load and prepare data
        self.load_data()
        X, y = self.prepare_features()

        # Split data
        self.split_data(X, y)

        # Train model
        self.train_model()

        # Evaluate
        self.evaluate_model()

        # Cross-validation
        if cross_val:
            self.cross_validate_model()

        # Feature importance
        self.feature_importance()

        # Save outputs
        model_path = self.save_model()
        results_path = self.save_results()

        self.results['model_path'] = str(model_path)
        self.results['results_path'] = str(results_path)

        # Plots
        if plot:
            plot_paths = self.plot_results()
            self.results['plot_paths'] = plot_paths

        logger.info("=" * 50)
        logger.info("Pipeline complete!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")

        return self.results


def main():
    """Command-line interface for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a parking ticket prediction model using scikit-learn"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='scripts/tickets_extracted.csv',
        help='Path to tickets_extracted.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models',
        help='Directory to save model and results'
    )
    parser.add_argument(
        '--model_output',
        type=str,
        default=None,
        help='Path to save the model.pkl file (e.g., api/model.pkl). Overrides output_dir for model only.'
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--no_cv',
        action='store_true',
        help='Skip cross-validation'
    )

    args = parser.parse_args()

    try:
        trainer = ParkingTicketModelTrainer(args.input_file, args.output_dir)
        results = trainer.run_full_pipeline(
            cross_val=not args.no_cv,
            plot=not args.no_plot
        )

        # Save model to custom location if specified
        if args.model_output:
            model_path = trainer.save_model(args.model_output)
            results['model_path'] = str(model_path)
            logger.info(f"Model also saved to: {model_path}")

        logger.info("\n" + "=" * 50)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model: {results['model_path']}")
        logger.info(f"Results: {results['results_path']}")
        if 'plot_paths' in results:
            logger.info("Plots generated:")
            for plot_name, plot_path in results['plot_paths'].items():
                logger.info(f"  - {plot_name}: {plot_path}")

        return 0

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
