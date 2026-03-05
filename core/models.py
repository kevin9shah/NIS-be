"""
Model Training Module
=====================

Supports three classification algorithms for network anomaly detection:
- XGBoost (XGBClassifier) — primary gradient boosting model
- Random Forest (RandomForestClassifier) — ensemble bagging model
- Gradient Boosting (GradientBoostingClassifier) — sklearn's GBM

Each model is trained with configurable hyperparameters and evaluated
with standard classification metrics.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

import logging

logger = logging.getLogger(__name__)

# Default hyperparameters for each algorithm
DEFAULT_HYPERPARAMS = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "eval_metric": "logloss",
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
    },
    "gradient_boosting": {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 5,  # Counter class imbalance
    },
}

# Directory to save trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ModelTrainer:
    """
    Unified model trainer supporting XGBoost, Random Forest, and
    Gradient Boosting classifiers.
    """

    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)

    def _get_model(self, algorithm: str, hyperparams: Dict[str, Any], y_train: Optional[np.ndarray] = None):
        """
        Instantiate the selected algorithm with merged hyperparameters.

        For XGBoost, auto-calculates scale_pos_weight from class
        distribution if not explicitly provided.
        """
        defaults = DEFAULT_HYPERPARAMS.get(algorithm, {}).copy()
        # Merge user overrides
        if hyperparams:
            # Filter valid params
            for k, v in hyperparams.items():
                if k in defaults or k == "scale_pos_weight":
                    defaults[k] = v

        if algorithm == "xgboost":
            # Auto-compute scale_pos_weight for imbalanced classes (if not explicitly provided)
            if "scale_pos_weight" not in defaults and y_train is not None:
                n_neg = int(np.sum(y_train == 0))
                n_pos = int(np.sum(y_train == 1))
                if n_pos > 0:
                    defaults["scale_pos_weight"] = n_neg / n_pos
            return XGBClassifier(**defaults)
        elif algorithm == "random_forest":
            return RandomForestClassifier(**defaults, random_state=42)
        elif algorithm == "gradient_boosting":
            # Remove scale_pos_weight if present (GradientBoosting doesn't support it)
            defaults.pop("scale_pos_weight", None)
            return GradientBoostingClassifier(**defaults, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                             f"Choose from: xgboost, random_forest, gradient_boosting")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        algorithm: str,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Train a classification model.

        Args:
            X_train: Feature matrix
            y_train: Binary labels (0=normal, 1=anomaly)
            algorithm: One of "xgboost", "random_forest", "gradient_boosting"
            hyperparams: Optional dict of algorithm-specific parameters

        Returns:
            Trained sklearn/xgboost model object
        """
        model = self._get_model(algorithm, hyperparams or {}, y_train)
        logger.info(f"Training {algorithm} with {len(X_train)} samples...")
        model.fit(X_train, y_train)
        return model

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Returns:
            Dict with: accuracy, precision, recall, f1, roc_auc,
            confusion_matrix (2×2 list), classification_report (dict),
            feature_importances (sorted list of {feature, importance})
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True)

        # Feature importances
        importances = model.feature_importances_
        fi_list = sorted(
            [{"feature": f, "importance": round(float(imp), 6)}
             for f, imp in zip(feature_names, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
            "confusion_matrix": cm,
            "classification_report": cr,
            "feature_importances": fi_list,
        }

    def save_model(self, model: Any, algorithm: str) -> str:
        """
        Save trained model to disk.

        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{algorithm}_{timestamp}.pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
