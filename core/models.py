"""
Model Training Module
=====================

Supports the industry-standard classification algorithm for network anomaly detection:
- XGBoost (XGBClassifier) — primary gradient boosting model

The model is trained with configurable hyperparameters and evaluated
with standard classification metrics.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

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

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "xgboost": {
        "n_estimators": 150,
        "max_depth": 4,           # Reduced: prevents majority-class overfit
        "learning_rate": 0.1,
        "subsample": 0.8,
        "min_child_weight": 5,    # Prevents splits on tiny minority-class nodes
        "gamma": 0.2,             # Minimum loss reduction for a split (regularization)
        "colsample_bytree": 0.7,  # Feature subsampling per tree for robustness
        "eval_metric": "logloss",
    }
}

# Directory to save trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ModelTrainer:
    """
    Unified model trainer supporting XGBoost classifier.
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
            return XGBClassifier(**defaults)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                             f"Choose from: xgboost")

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
            algorithm: Must be "xgboost"
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
