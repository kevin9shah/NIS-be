"""
SHAP Explainability Module
===========================

Provides model-agnostic explainability using:
1. SHAP (SHapley Additive exPlanations) via TreeExplainer
   - Global feature importance (mean |SHAP| values)
   - Per-sample SHAP value matrix for heatmap visualization
2. Partial Dependence Plots (PDP) via sklearn
   - Shows marginal effect of each feature on prediction

SHAP values decompose each prediction into per-feature contributions,
answering "which features drove this packet to be flagged as anomalous?"
"""

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ShapExplainer:
    """
    SHAP-based model explainability for tree ensemble models.
    
    Uses TreeExplainer for efficient computation on tree-based models
    (XGBoost, Random Forest, Gradient Boosting).
    """

    def compute_shap(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: List[str],
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for model explanations.

        Uses shap.TreeExplainer for tree-based models. Subsamples data
        to max_samples for computational efficiency (handles up to
        50,000 rows by default through this subsampling).

        Args:
            model: Trained tree-based classifier
            X: Feature matrix
            feature_names: List of feature column names
            max_samples: Maximum samples to explain (default 500)

        Returns:
            Dict with:
                shap_mean_abs: [{feature, shap_importance}] sorted desc
                shap_matrix: 2D list [samples × features] for heatmap
                expected_value: float baseline prediction
        """
        # Subsample for efficiency
        n_samples = min(len(X), max_samples)
        X_sample = X.iloc[:n_samples].copy()

        logger.info(f"Computing SHAP values for {n_samples} samples...")

        # TreeExplainer is optimized for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-output (binary classification returns 2D or list)
        if isinstance(shap_values, list):
            # For binary: use class 1 (anomaly) SHAP values
            shap_vals = np.array(shap_values[1]) if len(shap_values) > 1 else np.array(shap_values[0])
        else:
            shap_vals = np.array(shap_values)

        # Mean absolute SHAP importance per feature
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        shap_importance = sorted(
            [{"feature": f, "shap_importance": round(float(v), 6)}
             for f, v in zip(feature_names, mean_abs)],
            key=lambda x: x["shap_importance"],
            reverse=True,
        )

        # SHAP matrix for heatmap (cap at 50 samples for frontend)
        heatmap_samples = min(50, len(shap_vals))
        shap_matrix = shap_vals[:heatmap_samples].tolist()

        # Expected value (baseline)
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            expected_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            expected_value = float(ev)

        return {
            "shap_mean_abs": shap_importance,
            "shap_matrix": shap_matrix,
            "expected_value": round(expected_value, 6),
        }

    def compute_pdp(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: List[str],
        features_to_plot: Optional[List[str]] = None,
        n_grid_points: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Compute Partial Dependence Plots for top features.

        PDP shows the marginal effect of a feature on the predicted
        outcome. For each feature, we vary its value across a grid
        while keeping other features at their observed values.

        Args:
            model: Trained classifier
            X: Feature matrix
            feature_names: Column names
            features_to_plot: Specific features to compute PDP for.
                              If None, uses top 4 by SHAP importance.
            n_grid_points: Number of grid points for PDP curve

        Returns:
            List of {feature, grid_values, avg_predictions}
        """
        if features_to_plot is None:
            features_to_plot = feature_names[:4]

        # Limit to available features
        features_to_plot = [f for f in features_to_plot if f in feature_names][:4]

        results = []
        X_array = X[feature_names].values if isinstance(X, pd.DataFrame) else X

        for feature in features_to_plot:
            try:
                feature_idx = feature_names.index(feature)
                pdp_result = partial_dependence(
                    model,
                    X_array,
                    features=[feature_idx],
                    kind="average",
                    grid_resolution=n_grid_points,
                )

                grid_values = pdp_result["grid_values"][0].tolist()
                avg_preds = pdp_result["average"][0].tolist()

                results.append({
                    "feature": feature,
                    "grid_values": [round(float(v), 6) for v in grid_values],
                    "avg_predictions": [round(float(v), 6) for v in avg_preds],
                })
            except Exception as e:
                logger.warning(f"PDP computation failed for {feature}: {e}")
                continue

        return results
