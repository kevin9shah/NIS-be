"""
Explainability Router
=====================

POST /api/explain — Compute SHAP values and Partial Dependence Plots
for the trained model. Returns feature importance ranking, SHAP value
matrix for heatmap, and PDP curves for top features.
"""

from fastapi import APIRouter, HTTPException
from schemas.requests import ExplainRequest
from core.explainer import ShapExplainer
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["explain"])

SESSIONS = None


def set_sessions(sessions):
    global SESSIONS
    SESSIONS = sessions


@router.post("/explain")
async def explain(request: ExplainRequest):
    """
    Generate SHAP explanations and PDP for the trained model.
    
    Computes:
    - SHAP global feature importance (mean |SHAP| values)
    - SHAP value matrix (50 samples × features) for heatmap
    - Partial dependence plots for top 4 features
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    model_entry = session.get("models", {}).get(request.model_id)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not found.")

    model = model_entry["model"]
    feature_names = session.get("feature_names")
    X_test = session.get("X_test")

    if feature_names is None or X_test is None:
        raise HTTPException(status_code=422, detail="No training data available.")

    try:
        explainer = ShapExplainer()

        # Compute SHAP values
        shap_result = explainer.compute_shap(model, X_test, feature_names)

        # Compute PDP for top 4 features by SHAP importance
        top_features = [item["feature"] for item in shap_result["shap_mean_abs"][:4]]
        pdp_result = explainer.compute_pdp(model, X_test, feature_names, top_features)

        logger.info(f"Explainability computed: {len(shap_result['shap_mean_abs'])} features, "
                     f"{len(pdp_result)} PDP curves")

        return {
            "shap_importance": shap_result["shap_mean_abs"],
            "shap_matrix_sample": shap_result["shap_matrix"],
            "feature_names": feature_names,
            "pdp": pdp_result,
            "expected_value": shap_result["expected_value"],
        }

    except Exception as e:
        logger.exception("Explainability computation failed")
        raise HTTPException(status_code=500, detail=f"Explainability failed: {str(e)}")
