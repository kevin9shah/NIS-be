"""
Prediction + QTTA Router
=========================

POST /api/predict — Run anomaly detection with dynamic QTTA thresholding.
Returns per-packet predictions with threat levels and QTTA state evolution.
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from schemas.requests import PredictRequest
from core.qtta import QTTAThreshold
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["predict"])

SESSIONS = None


def set_sessions(sessions):
    global SESSIONS
    SESSIONS = sessions


def _classify_threat(score: float, threshold: float) -> str:
    """
    Classify threat level based on anomaly score.
    
    CRITICAL: score > 0.9
    HIGH:     score > 0.75
    MEDIUM:   score > 0.6
    LOW:      score > qtta_threshold
    NORMAL:   otherwise
    """
    if score > 0.9:
        return "CRITICAL"
    elif score > 0.75:
        return "HIGH"
    elif score > 0.6:
        return "MEDIUM"
    elif score > threshold:
        return "LOW"
    else:
        return "NORMAL"


@router.post("/predict")
async def predict(request: PredictRequest):
    """
    Run predictions on the test set with QTTA threshold adaptation.

    Processes each packet sequentially through the model and QTTA,
    returning per-packet anomaly scores, dynamic thresholds,
    tunneling probabilities, and threat classifications.
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    model_entry = session.get("models", {}).get(request.model_id)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not found. Train a model first.")

    model = model_entry["model"]

    # Get test data
    X_test = session.get("X_test")
    if X_test is None:
        raise HTTPException(status_code=422, detail="No test data available. Train a model first.")

    # QTTA parameters
    qtta_params = request.qtta_params or {}
    base_threshold = qtta_params.get("base_threshold", 0.5)
    alpha = qtta_params.get("alpha", 0.3)
    d = qtta_params.get("d", 1.0)

    try:
        # Get anomaly probabilities for all test packets
        proba = model.predict_proba(X_test)[:, 1]
        static_preds = (proba > base_threshold).astype(int)

        # Initialize QTTA and process each packet
        qtta = QTTAThreshold(base_threshold=base_threshold, alpha=alpha, d=d)

        predictions = []
        for i, score in enumerate(proba):
            is_anomaly, threshold, T = qtta.update(float(score))
            state = qtta.threshold_history[-1]

            predictions.append({
                "index": int(i),
                "anomaly_score": round(float(score), 4),
                "static_prediction": int(static_preds[i]),
                "qtta_threshold": round(float(threshold), 4),
                "qtta_prediction": int(is_anomaly),
                "tunneling_prob": round(float(T), 4),
                "alert_pressure": round(float(state["alert_pressure"]), 4),
                "noise_floor": round(float(state["noise_floor"]), 4),
                "threat_level": _classify_threat(float(score), float(threshold)),
            })

        # Summary statistics
        static_anomaly_count = int(np.sum(static_preds == 1))
        qtta_anomaly_count = sum(1 for p in predictions if p["qtta_prediction"] == 1)

        # Threshold evolution for chart
        threshold_evolution = [
            {
                "index": i,
                "threshold": round(float(h["threshold"]), 4),
                "tunneling_prob": round(float(h["tunneling_prob"]), 4),
            }
            for i, h in enumerate(qtta.get_state_series())
        ]

        logger.info(f"Prediction complete: {len(predictions)} packets, "
                     f"static={static_anomaly_count}, qtta={qtta_anomaly_count}")

        return {
            "predictions": predictions,
            "qtta_summary": {
                "static_anomaly_count": static_anomaly_count,
                "qtta_anomaly_count": qtta_anomaly_count,
                "additional_caught_by_qtta": max(0, qtta_anomaly_count - static_anomaly_count),
                "threshold_evolution": threshold_evolution,
            },
        }

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
