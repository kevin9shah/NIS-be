"""
Simulation Router
=================

POST /api/simulate — Run what-if simulations varying network parameters
to explore model and QTTA behavior under different conditions.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException
from schemas.requests import SimulateRequest
from core.simulator import SimulationEngine
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["simulate"])

SESSIONS = None


def set_sessions(sessions):
    global SESSIONS
    SESSIONS = sessions


@router.post("/simulate")
async def simulate(request: SimulateRequest):
    """
    Run what-if simulation across parameter combinations.

    Varies packet_count_5s and mean_packet_size across their
    specified ranges, generating a heatmap of anomaly probabilities
    and QTTA thresholds.
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    model_entry = session.get("models", {}).get(request.model_id)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not found.")

    model = model_entry["model"]
    preprocessor = session.get("preprocessor")
    if not preprocessor:
        raise HTTPException(status_code=422, detail="Preprocessor not available. Train a model first.")

    df = session["dataframe"]

    try:
        # Use first normal sample as base template
        normal_samples = df[df["label"] == 0] if "label" in df.columns else df
        base_sample = normal_samples.iloc[0].copy()

        # Build parameter grid
        param_grid = {
            "packet_count_5s": [float(v) for v in request.vary_packet_count],
            "mean_packet_size": [float(v) for v in request.vary_mean_packet_size],
        }

        # QTTA parameters
        qtta_params = request.qtta_params or {
            "base_threshold": 0.5,
            "alpha": 0.3,
            "d": 1.0,
        }

        # Run simulation
        engine = SimulationEngine()
        result = engine.run(
            model=model,
            preprocessor=preprocessor,
            qtta_params=qtta_params,
            base_sample=base_sample,
            param_grid=param_grid,
        )

        logger.info(f"Simulation complete: {len(result['results'])} combinations")

        return result

    except Exception as e:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
