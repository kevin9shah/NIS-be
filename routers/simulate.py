"""
Simulation Router
=================

POST /api/simulate — What-If anomaly simulation.
"""

from fastapi import APIRouter, HTTPException
from schemas.requests import SimulateRequest
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
    What-if simulation endpoint.
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    model_entry = session.get("models", {}).get(request.model_id)
    if not model_entry:
        raise HTTPException(status_code=404, detail="Model not found.")

    # Stub response to satisfy API contract
    return {
        "simulation_results": []
    }
