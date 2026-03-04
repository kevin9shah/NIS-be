"""
Q-SAND API — Main Entry Point
===============================

FastAPI application for the Quantum-Inspired Network Anomaly Detection
Dashboard. Provides REST API endpoints for dataset upload, model training,
anomaly detection with QTTA, SHAP explainability, and what-if simulation.

Session Management:
    In-memory dict keyed by session_id (UUID). Each session stores:
    - Uploaded DataFrame
    - Trained models
    - Fitted preprocessor
    - Feature names and test splits

    Sessions expire after 2 hours of inactivity (cleaned by background task).
"""

import time
import threading
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers import upload, train, predict, explain, simulate

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("qsand")

# ---- FastAPI App ----
app = FastAPI(
    title="Q-SAND API",
    description="Quantum-Inspired Network Anomaly Detection Dashboard API",
    version="1.0",
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Session Storage ----
# In-memory session dict: session_id → {dataframe, models, preprocessor, ...}
# Each session entry also has a 'last_accessed' timestamp for cleanup.
SESSIONS: dict = {}

# Session timeout: 2 hours
SESSION_TIMEOUT = 2 * 60 * 60  # seconds


def _cleanup_expired_sessions():
    """Background thread that removes sessions inactive for > 2 hours."""
    while True:
        time.sleep(300)  # Check every 5 minutes
        now = time.time()
        expired = [
            sid for sid, data in SESSIONS.items()
            if now - data.get("last_accessed", now) > SESSION_TIMEOUT
        ]
        for sid in expired:
            del SESSIONS[sid]
            logger.info(f"Cleaned up expired session: {sid}")


# Start cleanup thread
cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
cleanup_thread.start()

# ---- Inject Sessions into Routers ----
upload.set_sessions(SESSIONS)
train.set_sessions(SESSIONS)
predict.set_sessions(SESSIONS)
explain.set_sessions(SESSIONS)
simulate.set_sessions(SESSIONS)

# ---- Include Routers ----
app.include_router(upload.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(simulate.router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "app": "Q-SAND API",
        "version": "1.0",
        "status": "running",
        "active_sessions": len(SESSIONS),
    }


@app.get("/api/health")
async def health():
    """Health check endpoint for frontend."""
    return {"status": "ok", "sessions": len(SESSIONS)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
