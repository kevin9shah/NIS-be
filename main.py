"""
Q-SAND API — Main Entry Point
===============================
Live Network Anomaly Detection backend powered by QTTA.
"""

import time
import threading
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers import upload, train, predict, stream

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("qsand")

# ---- FastAPI App ----
app = FastAPI(
    title="Q-SAND API",
    description="Quantum-Inspired Network Anomaly Detection — Live Detection API",
    version="2.0",
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Session Storage ----
SESSIONS: dict = {}
SESSION_TIMEOUT = 2 * 60 * 60  # 2 hours


def _cleanup_expired_sessions():
    """Background thread: remove sessions inactive for > 2 hours."""
    while True:
        time.sleep(300)
        now = time.time()
        expired = [
            sid for sid, data in SESSIONS.items()
            if now - data.get("last_accessed", now) > SESSION_TIMEOUT
        ]
        for sid in expired:
            del SESSIONS[sid]
            logger.info(f"Cleaned up expired session: {sid}")


cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
cleanup_thread.start()

# ---- Inject Sessions into Routers ----
upload.set_sessions(SESSIONS)
train.set_sessions(SESSIONS)
predict.set_sessions(SESSIONS)
stream.set_sessions(SESSIONS)

# ---- Include Routers ----
app.include_router(upload.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(stream.router)


@app.get("/")
async def root():
    return {"app": "Q-SAND API", "version": "2.0", "status": "running", "active_sessions": len(SESSIONS)}


@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions and model IDs — useful for packet_capture.py."""
    result = []
    for sid, data in SESSIONS.items():
        result.append({
            "session_id": sid,
            "model_ids": list(data.get("models", {}).keys()),
            "has_model": len(data.get("models", {})) > 0,
        })
    return {"sessions": result}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
