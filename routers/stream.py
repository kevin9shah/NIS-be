"""
Live Data Stream Router
======================

WebSocket endpoint for real-time data ingestion and anomaly detection.
Each packet is preprocessed, scored, and run through QTTA in real time.
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import uuid
import pandas as pd
from core.qtta import QTTAThreshold

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["stream"])

SESSIONS = None


def set_sessions(sessions):
    global SESSIONS
    SESSIONS = sessions


@router.websocket("/stream")
async def stream_data(websocket: WebSocket):
    """
    WebSocket endpoint for live anomaly detection.

    Protocol:
      1. Client sends init message: {session_id, model_id, qtta_params?}
      2. Server replies: {status: "ready"}
      3. Client sends raw feature packets as JSON repeatedly
      4. Server replies per-packet: {anomaly_prob, qtta_threshold, is_anomaly, ...}
    """
    await websocket.accept()
    ws_id = str(uuid.uuid4())

    try:
        # Step 1: Handshake — get session + model refs
        init_msg = await websocket.receive_json()
        session_id = init_msg.get("session_id")
        model_id = init_msg.get("model_id")

        session = SESSIONS.get(session_id)
        if not session:
            await websocket.send_json({"error": "Session not found. Train a model first."})
            await websocket.close()
            return

        model_entry = session.get("models", {}).get(model_id)
        if not model_entry:
            await websocket.send_json({"error": "Model not found. Provide a valid model_id."})
            await websocket.close()
            return

        model = model_entry["model"]
        preprocessor = session.get("preprocessor")
        var_selector = session.get("var_selector")  # May be None for older sessions

        # Use the F1-optimal threshold found during training (falls back to 0.4)
        optimal_threshold = session.get("optimal_threshold", 0.4)

        # QTTA params — client can override base_threshold; otherwise use optimal
        qtta_params = init_msg.get("qtta_params", {})
        base_threshold = qtta_params.get("base_threshold", optimal_threshold)
        alpha = qtta_params.get("alpha", 0.3)
        d = qtta_params.get("d", 1.0)
        qtta = QTTAThreshold(base_threshold=base_threshold, alpha=alpha, d=d)

        await websocket.send_json({"status": "ready", "optimal_threshold": round(optimal_threshold, 4)})
        logger.info(f"[{ws_id}] Stream ready — model={model_id[:8]} threshold={base_threshold:.4f}")

        # Step 3+: Process incoming packets
        while True:
            raw = await websocket.receive_json()

            try:
                df = pd.DataFrame([raw])

                # Preprocess with fitted pipeline
                processed = preprocessor.transform(df)
                if "label" in processed.columns:
                    processed = processed.drop(columns=["label"])

                # Apply variance selection if it was used during training
                if var_selector is not None:
                    import pandas as _pd
                    sel_cols = [preprocessor.feature_names[i] for i in var_selector.get_support(indices=True)
                                if i < len(preprocessor.feature_names)]
                    # Keep only selected columns that exist
                    keep = [c for c in sel_cols if c in processed.columns]
                    processed = processed[keep] if keep else processed

                # Score
                proba = float(model.predict_proba(processed)[:, 1][0])
                is_anomaly, threshold, T = qtta.update(proba)

                state = qtta.threshold_history[-1]
                result = {
                    "anomaly_prob":    round(proba, 4),
                    "qtta_threshold":  round(float(threshold), 4),
                    "is_anomaly":      bool(is_anomaly),
                    "tunneling_prob":  round(float(T), 4),
                    "alert_pressure":  round(float(state["alert_pressure"]), 4),
                    "noise_floor":     round(float(state["noise_floor"]), 4),
                }

            except Exception as e:
                logger.warning(f"[{ws_id}] Packet processing failed: {e}")
                result = {"error": str(e), "raw": raw}

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"[{ws_id}] WebSocket disconnected")
