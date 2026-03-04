"""
Dataset Upload Router
=====================

POST /api/upload — Accepts CSV file upload, validates columns,
stores dataset in session, returns preview and stats.
"""

import uuid
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from io import StringIO
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

# Reference to globally shared sessions dict (set in main.py)
SESSIONS = None


def set_sessions(sessions):
    global SESSIONS
    SESSIONS = sessions


REQUIRED_COLUMNS = [
    "packet_size",
    "inter_arrival_time",
    "src_port",
    "dst_port",
    "packet_count_5s",
    "mean_packet_size",
    "spectral_entropy",
    "frequency_band_energy",
    "protocol_type_TCP",
]


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset for analysis.

    Validates that required columns are present. The 'label' column
    is required for training but optional for inference.

    Returns session_id and dataset summary.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are accepted")

    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {str(e)}")

    # Validate required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing}"
        )

    session_id = str(uuid.uuid4())

    # Convert boolean columns to int
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Label distribution
    label_dist = {"normal": 0, "anomaly": 0}
    if "label" in df.columns:
        label_dist["normal"] = int((df["label"] == 0).sum())
        label_dist["anomaly"] = int((df["label"] == 1).sum())

    # Preview (first 10 rows)
    preview = df.head(10).to_dict(orient="records")

    # Store in session
    SESSIONS[session_id] = {
        "dataframe": df,
        "models": {},
        "preprocessor": None,
        "feature_names": None,
    }

    logger.info(f"Session {session_id}: uploaded {len(df)} rows, {len(df.columns)} cols")

    return {
        "session_id": session_id,
        "rows": len(df),
        "columns": list(df.columns),
        "label_distribution": label_dist,
        "preview": preview,
    }
