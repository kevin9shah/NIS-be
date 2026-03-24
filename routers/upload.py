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

from schemas.requests import LoadDirectoryRequest
from core.data_loader import load_directory_sampled

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

    session_id = str(uuid.uuid4())

    # Map specific dataset target columns to standard "label"
    target_mappings = {"Attack_label": "label"}
    df = df.rename(columns=target_mappings)
    
    # Drop multiclass string columns to avoid data leakage
    if "Attack_type" in df.columns:
        df = df.drop(columns=["Attack_type"])

    # Intelligently preserve continuous numbers while mathematically encoding true categories
    for col in df.columns:
        if col != "label":
            if df[col].dtype == object or str(df[col].dtype) == 'category':
                # Force coerce to numeric to see how many were actually numbers
                coerced = pd.to_numeric(df[col], errors='coerce')
                # If more than 50% became NaN, it's a true string/categorical column (like IP, Protocol)
                if coerced.isna().sum() > len(df) * 0.5:
                    df[col] = pd.factorize(df[col].astype(str))[0]
                else:
                    # It's a continuous number column with some garbage string rows. Keep coerced floats!
                    df[col] = coerced

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
    preview_df = df.head(10).copy()
    import numpy as np
    preview_df = preview_df.replace({np.nan: None})
    preview = preview_df.to_dict(orient="records")

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


@router.post("/load_directory")
async def load_directory(request: LoadDirectoryRequest):
    """
    Load a dataset from a local directory of CSV files.
    Samples the data to fit into memory.
    """
    try:
        df = load_directory_sampled(request.directory_path, request.sample_frac)
    except Exception as e:
        logger.exception("Failed to load directory")
        raise HTTPException(status_code=422, detail=f"Failed to load directory: {str(e)}")

    session_id = str(uuid.uuid4())

    # Map specific dataset target columns to standard "label"
    target_mappings = {"Attack_label": "label"}
    df = df.rename(columns=target_mappings)
    
    # Drop multiclass string columns to avoid data leakage
    if "Attack_type" in df.columns:
        df = df.drop(columns=["Attack_type"])

    # Intelligently preserve continuous numbers while mathematically encoding true categories
    for col in df.columns:
        if col != "label":
            if df[col].dtype == object or str(df[col].dtype) == 'category':
                # Force coerce to numeric to see how many were actually numbers
                coerced = pd.to_numeric(df[col], errors='coerce')
                # If more than 50% became NaN, it's a true string/categorical column (like IP, Protocol)
                if coerced.isna().sum() > len(df) * 0.5:
                    df[col] = pd.factorize(df[col].astype(str))[0]
                else:
                    # It's a continuous number column with some garbage string rows. Keep coerced floats!
                    df[col] = coerced

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
    preview_df = df.head(10).copy()
    # Replace NaN with None so FastAPI/JSON can serialize it without crashing
    import numpy as np
    preview_df = preview_df.replace({np.nan: None})
    preview = preview_df.to_dict(orient="records")

    # Store in session
    SESSIONS[session_id] = {
        "dataframe": df,
        "models": {},
        "preprocessor": None,
        "feature_names": None,
    }

    logger.info(f"Session {session_id}: loaded {len(df)} rows from directory {request.directory_path}")

    return {
        "session_id": session_id,
        "rows": len(df),
        "columns": list(df.columns),
        "label_distribution": label_dist,
        "preview": preview,
    }
