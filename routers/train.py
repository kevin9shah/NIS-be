"""
Model Training Router
=====================

POST /api/train — Train a classification model on the uploaded dataset.
Supports XGBoost, Random Forest, and Gradient Boosting with configurable
hyperparameters and test split size.
"""

import time
import uuid
import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.model_selection import train_test_split

from schemas.requests import TrainRequest
from core.preprocessor import NetworkPreprocessor
from core.models import ModelTrainer

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["train"])

SESSIONS = None


def set_sessions(sessions):
    global SESSIONS
    SESSIONS = sessions


@router.post("/train")
async def train_model(request: TrainRequest):
    """
    Train a classification model on the uploaded dataset.

    Performs preprocessing, train/test split, model training,
    and evaluation. Returns metrics, feature importances, and
    confusion matrix.
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Upload a dataset first.")

    df = session["dataframe"]
    if "label" not in df.columns:
        raise HTTPException(status_code=422, detail="Dataset must contain 'label' column for training.")

    try:
        start_time = time.time()

        # Preprocess
        preprocessor = NetworkPreprocessor()
        processed_df, feature_names = preprocessor.fit_transform(df)

        # Separate features and labels
        X = processed_df.drop(columns=["label"])
        y = processed_df["label"].astype(int).values

        # Ensure feature names don't include label
        feature_names = [f for f in feature_names if f != "label"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42, stratify=y
        )

        # Train model
        trainer = ModelTrainer()
        model = trainer.train(
            X_train, y_train,
            request.algorithm,
            request.hyperparams
        )

        # Evaluate
        metrics = trainer.evaluate(model, X_test, y_test, feature_names)

        # Save model
        model_path = trainer.save_model(model, request.algorithm)
        model_id = str(uuid.uuid4())

        training_time = round(time.time() - start_time, 2)

        # Store in session
        session["models"][model_id] = {
            "model": model,
            "algorithm": request.algorithm,
            "model_path": model_path,
        }
        session["preprocessor"] = preprocessor
        session["feature_names"] = feature_names
        session["X_test"] = X_test
        session["y_test"] = y_test
        session["X_train"] = X_train
        session["y_train"] = y_train
        session["processed_df"] = processed_df

        # Class distribution
        class_dist = {
            "train_normal": int(np.sum(y_train == 0)),
            "train_anomaly": int(np.sum(y_train == 1)),
            "test_normal": int(np.sum(y_test == 0)),
            "test_anomaly": int(np.sum(y_test == 1)),
        }

        logger.info(f"Training complete: {request.algorithm}, "
                     f"accuracy={metrics['accuracy']}, time={training_time}s")

        return {
            "model_id": model_id,
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            },
            "confusion_matrix": metrics["confusion_matrix"],
            "feature_importances": metrics["feature_importances"],
            "training_time_seconds": training_time,
            "class_distribution": class_dist,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
