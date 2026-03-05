"""
Model Training Router
=====================

POST /api/train — Train a classification model on the uploaded dataset.
Supports XGBoost, Random Forest, and Gradient Boosting with configurable
hyperparameters and test split size.

Class Imbalance Handling:
    Uses SMOTE (Synthetic Minority Over-sampling Technique) to balance
    the training set, improving minority class detection.
"""

import time
import uuid
import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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

        # Apply SMOTE to balance the training set
        # This generates synthetic samples of the minority class
        logger.info(f"Applying SMOTE to handle class imbalance...")
        n_minority = np.sum(y_train == 1)
        n_majority = np.sum(y_train == 0)
        original_imbalance_ratio = n_majority / n_minority if n_minority > 0 else 1.0
        logger.info(f"Before SMOTE - Normal: {n_majority}, Anomaly: {n_minority}, Ratio: {original_imbalance_ratio:.2f}")
        
        # Use adaptive k_neighbors (min 1, typically 3-5)
        k_neighbors = min(3, max(1, n_minority - 1))
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            n_minority_balanced = np.sum(y_train_balanced == 1)
            n_majority_balanced = np.sum(y_train_balanced == 0)
            logger.info(f"After SMOTE - Normal: {n_majority_balanced}, Anomaly: {n_minority_balanced}")
        except Exception as e:
            # Fallback: if SMOTE fails, use original training set with class weights
            logger.warning(f"SMOTE failed ({str(e)}). Using class_weight instead.")
            X_train_balanced = X_train
            y_train_balanced = y_train

        # Train model with explicit scale_pos_weight to boost minority class confidence
        trainer = ModelTrainer()
        hyperparams = request.hyperparams.copy() if request.hyperparams else {}
        
        # For XGBoost, use original imbalance ratio even when training on balanced data
        # This ensures high confidence predictions for the minority class
        if request.algorithm == "xgboost" and "scale_pos_weight" not in hyperparams:
            hyperparams["scale_pos_weight"] = original_imbalance_ratio
            logger.info(f"Set scale_pos_weight={original_imbalance_ratio:.2f} for XGBoost")
        
        model = trainer.train(
            X_train_balanced, y_train_balanced,
            request.algorithm,
            hyperparams
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
        session["X_train"] = X_train  # Original training set
        session["y_train"] = y_train  # Original labels
        session["X_train_balanced"] = X_train_balanced  # SMOTE-balanced training set
        session["y_train_balanced"] = y_train_balanced  # SMOTE-balanced labels
        session["processed_df"] = processed_df

        # Class distribution (original split, before SMOTE)
        class_dist = {
            "train_normal_original": int(np.sum(y_train == 0)),
            "train_anomaly_original": int(np.sum(y_train == 1)),
            "train_normal_after_smote": int(np.sum(y_train_balanced == 0)),
            "train_anomaly_after_smote": int(np.sum(y_train_balanced == 1)),
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
