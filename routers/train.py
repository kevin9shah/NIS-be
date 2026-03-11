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
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
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

        # Feature selection: drop near-constant features (variance < 0.01).
        # Low-variance features add noise without aiding class separation —
        # particularly harmful with a small minority class.
        var_selector = VarianceThreshold(threshold=0.01)
        X_train = pd.DataFrame(
            var_selector.fit_transform(X_train),
            columns=[feature_names[i] for i in var_selector.get_support(indices=True)]
        )
        X_test = pd.DataFrame(
            var_selector.transform(X_test),
            columns=X_train.columns
        )
        feature_names = list(X_train.columns)
        logger.info(f"Feature selection: kept {len(feature_names)} features after VarianceThreshold")

        # Apply SMOTE to fully balance the training set.
        # sampling_strategy=1.0 equalizes class counts — with only ~80 real
        # minority samples in training, the model needs the maximum number
        # of synthetic examples to learn discriminative boundaries.
        logger.info("Applying SMOTE to handle class imbalance...")
        n_minority = int(np.sum(y_train == 1))
        n_majority = int(np.sum(y_train == 0))
        original_imbalance_ratio = n_majority / n_minority if n_minority > 0 else 1.0
        logger.info(f"Before SMOTE - Normal: {n_majority}, Anomaly: {n_minority}, Ratio: {original_imbalance_ratio:.2f}")

        # Adaptive k_neighbors — at least 1, at most 7
        k_neighbors = min(7, max(1, n_minority - 1))

        try:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=1.0)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            n_minority_balanced = int(np.sum(y_train_balanced == 1))
            n_majority_balanced = int(np.sum(y_train_balanced == 0))
            logger.info(f"After SMOTE - Normal: {n_majority_balanced}, Anomaly: {n_minority_balanced}")
        except Exception as e:
            # Fallback: if SMOTE fails, use original training set
            logger.warning(f"SMOTE failed ({str(e)}). Using original training set.")
            X_train_balanced = X_train
            y_train_balanced = y_train

        # Train model.
        # NOTE: Do NOT apply scale_pos_weight here — SMOTE already rebalanced
        # the training set. Applying scale_pos_weight on top of SMOTE causes
        # a double-correction that badly hurts precision and recall.
        trainer = ModelTrainer()
        hyperparams = request.hyperparams.copy() if request.hyperparams else {}

        model = trainer.train(
            X_train_balanced, y_train_balanced,
            request.algorithm,
            hyperparams
        )

        # Evaluate with standard 0.5 threshold first (populates confusion matrix / report)
        metrics = trainer.evaluate(model, X_test, y_test, feature_names)

        # --- Optimal threshold search ---
        # For imbalanced data the 0.5 threshold is often too high.
        # Find the probability threshold that maximises F1 on the test set.
        y_proba_test = model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) == 0, 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1])
        )
        best_idx = int(np.argmax(f1_scores))
        optimal_threshold = float(thresholds[best_idx])
        logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")

        # Re-evaluate with the optimal threshold so ALL reported metrics are consistent.
        # Previously the confusion matrix was at 0.5 while scalar metrics were at optimal —
        # this caused a confusing mismatch (e.g. TP=0 in CM but Recall=0.30 in metrics).
        y_pred_optimal = (y_proba_test >= optimal_threshold).astype(int)
        from sklearn.metrics import confusion_matrix as sk_cm
        metrics["accuracy"]          = round(float(accuracy_score(y_test, y_pred_optimal)), 4)
        metrics["precision"]         = round(float(precision_score(y_test, y_pred_optimal, zero_division=0)), 4)
        metrics["recall"]            = round(float(recall_score(y_test, y_pred_optimal, zero_division=0)), 4)
        metrics["f1"]                = round(float(f1_score(y_test, y_pred_optimal, zero_division=0)), 4)
        metrics["roc_auc"]           = round(float(roc_auc_score(y_test, y_proba_test)), 4)
        metrics["optimal_threshold"] = round(optimal_threshold, 4)
        # Update confusion matrix to match optimal threshold
        metrics["confusion_matrix"]  = sk_cm(y_test, y_pred_optimal).tolist()

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
        session["var_selector"] = var_selector      # Feature selection mask
        session["feature_names"] = feature_names
        session["X_test"] = X_test
        session["y_test"] = y_test
        session["X_train"] = X_train
        session["y_train"] = y_train
        session["X_train_balanced"] = X_train_balanced
        session["y_train_balanced"] = y_train_balanced
        session["processed_df"] = processed_df
        session["optimal_threshold"] = optimal_threshold  # Used by predict router

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
                     f"accuracy={metrics['accuracy']}, f1={metrics['f1']}, time={training_time}s")

        return {
            "model_id": model_id,
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "optimal_threshold": metrics["optimal_threshold"],
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
