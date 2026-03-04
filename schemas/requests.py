"""
Pydantic v2 schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class TrainRequest(BaseModel):
    """Request body for POST /api/train"""
    session_id: str
    algorithm: str = Field(
        default="xgboost",
        pattern="^(xgboost|random_forest|gradient_boosting)$",
        description="ML algorithm to train"
    )
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    hyperparams: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    """Request body for POST /api/predict"""
    session_id: str
    model_id: str
    qtta_params: Optional[Dict[str, float]] = Field(
        default=None,
        description="QTTA parameters: base_threshold, alpha, d"
    )


class ExplainRequest(BaseModel):
    """Request body for POST /api/explain"""
    session_id: str
    model_id: str


class SimulateRequest(BaseModel):
    """Request body for POST /api/simulate"""
    session_id: str
    model_id: str
    vary_packet_count: List[int] = Field(
        default=[5, 10, 20, 50, 100, 200]
    )
    vary_mean_packet_size: List[int] = Field(
        default=[64, 128, 256, 512, 1024, 1500]
    )
    qtta_params: Optional[Dict[str, float]] = None


class QTTAParams(BaseModel):
    """QTTA configuration parameters"""
    base_threshold: float = Field(default=0.5, ge=0.1, le=0.95)
    alpha: float = Field(default=0.3, ge=0.1, le=0.5)
    d: float = Field(default=1.0, ge=0.5, le=3.0)
