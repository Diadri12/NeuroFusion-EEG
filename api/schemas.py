from pydantic import BaseModel
from typing import List


class EEGRequest(BaseModel):
    signal: List[float]


class PredictionResponse(BaseModel):
    pred_class: int
    label: str
    confidence: float
    pred_prob: float
