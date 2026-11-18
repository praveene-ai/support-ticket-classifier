from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_tfidf_logreg.joblib"


class PredictRequest(BaseModel):
    text: str = Field(..., description="Support ticket description text")

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Ticket text must not be empty or whitespace only.")
        return cleaned


class PredictResponse(BaseModel):
    category: str
    priority: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    artifact: dict[str, Any] = joblib.load(MODEL_PATH)
    app.state.category_model = artifact["category_pipeline"]
    app.state.priority_model = artifact["priority_pipeline"]
    yield


app = FastAPI(
    title="Support Ticket Classifier API",
    description="Predicts ticket category, priority, and confidence score.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check(request: Request) -> dict[str, Any]:
    models_loaded = hasattr(request.app.state, "category_model") and hasattr(
        request.app.state, "priority_model"
    )
    return {
        "status": "ok" if models_loaded else "degraded",
        "models_loaded": models_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_ticket(payload: PredictRequest, request: Request) -> PredictResponse:
    category_model = getattr(request.app.state, "category_model", None)
    priority_model = getattr(request.app.state, "priority_model", None)
    if category_model is None or priority_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    texts = [payload.text]

    cat_probs = category_model.predict_proba(texts)[0]
    cat_idx = int(np.argmax(cat_probs))
    category = str(category_model.classes_[cat_idx])
    cat_conf = float(cat_probs[cat_idx])

    prio_probs = priority_model.predict_proba(texts)[0]
    prio_idx = int(np.argmax(prio_probs))
    priority = str(priority_model.classes_[prio_idx])
    prio_conf = float(prio_probs[prio_idx])

    confidence = round((cat_conf + prio_conf) / 2.0, 4)

    return PredictResponse(
        category=category,
        priority=priority,
        confidence=confidence,
    )