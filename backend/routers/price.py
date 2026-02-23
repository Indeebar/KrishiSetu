"""
GET /api/price?state=<state>&waste_type=<waste_type>
Returns the predicted market price per kg using the Random Forest pipeline.
"""

import joblib
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_rf_pipeline = None


def get_pipeline():
    global _rf_pipeline
    if _rf_pipeline is None:
        model_path = Path(__file__).parent.parent.parent / "models" / "price_predictor.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"RF model not found at {model_path}")
        _rf_pipeline = joblib.load(str(model_path))
    return _rf_pipeline


@router.get("/price")
def price(
    state: str = Query(..., description="Indian state name"),
    waste_type: str = Query(..., description="Waste class as predicted by CNN"),
):
    """
    Predict market price per kg for a given state and waste type.
    Returns: { price_per_kg: float }
    """
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    input_df = pd.DataFrame([{
        "State": state,
        "Agricultural Waste Type": waste_type,
    }])

    try:
        predicted_price = float(pipeline.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Price prediction failed: {e}")

    return {
        "state":        state,
        "waste_type":   waste_type,
        "price_per_kg": round(predicted_price, 2),
    }
