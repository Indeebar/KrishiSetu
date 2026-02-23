"""
GET /api/shap?state=<state>&waste_type=<waste_type>
Returns SHAP feature contribution values explaining why the RF model
predicted a specific price for the given inputs.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_rf_pipeline = None
_explainer    = None


def get_explainer():
    global _rf_pipeline, _explainer
    if _explainer is None:
        model_path = Path(__file__).parent.parent.parent / "models" / "price_predictor.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"RF model not found at {model_path}")

        _rf_pipeline = joblib.load(str(model_path))

        # Build the explainer against the preprocessor output
        # We use a small background dataset (just zeros — fine for TreeExplainer)
        preprocessor = _rf_pipeline.named_steps["preprocessor"]
        rf_model      = _rf_pipeline.named_steps["model"]

        # Create a tiny background sample to initialise the explainer
        background = pd.DataFrame([{
            "State": "Maharashtra",
            "Agricultural Waste Type": "Rice_straw",
        }])
        background_transformed = preprocessor.transform(background)
        # Convert sparse matrix from OneHotEncoder to dense array for SHAP
        if hasattr(background_transformed, "toarray"):
            background_transformed = background_transformed.toarray()
            
        _explainer = shap.TreeExplainer(rf_model, background_transformed)

    return _rf_pipeline, _explainer


@router.get("/shap")
def shap_explain(
    state: str = Query(..., description="Indian state name"),
    waste_type: str = Query(..., description="Waste class as predicted by CNN"),
):
    """
    Returns SHAP values explaining the price prediction.
    Response: { base_value, shap_contributions: [{ feature, value, shap_value }] }
    """
    try:
        pipeline, explainer = get_explainer()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    input_df = pd.DataFrame([{
        "State": state,
        "Agricultural Waste Type": waste_type,
    }])

    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        X_transformed = preprocessor.transform(input_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
            
        shap_values   = explainer.shap_values(X_transformed)
        base_value    = float(explainer.expected_value)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"SHAP computation failed: {e}")

    # Get feature names from the one-hot encoder
    feature_names = preprocessor.get_feature_names_out().tolist()
    shap_row      = shap_values[0].tolist()

    # Return only non-zero contributions, sorted by absolute impact
    contributions = [
        {"feature": fname.replace("cat__", ""), "shap_value": round(sv, 4)}
        for fname, sv in zip(feature_names, shap_row)
        if abs(sv) > 0.001
    ]
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "state":            state,
        "waste_type":       waste_type,
        "base_value":       round(base_value, 2),
        "shap_contributions": contributions[:10],  # top 10 features
    }
