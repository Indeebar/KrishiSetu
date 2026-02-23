"""
KrishiSetu FastAPI Backend
==========================
Serves the CNN image classifier, RF price predictor, Grad-CAM heatmaps,
SHAP explainability, and prediction history via a REST API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import predict, gradcam, price, shap_explain, history
from backend.db import init_db

app = FastAPI(
    title="KrishiSetu API",
    description="Agricultural Waste Valuation System — ML/DL Backend",
    version="2.0.0",
)

# Allow requests from the React frontend (Vercel) and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to specific Vercel URL before final deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict.router, prefix="/api")
app.include_router(gradcam.router, prefix="/api")
app.include_router(price.router, prefix="/api")
app.include_router(shap_explain.router, prefix="/api")
app.include_router(history.router, prefix="/api")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"message": "KrishiSetu API is running. Visit /docs for Swagger UI."}
