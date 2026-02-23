# 🌾 KrishiSetu | Agricultural Waste Valuation System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React / Vanilla JS](https://img.shields.io/badge/Frontend-TailwindCSS-38B2AC?style=flat&logo=tailwind-css)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

KrishiSetu is an end-to-end Machine Learning and Deep Learning web application designed to help farmers identify classify agricultural waste from images and estimate its market value.

This project goes beyond a simple predictive script by incorporating **Production ML Serving (FastAPI)**, **Explainable AI / XAI (Grad-CAM & SHAP)**, and a **Modern UI (Tailwind CSS)**.

---

## 🌟 Demo & Live Links

- **Frontend (Vercel):** *[Link your Vercel deployment here]*
- **Backend API Docs (Render):** *[Link your Render /docs URL here]*

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────┐
│           HTML5/JS Frontend              │
│  - Drag & Drop Image Upload             │
│  - Results Dashboard                    │
│  - XAI Heatmap & SHAP Charts            │
│  Deploy: Vercel (free)                  │
└──────────────────┬──────────────────────┘
                   │ REST API calls (CORS)
┌──────────────────▼──────────────────────┐
│          FastAPI Backend (Python)        │
│  POST /api/predict  ← CNN inference     │
│  POST /api/gradcam  ← Heatmap overlay   │
│  GET  /api/price    ← Random Forest     │
│  GET  /api/shap     ← Feature Explainer │
│  GET  /api/history  ← SQLite DB logging │
│  Deploy: Render (Docker free tier)      │
└────────┬────────┬────────┬──────────────┘
         │        │        │
    ┌────▼────┐ ┌─▼─┐  ┌───▼──────────┐
    │  Keras  │ │DB │  │ scikit-learn │
    │CNN Model│ │   │  │  RF Pipeline │
    └─────────┘ └───┘  └──────────────┘
```

---

## 🚀 Key Features

1. **Deep Learning Image Classification (CNN):** Custom trained model on top of `MobileNetV2` to accurately classify 10+ agricultural waste types from noisy images.
2. **Predictive Pricing (Random Forest):** Evaluates waste type alongside geographical data (Indian State) to predict local market value per kg.
3. **Deep Learning Explainability (Grad-CAM):** Generates transparent heatmaps showing exactly which pixel regions the CNN looked at to make its classification, proving the model isn't just memorizing backgrounds.
4. **Machine Learning Explainability (SHAP):** Calculates Shapley values for the Random Forest pipeline to explain *why* a specific price was predicted (e.g., State=+₹3.0, WasteType=-₹1.0).
5. **RESTful ML Serving:** Hosted via FastAPI with strict pydantic validation and auto-generated Swagger UI.
6. **Robust Testing:** Backed by a `pytest` suite ensuring all endpoints are stable.

---

## 🛠️ Tech Stack

**Frontend:** Vanilla HTML5, JavaScript, TailwindCSS, Chart.js  
**Backend:** Python 3.11, FastAPI, Uvicorn, SQLite  
**Machine Learning / AI:** TensorFlow/Keras, Scikit-Learn, SHAP, OpenCV  
**MLOps & Deployment:** Docker, Docker-Compose, Vercel, Render

---

## 💻 Running Locally

### 1. Prerequisites
- Python 3.11+
- Virtual Environment tool (venv, conda)
- Docker (optional, for containerised run)

### 2. Setup standard environment
```bash
# Clone the repository
git clone https://github.com/yourusername/KrishiSetu.git
cd KrishiSetu

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
```

### 3. Run the Backend API
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at `http://localhost:8000`. You can test endpoints via Swagger UI at `http://localhost:8000/docs`.

### 4. Run the Frontend
Since the frontend uses basic HTML and absolute/relative paths, simply open `frontend/index.html` in your web browser. Or serve it using python:
```bash
cd frontend
python -m http.server 3000
```
Then visit `http://localhost:3000`.

---

## 🐳 Running with Docker

You can spin up the full backend instantly using Docker Compose:
```bash
docker-compose up --build
```
This mounts the `models/` directory into the container and spins up the FastAPI backend on port 8000.

---

## 🧪 Testing

To run the unit test suite:
```bash
# Ensure you are at the project root
set PYTHONPATH=.
pytest tests/ -v
```

---

## 🚀 Deployment Guide (For the specified Architecture)

### 1. Frontend on Vercel
1. Push this repository to GitHub.
2. Go to [Vercel](https://vercel.com/) and create a new project.
3. Import your GitHub repository.
4. Set the **Root Directory** to `frontend`.
5. Click **Deploy**. (It will deploy instantly as a static site).

### 2. Backend on Render
1. Go to [Render](https://render.com/) and create a new **Web Service**.
2. Connect your GitHub repository.
3. Choose the **Docker** runtime.
4. Set the Dockerfile path to `backend/Dockerfile`.
5. Select the **Free** instance type.
6. Click **Create Web Service**. 
   - Note: *The free tier spins down after 15 minutes of inactivity. The first request after a sleep period may take ~30-50 seconds to complete.*

---
*Note: This project demonstrates end-to-end full-stack machine learning engineering, transitioning a local data science prototype into a scalable web product.*
