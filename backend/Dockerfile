# KrishiSetu Backend — Dockerfile
FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy model files and dataset class structure
COPY models/ ./models/
COPY Agri_Waste_Images/ ./Agri_Waste_Images/
COPY Bio_WASTE_Cleaned_Mapped.csv .

# Copy source
COPY backend/ ./backend/
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
