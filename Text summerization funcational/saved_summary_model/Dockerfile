# ---- Base image ----
FROM python:3.12-slim

# ---- Working directory ----
WORKDIR /app

# ---- System deps (needed by torch/sentencepiece build chains) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (cached layer: only re-runs if requirements.txt changes) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- App code + model files ----
# Copy everything: app.py, index.html, config.json, generation_config.json,
# tokenizer_config.json, spiece.model, model.safetensors
COPY . .

# ---- Runtime ----
ENV PORT=8000
EXPOSE 8000

# No --reload in production; single worker is usually fine for a small model,
# increase --workers if you have more CPU cores and traffic.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
