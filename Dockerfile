FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .    
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ ./src/
COPY data/ ./data/
# COPY run.sh .
COPY tests/ ./tests/

RUN mkdir -p visualizations
