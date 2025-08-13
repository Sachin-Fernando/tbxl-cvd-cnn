FROM python:3.11-slim

# System deps (libgomp for xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
# Place your models under ./models in the build context
COPY models /app/models

# non-root for security
RUN useradd -m appuser
USER appuser

EXPOSE 8080
ENV PORT=8080
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8080"]
