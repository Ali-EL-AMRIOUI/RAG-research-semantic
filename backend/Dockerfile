FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libtesseract-dev \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/ .

# Was missing entirely — the image built successfully but had no
# dependencies installed at all (fastapi, qdrant-client, sentence-
# transformers... none of it).
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Runs uvicorn directly rather than `python -m app.main`, so the process
# is PID 1 and receives SIGTERM directly from Docker/Kubernetes for a
# clean shutdown, instead of it being swallowed by a wrapping Python
# interpreter.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]