FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libtesseract-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/ .

# Was missing entirely — the image built successfully but had no
# dependencies installed at all (fastapi, qdrant-client, sentence-
# transformers... none of it).
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Was `CMD ["sleep", "infinity"]` — a debug placeholder that never started
# the server. The container ran "successfully" (docker-compose reported it
# as Up) while doing absolutely nothing.
CMD ["python", "-m", "app.main"]