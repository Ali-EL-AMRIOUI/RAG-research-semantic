FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libtesseract-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/ .

CMD ["sleep", "infinity"]