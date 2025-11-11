# Jetson L4T ML image with CUDA, cuDNN, TensorRT, Python preinstalled
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (TensorRT comes from the base image, DO NOT pip-install tensorrt)
COPY app/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./

# Models will be mounted at /models
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python3", "server.py", "--engine", "/models/yolov8n_fp16.engine", "--host", "0.0.0.0", "--port", "8000"]
