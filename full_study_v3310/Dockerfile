# LatBOND v3.3.10 — Vertex AI Training Container
# FIXES: Stepped geometric curriculum + deployment-aligned validation
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest

ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt torch==2.1.0 torchvision==0.16.0 && pip install --no-cache-dir --no-deps torchaudio==2.1.0
COPY . .
ENTRYPOINT ["python", "-u", "scripts/vertex_entrypoint.py"]
