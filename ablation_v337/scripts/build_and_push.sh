#!/bin/bash
# Build and push LatBOND training container to Artifact Registry
# Usage: bash scripts/build_and_push.sh

set -e

PROJECT_ID="YOUR_GCP_PROJECT"
REGION="us-west1"
REPO="latbond"
IMAGE="training"
TAG="latest"

FULL_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "=== LatBOND: Building Docker Container ==="
echo "Image: ${FULL_URI}"
echo ""

# Authenticate Docker with Artifact Registry
echo "[1/3] Authenticating with Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build the image
echo "[2/3] Building Docker image..."
docker build -t ${FULL_URI} -f Dockerfile .

# Push to Artifact Registry
echo "[3/3] Pushing to Artifact Registry..."
docker push ${FULL_URI}

echo ""
echo "=== Container pushed successfully ==="
echo "URI: ${FULL_URI}"
echo ""
echo "To verify:"
echo "  gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"
echo ""
echo "To launch training:"
echo "  python scripts/launch_vertex_ai.py"
