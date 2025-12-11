#!/bin/bash
set -e  # Exit on any error

echo "🚀 Starting deployment to Google Cloud Run..."
echo ""

# Configuration
PROJECT_ID="cyclemore"
REGION="europe-west2"
API_IMAGE="gcr.io/${PROJECT_ID}/cyclemore-backend:latest"
FRONTEND_IMAGE="gcr.io/${PROJECT_ID}/cyclemore-frontend:latest"

# Build API Docker image (with correct platform for Cloud Run)
echo "📦 Building API Docker image for linux/amd64..."
docker build --platform linux/amd64 \
  -f api_faf/Dockerfile \
  -t ${API_IMAGE} .

# Build Frontend Docker image (with correct platform for Cloud Run)
echo "📦 Building Frontend Docker image for linux/amd64..."
docker build --platform linux/amd64 \
  -f /Users/robford/code/faf_frontend/faf_frontend/Dockerfile \
  -t ${FRONTEND_IMAGE} \
  /Users/robford/code/faf_frontend/faf_frontend

# Push images to Google Container Registry
echo "⬆️  Pushing API image to GCR..."
docker push ${API_IMAGE}

echo "⬆️  Pushing Frontend image to GCR..."
docker push ${FRONTEND_IMAGE}

# Deploy API to Cloud Run
echo "🚢 Deploying API to Cloud Run..."
gcloud run deploy cyclemore-backend \
  --image ${API_IMAGE} \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID}

# Deploy Frontend to Cloud Run
echo "🚢 Deploying Frontend to Cloud Run..."
gcloud run deploy cyclemore-frontend \
  --image ${FRONTEND_IMAGE} \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID}

echo ""
echo "✅ Deployment complete!"
echo "API: https://cyclemore-backend-${REGION}.run.app"
echo "Frontend: https://cyclemore-frontend-${REGION}.run.app"
