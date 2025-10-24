
#!/bin/bash
set -e  # abort on error

IMAGE_URI=$1

echo ">>> Starting deployment for image: $IMAGE_URI"

if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo ">>> Activating GCP service account..."
  gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
  echo ">>> GCP authentication successful"
else
  echo ">>> ERROR: Credentials file not found at $GOOGLE_APPLICATION_CREDENTIALS"
  exit 1
fi

echo ">>> Logging in to AWS ECR.."
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 034154586122.dkr.ecr.eu-north-1.amazonaws.com

echo ">>> Stopping existing container (if exists)..."
docker stop nlp-container || true
docker rm nlp-container || true

echo ">>> Removing unused images..."
docker system prune -af || true

echo ">>> Pulling new image..."
docker pull $IMAGE_URI

echo ">>> Running new container..."
docker run -d --name nlp-container -p 80:80 $IMAGE_URI

echo ">>> Deployment complete. Checking status..."
docker ps

