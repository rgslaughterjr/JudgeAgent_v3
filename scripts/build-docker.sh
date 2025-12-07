#!/bin/bash
# Build and push Docker image to ECR

set -e

REPO_NAME=${1:-judge-agent}
REGION=${2:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME"

echo "🐳 Building and pushing Docker image to ECR"
echo "============================================"
echo "Repository: $ECR_URI"
echo ""

# Create ECR repository if it doesn't exist
echo "📦 Creating ECR repository (if needed)..."
aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $REPO_NAME --region $REGION

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build image
echo "🔨 Building Docker image..."
docker build -t $REPO_NAME:latest .

# Tag for ECR
echo "🏷️ Tagging image..."
docker tag $REPO_NAME:latest $ECR_URI:latest
docker tag $REPO_NAME:latest $ECR_URI:$(date +%Y%m%d-%H%M%S)

# Push to ECR
echo "📤 Pushing to ECR..."
docker push $ECR_URI:latest
docker push $ECR_URI:$(date +%Y%m%d-%H%M%S)

echo ""
echo "✅ Image pushed successfully!"
echo "   URI: $ECR_URI:latest"
