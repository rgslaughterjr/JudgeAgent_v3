#!/bin/bash
# Deploy Judge Agent to AWS using CDK

set -e

ENV=${1:-dev}
REGION=${2:-us-east-1}

echo "🚀 Deploying Judge Agent v3 to AWS ($ENV environment)"
echo "=================================================="

# Check for required tools
command -v cdk >/dev/null 2>&1 || { echo "❌ AWS CDK not installed. Run: npm install -g aws-cdk"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI not installed"; exit 1; }

# Verify AWS credentials
echo "📋 Verifying AWS credentials..."
aws sts get-caller-identity --region $REGION

# Install CDK dependencies
echo "📦 Installing CDK dependencies..."
cd infrastructure
pip install -r requirements.txt

# Bootstrap CDK (if needed)
echo "🔧 Bootstrapping CDK..."
cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/$REGION

# Synthesize CloudFormation template
echo "📝 Synthesizing CloudFormation template..."
cdk synth -c env=$ENV

# Deploy stack
echo "🚀 Deploying stack..."
cdk deploy JudgeAgentDev -c env=$ENV --require-approval never

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 To view the CloudWatch dashboard:"
echo "   aws cloudwatch get-dashboard --dashboard-name JudgeAgent-$ENV"
echo ""
echo "🔗 API endpoint and other outputs:"
cdk outputs JudgeAgentDev
