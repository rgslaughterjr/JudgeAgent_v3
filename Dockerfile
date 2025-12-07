# Judge Agent v3 - Lambda Container Image
# Uses AWS Lambda Python 3.11 base image

FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code - new src/ structure
COPY src/judge_agent ./judge_agent/
COPY lambda/lambda_handler.py .

# Set the handler
CMD ["lambda_handler.handler"]
