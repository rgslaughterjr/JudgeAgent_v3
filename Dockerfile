# Judge Agent v3 - Lambda Container Image
# Uses AWS Lambda Python 3.11 base image

FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first for layer caching
COPY requirements_langchain.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_langchain.txt

# Copy application code
COPY judge_agent_langchain.py .
COPY judge_agent_supervisor.py .
COPY judge_agent_enhanced_dimensions.py .
COPY lambda/lambda_handler.py .
COPY utils/ ./utils/

# Set the handler
CMD ["lambda_handler.handler"]
