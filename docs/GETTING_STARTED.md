# Getting Started for Developers

This guide helps you get Judge Agent running and customized for your needs.

## Prerequisites

- Python 3.10+
- AWS account with Bedrock access enabled
- AWS CLI configured (`aws configure`)

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/rgslaughterjr/JudgeAgent_v3.git
cd JudgeAgent_v3
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Ensure your AWS credentials have Bedrock access:

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1)
```

### 3. Verify Setup

```bash
# Quick test (uses MockAgent, no real AWS calls)
cd JudgeAgent_v3
python examples/quick_test.py
```

## Running Evaluations

### Option A: Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Option B: Python Script

```python
import asyncio
import sys
sys.path.insert(0, "src")

from judge_agent.supervisor import JudgeAgentSupervisor, AgentConfig, MockAgent

config = AgentConfig(
    agent_id="test-001",
    name="My Agent",
    framework="langchain",
    risk_level="medium"
)

connector = MockAgent()  # Replace with your agent
judge = JudgeAgentSupervisor(connector)

result = asyncio.run(judge.evaluate_parallel(config))
print(f"Score: {result['overall_score']:.1%}")
```

## Connecting Your Real Agent

### Step 1: Implement AgentConnector

```python
from judge_agent.supervisor import AgentConnector

class MyAgent(AgentConnector):
    async def invoke(self, prompt: str) -> str:
        # Your agent logic here
        response = await my_llm_call(prompt)
        return response
```

### Step 2: Use Your Connector

```python
connector = MyAgent()
judge = JudgeAgentSupervisor(connector)
result = await judge.evaluate_parallel(config)
```

## Customization

### Adjust Thresholds

Edit `src/judge_agent/supervisor.py`:

```python
class Config:
    SECURITY_THRESHOLD = 0.90  # Default: 0.85
    PRIVACY_THRESHOLD = 0.90   # Default: 0.85
    GENERAL_THRESHOLD = 0.75   # Default: 0.70
```

### Select Specific Dimensions

```python
# Only evaluate security and privacy
config = AgentConfig(...)
state = SupervisorState(
    agent_config=config,
    pending_dimensions=[EvaluationDimension.SECURITY, EvaluationDimension.PRIVACY]
)
```

### Customize Test Counts

Controlled by risk level:

| Risk Level | Tests per Dimension |
|------------|---------------------|
| low | 5 |
| medium | 10 |
| high | 15 |
| critical | 20 |

## AWS Deployment

See the main [README.md](../README.md#aws-deployment) for CDK deployment instructions.

## Troubleshooting

### "Bedrock access denied"

1. Verify your AWS credentials: `aws sts get-caller-identity`
2. Ensure Bedrock is enabled in your region
3. Check IAM permissions for `bedrock:InvokeModel`

### Import errors

Add `src/` to your Python path:

```python
import sys
sys.path.insert(0, "src")
```

Or set environment variable:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Next Steps

- See [CONTRIBUTING.md](../CONTRIBUTING.md) for extending the project
- See [docs/utils.md](utils.md) for utility documentation
- Run `pytest tests/ -v` to verify everything works
