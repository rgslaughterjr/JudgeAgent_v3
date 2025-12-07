# Judge Agent v3

Enterprise AI Agent Evaluation System built with LangChain/LangGraph for AWS Bedrock. Provides comprehensive multi-dimensional assessment of agent performance, safety, and reliability.

## Overview

Judge Agent v3 systematically evaluates AI agents across **8 critical dimensions** before production deployment. It serves as a "production gate" ensuring agents meet safety, security, and quality standards.

## Key Features

- **8-Dimensional Evaluation**: Privacy, Security, Accuracy, Performance, UX, Bias, Harm Prevention, Guardrails
- **LLM vs Agent Control Attribution**: Distinguishes between base model safety and agent-specific controls
- **Parallel Supervisor Architecture**: Concurrent evaluation using LangGraph's supervisor pattern
- **AWS Bedrock Integration**: Native support for Claude and other Bedrock models
- **Streamlit Dashboard**: Interactive UI for running evaluations and viewing results
- **Production-Ready Utilities**: Retry logic, PII sanitization, structured audit logging

## Quick Start

```bash
# Clone the repository
git clone https://github.com/rgslaughterjr/JudgeAgent_v3.git
cd JudgeAgent_v3

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app/streamlit_app.py

# Or use programmatically (add src to PYTHONPATH)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -c "from judge_agent import JudgeAgentSupervisor; print('Ready!')"
```

> **Note**: Requires AWS credentials configured for Bedrock access. See [AWS Deployment](#aws-deployment).

## Architecture

### Repository Structure

```
JudgeAgent_v3/
├── src/judge_agent/           # Core Python package
│   ├── __init__.py            # Package exports
│   ├── langchain.py           # Core LangChain/LangGraph implementation
│   ├── supervisor.py          # Parallel supervisor architecture
│   ├── enhanced_dimensions.py # Specialized evaluators
│   └── utils/                 # Utility modules
├── tests/                     # All test files
├── app/                       # Streamlit UI
├── docs/                      # Additional documentation
├── examples/                  # Example scripts
├── infrastructure/            # AWS CDK stack
├── lambda/                    # Lambda handler
└── scripts/                   # Deployment scripts
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `judge_agent.langchain` | Core LangChain/LangGraph implementation |
| `judge_agent.supervisor` | Parallel supervisor for concurrent evaluation |
| `judge_agent.enhanced_dimensions` | Specialized evaluators (Performance, UX, Bias, Harm, Guardrails) |

### Utilities (`judge_agent.utils`)

| Utility | Purpose |
|---------|---------|
| `json_parser` | Robust JSON extraction from LLM responses |
| `retry` | Async retry decorator with exponential backoff |
| `pii_sanitizer` | PII generation and redaction |
| `audit_log` | JSONL audit logging (local + S3) |
| `observability` | LangSmith tracing, metrics collection |

## Evaluation Dimensions

| Dimension | Description | Key Metrics |
|-----------|-------------|-------------|
| **Privacy** | PII protection, data handling, consent | Data leakage score, compliance |
| **Security** | Prompt injection, jailbreaks, vulnerabilities | Attack resistance, vulnerability count |
| **Accuracy** | Correctness, hallucination detection | Factual accuracy, error rate |
| **Performance** | Latency (p50/p95/p99), throughput | Response time, requests/sec |
| **User Experience** | Clarity, helpfulness, tone, error handling | UX factor scores |
| **Bias** | Demographic fairness, stereotype detection | Quality differential, bias instances |
| **Harm Prevention** | Safety controls, content filtering | Control attribution (LLM vs Agent) |
| **Guardrails** | Boundary enforcement, bypass resistance | Topic/PII/content blocking scores |

## Control Attribution (Key Innovation)

The Harm Prevention evaluator uniquely distinguishes control types:

| Control Type | Source | Example |
|-------------|--------|---------|
| `llm_builtin` | Base model training | Claude refuses "how to make explosives" |
| `agent_specific` | Developer-added logic | Domain-specific content blocking |
| `guardrail_service` | External service | AWS Bedrock Guardrails |

**Why it matters**: If `llm_catches >> agent_catches`, the agent relies too heavily on the base LLM for safety. If `neither_catches > 0`, harmful requests are getting through — **critical issue**.

## Usage Examples

### Programmatic Evaluation

```python
import asyncio
import sys
sys.path.insert(0, "src")  # Add src to path

from judge_agent.supervisor import JudgeAgentSupervisor, AgentConfig, MockAgent

# Configure agent to evaluate
config = AgentConfig(
    agent_id="my-agent-001",
    name="Customer Service Bot",
    framework="langchain",
    risk_level="high",
    description="Handles customer inquiries",
    data_access=["customer_db", "order_history"]
)

# Create connector (use MockAgent for testing, or implement your own)
connector = MockAgent()

# Run evaluation
judge = JudgeAgentSupervisor(connector)
result = asyncio.run(judge.evaluate_parallel(config, evaluator_user="admin@company.com"))

print(f"Overall Score: {result['overall_score']:.1%}")
print(f"Production Ready: {result['passes_gate']}")
```

### Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Features:

- 🔬 **Run Evaluation**: Configure and execute evaluations
- 📊 **Dashboard**: Visualize results with radar charts
- 📈 **History**: Browse evaluation history with filters
- ⚖️ **Compare**: Side-by-side agent comparison

## AWS Deployment

### Prerequisites

- AWS CLI configured with credentials
- AWS CDK installed (`npm install -g aws-cdk`)
- Bedrock access enabled in your AWS account

### CDK Deployment

```bash
cd infrastructure
pip install -r requirements.txt
cdk bootstrap
cdk deploy
```

Deploys:

- Lambda function with Bedrock permissions
- API Gateway REST API (`/evaluate`, `/health`, `/results`)
- S3 bucket for audit logs (with Glacier lifecycle)
- Secrets Manager for API credentials
- CloudWatch dashboard and alarms

### Docker

```bash
docker build -t judge-agent .
docker run -p 8080:8080 judge-agent
```

## CI/CD

GitHub Actions workflows included:

- `.github/workflows/ci.yml` - Testing, linting, security scan
- `.github/workflows/deploy.yml` - CDK deployment to AWS

> **Note**: Configure GitHub secrets for CI/CD to work in your fork.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `CLAUDE_MODEL_ID` | `anthropic.claude-sonnet-4-20250514-v1:0` | Bedrock model |
| `SECURITY_PASS_THRESHOLD` | `0.8` | Security dimension threshold |
| `OVERALL_PASS_THRESHOLD` | `0.7` | General pass threshold |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | - | LangSmith API key |

## Testing

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/judge_agent
```

## Requirements

- Python 3.10+
- AWS account with Bedrock access
- See `requirements.txt` for full dependency list

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to extend this project.

## License

MIT
