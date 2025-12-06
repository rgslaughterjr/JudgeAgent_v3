# Judge Agent v3

AI Agent evaluation system built with LangChain/LangGraph for AWS Bedrock, providing comprehensive multi-dimensional assessment of agent performance, safety, and reliability.

## Overview

Judge Agent v3 is an advanced AI agent evaluation framework that systematically assesses AI agents across 8 critical dimensions. Built on LangChain and LangGraph, it provides robust, scalable evaluation capabilities for AWS Bedrock-based agents.

## Key Features

- **8-Dimensional Evaluation Framework**: Comprehensive assessment across Privacy, Security, Accuracy, Performance, UX, Bias, Harm Prevention, and Guardrail Effectiveness
- **LLM vs Agent Control Attribution**: Innovative harm prevention analysis distinguishing between LLM and agent-level control
- **Parallel Supervisor Architecture**: Efficient concurrent evaluation using LangGraph's supervisor pattern
- **AWS Bedrock Integration**: Native support for AWS Bedrock models
- **Enhanced Evaluators**: Specialized evaluators for each dimension with detailed scoring and recommendations

## Architecture

### Core Components

- **`judge_agent_langchain.py`**: Core LangChain/LangGraph implementation with main evaluation logic
- **`judge_agent_supervisor.py`**: Parallel supervisor architecture for concurrent evaluation workflows
- **`judge_agent_enhanced_dimensions.py`**: Enhanced evaluators for Performance, UX, Bias, Harm Prevention, and Guardrails

### Evaluation Dimensions

1. **Privacy**: Data handling, PII protection, consent management
2. **Security**: Vulnerability assessment, authentication, encryption
3. **Accuracy**: Response correctness, hallucination detection, factual grounding
4. **Performance**: Response time, throughput, resource utilization
5. **User Experience**: Interface quality, error handling, accessibility
6. **Bias**: Fairness, representation, stereotype detection
7. **Harm Prevention**: Safety controls, content filtering, risk mitigation
8. **Guardrail Effectiveness**: Policy enforcement, boundary detection, override prevention

## Installation

```bash
pip install -r requirements_langchain.txt
```

## Usage

```python
from judge_agent_langchain import JudgeAgent

# Initialize the judge agent
judge = JudgeAgent()

# Evaluate an AI agent
results = judge.evaluate(
    agent_output="...",
    context="...",
    dimensions=["privacy", "security", "accuracy"]
)

# View results
print(results)
```

## Key Innovation: Control Attribution

The Harm Prevention evaluator uniquely distinguishes between:
- **LLM-level controls**: Base model safety features and training
- **Agent-level controls**: Application-specific guardrails and policies

This attribution enables precise identification of safety responsibility and improvement opportunities.

## Documentation

See [ENHANCED_CAPABILITIES_README.md](ENHANCED_CAPABILITIES_README.md) for detailed documentation on enhanced capabilities and evaluation criteria.

## Requirements

- Python 3.8+
- AWS Bedrock access
- LangChain
- LangGraph

## License

MIT

## Contributing

Contributions welcome. Please open an issue or PR.
