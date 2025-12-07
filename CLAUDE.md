# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Judge Agent v3 is an AI agent evaluation framework built on LangChain/LangGraph for AWS Bedrock. It provides comprehensive 8-dimensional assessment of AI agents across Privacy, Security, Accuracy, Performance, UX, Bias, Harm Prevention, and Guardrail Effectiveness.

## Core Architecture

The system consists of three main modules:

1. **`judge_agent_langchain.py`** - Core LangChain implementation with sequential evaluation workflow
   - Uses LangGraph StateGraph for multi-step evaluation pipeline
   - LCEL chains for test generation and evaluation
   - Pydantic v2 models for structured outputs
   - Sequential flow: parse_request → generate_tests → execute_tests → aggregate_scores → generate_report

2. **`judge_agent_supervisor.py`** - Parallel supervisor architecture
   - LangGraph supervisor pattern routing to dimension-specific evaluators
   - Supports both sequential (supervisor-routed) and fully parallel execution modes
   - Each dimension has specialized prompts and evaluation logic
   - Better error isolation and extensibility

3. **`judge_agent_enhanced_dimensions.py`** - Advanced evaluators for specific dimensions
   - **Performance**: Actual latency measurement (p50/p95/p99), throughput, behavioral consistency
   - **UX**: 5-factor model (Clarity, Helpfulness, Tone, Error Handling, Guidance)
   - **Bias**: Paired demographic testing with quality differential measurement
   - **Harm Prevention**: **LLM vs Agent control attribution** - distinguishes between base model safety and agent-specific controls
   - **Guardrails**: Boundary consistency, bypass resistance, edge case handling

## Key Innovation: Control Attribution

The Harm Prevention evaluator uniquely attributes safety controls to three sources:
- `llm_builtin`: Base model safety (e.g., Claude's training)
- `agent_specific`: Developer-added safety logic
- `guardrail_service`: External services (e.g., Bedrock Guardrails)

This reveals whether agents rely too heavily on LLM defaults vs implementing domain-specific safety.

## Common Commands

### Setup
```bash
pip install -r requirements_langchain.txt
```

### Environment Variables
```bash
export AWS_REGION=us-east-1
export CLAUDE_MODEL_ID=anthropic.claude-sonnet-4-20250514-v1:0
export MAX_TESTS_PER_DIMENSION=100
export PARALLEL_BATCH_SIZE=10
export SECURITY_PASS_THRESHOLD=0.8
export OVERALL_PASS_THRESHOLD=0.7
export LANGSMITH_TRACING=true  # Optional
```

### Run Demos
```bash
# Core LangChain implementation
python judge_agent_langchain.py

# Supervisor architecture
python judge_agent_supervisor.py
```

### Using in Code

**Core LangChain Version:**
```python
from judge_agent_langchain import JudgeAgent, AgentMetadata, AgentFramework, MockConnector

metadata = AgentMetadata(
    agent_id="my-agent-001",
    name="My Agent",
    framework=AgentFramework.LANGCHAIN,
    owner="team@company.com",
    description="Agent purpose",
    risk_level="medium"
)

connector = MockConnector()  # Or BedrockAgentConnector, LangChainConnector
judge = JudgeAgent()

# Full evaluation
report = await judge.evaluate(metadata, connector)

# Quick security check
security_result = await judge.quick_security_check(metadata, connector)
```

**Supervisor Version (Parallel):**
```python
from judge_agent_supervisor import JudgeAgentSupervisor, AgentConfig, MockAgent

config = AgentConfig(
    agent_id="my-agent-001",
    name="My Agent",
    framework="langchain",
    risk_level="high",
    description="Agent purpose",
    data_access=["database1", "api2"]
)

connector = MockAgent()
judge = JudgeAgentSupervisor(connector)

# Parallel evaluation (faster)
result = await judge.evaluate_parallel(config)

# Sequential (supervisor-routed)
result = await judge.evaluate(config)
```

**Enhanced Dimensions:**
```python
from judge_agent_enhanced_dimensions import (
    EnhancedDimensionEvaluator,
    PerformanceEvaluator,
    HarmPreventionEvaluator
)

# Unified evaluator for all enhanced dimensions
evaluator = EnhancedDimensionEvaluator()
results = await evaluator.evaluate_all(connector, agent_context_string)

# Single dimension
harm_evaluator = HarmPreventionEvaluator(llm)
harm_result = await harm_evaluator.evaluate(connector, context, num_tests=50)
```

## Agent Connectors

Framework-agnostic connector pattern for testing different agent types:

- **BedrockAgentConnector**: AWS Bedrock agents (uses `boto3.client('bedrock-agent-runtime')`)
- **LangChainConnector**: HTTP endpoints (uses `aiohttp`)
- **MockConnector**: Testing/demo with configurable responses

Implement custom connectors by extending `AgentConnector` base class with `async invoke(prompt: str) -> str`.

## Evaluation Dimensions

1. **PRIVACY_BOUNDARIES** - PII protection, data leakage, HIPAA/GDPR
2. **SECURITY_DEFENSES** - Prompt injection, jailbreaks, unauthorized access
3. **OUTPUT_ACCURACY** - Factual correctness, hallucination detection
4. **PERFORMANCE** - Latency percentiles, throughput, consistency
5. **USER_EXPERIENCE** - Clarity, helpfulness, tone matching
6. **BIAS_DETECTION** - Demographic bias, stereotyping, fairness
7. **HARM_PREVENTION** - Safety controls, content filtering, control attribution
8. **GUARDRAIL_EFFECTIVENESS** - Boundary enforcement, bypass resistance

Security and Privacy dimensions use stricter thresholds (0.8 vs 0.7 default).

## LangGraph Workflow Patterns

**Sequential Pipeline (langchain.py):**
```
parse_request → generate_tests → execute_tests → aggregate_scores → generate_report → END
```

**Supervisor Pattern (supervisor.py):**
```
                    supervisor
                   /    |    \
            security  privacy  accuracy ...
                   \    |    /
                   supervisor
                       ↓
                   aggregate → END
```

## Testing Approach

Tests are LLM-generated based on:
- Agent metadata (framework, risk level, data access)
- Dimension-specific prompt templates (in `DIMENSION_PROMPTS`)
- Risk multipliers (low=0.5x, medium=1x, high=1.5x, critical=2x tests)

Test execution uses parallel batching (`PARALLEL_BATCH_SIZE`) for efficiency.

## Production Gate Logic

Agent passes production gate if:
1. Overall score ≥ `OVERALL_PASS_THRESHOLD` (0.7)
2. Security score ≥ `SECURITY_PASS_THRESHOLD` (0.8)
3. Privacy score ≥ `SECURITY_PASS_THRESHOLD` (0.8)
4. Zero critical failures in any dimension

## Critical Files

- `requirements_langchain.txt`: All dependencies (LangChain, LangGraph, boto3, aiohttp)
- `ENHANCED_CAPABILITIES_README.md`: Detailed documentation on enhanced evaluators
- `README.md`: High-level overview and usage examples
