"""
Judge Agent v3 - Enterprise AI Agent Evaluation Framework

A comprehensive evaluation system for AI agents built on LangChain/LangGraph
for AWS Bedrock. Provides 8-dimensional assessment of agent performance,
safety, and reliability.

Modules:
    langchain: Core LangChain/LangGraph evaluation implementation
    supervisor: Parallel supervisor architecture for concurrent evaluation
    enhanced_dimensions: Specialized evaluators (Performance, UX, Bias, Harm, Guardrails)
    utils: Utility functions (JSON parsing, retry logic, PII sanitization, audit logging)
"""

from judge_agent.langchain import (
    JudgeAgent,
    JudgeAgentState,
    AgentMetadata,
    EvaluationReport,
    EvaluationDimension,
    TestCase,
    TestResult,
    DimensionScore,
)

from judge_agent.supervisor import (
    JudgeAgentSupervisor,
    SupervisorState,
    AgentConfig,
    DimensionResult,
    EvaluationDimension as SupervisorDimension,
    MockAgent,
)

__version__ = "3.1.0"
__author__ = "JudgeAgent Team"

__all__ = [
    # Core classes
    "JudgeAgent",
    "JudgeAgentSupervisor",
    
    # State models
    "JudgeAgentState",
    "SupervisorState",
    
    # Configuration
    "AgentMetadata",
    "AgentConfig",
    
    # Results
    "EvaluationReport",
    "DimensionResult",
    "DimensionScore",
    "TestCase",
    "TestResult",
    
    # Enums
    "EvaluationDimension",
    "SupervisorDimension",
    
    # Testing
    "MockAgent",
    
    # Version
    "__version__",
]
