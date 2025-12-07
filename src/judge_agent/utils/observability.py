"""
Observability configuration for Judge Agent v3

Configures:
- LangSmith tracing for LLM chain observability
- Structured logging
- Metrics collection
"""

import os
import logging
from typing import Optional

# ============================================================================
# LANGSMITH CONFIGURATION
# ============================================================================

def configure_langsmith(
    project_name: str = "judge-agent",
    tracing_enabled: Optional[bool] = None
) -> bool:
    """
    Configure LangSmith tracing for observability.
    
    Environment variables:
        LANGCHAIN_TRACING_V2: Enable tracing (true/false)
        LANGCHAIN_API_KEY: LangSmith API key
        LANGCHAIN_PROJECT: Project name
        LANGCHAIN_ENDPOINT: API endpoint (optional)
    
    Args:
        project_name: LangSmith project name
        tracing_enabled: Override for enabling tracing
    
    Returns:
        True if tracing is enabled and configured
    """
    # Check if tracing should be enabled
    if tracing_enabled is None:
        tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    if not tracing_enabled:
        logging.info("LangSmith tracing disabled")
        return False
    
    # Check for API key
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        logging.warning("LANGCHAIN_API_KEY not set - LangSmith tracing disabled")
        return False
    
    # Set environment variables for LangChain
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    # Optional: custom endpoint
    endpoint = os.environ.get("LANGCHAIN_ENDPOINT")
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    
    logging.info(f"LangSmith tracing enabled for project: {project_name}")
    return True


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

def configure_logging(
    level: str = "INFO",
    json_format: bool = False
) -> logging.Logger:
    """
    Configure structured logging for production.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format for log messages
    
    Returns:
        Configured logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if json_format:
        import json
        from datetime import datetime
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_obj)
        
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]
    
    # Configure Judge Agent logger specifically
    judge_logger = logging.getLogger("judge_agent")
    judge_logger.setLevel(log_level)
    
    return judge_logger


# ============================================================================
# METRICS HELPERS
# ============================================================================

class MetricsCollector:
    """
    Simple metrics collector for evaluation tracking.
    
    In production, integrate with CloudWatch, Prometheus, or Datadog.
    """
    
    def __init__(self):
        self.metrics = {
            "evaluations_total": 0,
            "evaluations_passed": 0,
            "evaluations_failed": 0,
            "tests_executed": 0,
            "average_score": 0.0,
            "dimension_scores": {},
            "latency_ms": []
        }
    
    def record_evaluation(
        self,
        passed: bool,
        score: float,
        tests_run: int,
        dimension_scores: dict,
        latency_ms: float
    ):
        """Record an evaluation result."""
        self.metrics["evaluations_total"] += 1
        if passed:
            self.metrics["evaluations_passed"] += 1
        else:
            self.metrics["evaluations_failed"] += 1
        
        self.metrics["tests_executed"] += tests_run
        self.metrics["latency_ms"].append(latency_ms)
        
        # Update running average
        total = self.metrics["evaluations_total"]
        prev_avg = self.metrics["average_score"]
        self.metrics["average_score"] = prev_avg + (score - prev_avg) / total
        
        # Track dimension scores
        for dim, dim_score in dimension_scores.items():
            if dim not in self.metrics["dimension_scores"]:
                self.metrics["dimension_scores"][dim] = []
            self.metrics["dimension_scores"][dim].append(dim_score)
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        latencies = self.metrics["latency_ms"]
        return {
            "evaluations": {
                "total": self.metrics["evaluations_total"],
                "passed": self.metrics["evaluations_passed"],
                "failed": self.metrics["evaluations_failed"],
                "pass_rate": (
                    self.metrics["evaluations_passed"] / self.metrics["evaluations_total"]
                    if self.metrics["evaluations_total"] > 0 else 0
                )
            },
            "tests_executed": self.metrics["tests_executed"],
            "average_score": self.metrics["average_score"],
            "latency_ms": {
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "avg": sum(latencies) / len(latencies) if latencies else 0
            }
        }


# Global metrics collector
_metrics = MetricsCollector()

def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_observability(
    langsmith_project: str = "judge-agent",
    log_level: str = "INFO",
    json_logs: bool = False
) -> dict:
    """
    Initialize all observability components.
    
    Args:
        langsmith_project: LangSmith project name
        log_level: Logging level
        json_logs: Use JSON formatted logs
    
    Returns:
        Configuration status dict
    """
    logger = configure_logging(level=log_level, json_format=json_logs)
    langsmith_enabled = configure_langsmith(project_name=langsmith_project)
    
    return {
        "logging_configured": True,
        "log_level": log_level,
        "langsmith_enabled": langsmith_enabled,
        "langsmith_project": langsmith_project if langsmith_enabled else None
    }
