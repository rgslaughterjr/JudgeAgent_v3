# Changelog

All notable changes to Judge Agent v3.

## [3.1.0] - 2024-12-06

### Added

**Phase 1: Code Quality**

- Applied `invoke_llm_with_retry()` to all LLM calls in supervisor and enhanced dimensions
- Added `sanitize_response()` for PII protection in stored responses
- Replaced manual JSON parsing with `extract_json()` utility throughout
- Added `error_tracking` dictionary to `SupervisorState` for debugging
- Created comprehensive unit tests (`test_judge_agent.py`)

**Phase 2: AWS Deployment**

- AWS CDK stack (`infrastructure/app.py`) with:
  - Lambda function with Bedrock permissions
  - API Gateway REST API
  - S3 bucket with Glacier lifecycle policies
  - Secrets Manager for API credentials
  - CloudWatch dashboard and alarms
- Lambda handler (`lambda/lambda_handler.py`) with REST endpoints
- Dockerfile for container deployment
- Deployment scripts (`scripts/deploy.sh`, `scripts/build-docker.sh`)

**Phase 3: CI/CD & Observability**

- GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- GitHub Actions CD workflow (`.github/workflows/deploy.yml`)
- Observability module (`utils/observability.py`) with:
  - LangSmith tracing configuration
  - Structured JSON logging
  - Metrics collection

**Phase 4: User Interface**

- Full Streamlit dashboard (`streamlit_app.py`) with:
  - Run Evaluation page
  - Dashboard with radar charts
  - History viewer with filtering
  - Agent comparison page

**Utilities**

- Added `sanitize_response` alias in utils `__init__.py`
- Exported observability functions from utils package

### Changed

- `evaluate()` and `evaluate_parallel()` now accept `evaluator_user` parameter
- Evaluation results now include `error_tracking` dictionary
- All evaluations now logged to audit trail automatically

### Fixed

- Retry logic now properly applied to all 6 remaining LLM calls in enhanced dimensions
- JSON parsing no longer uses manual `split()` logic

---

## [3.0.0] - 2024-12-05

### Added

**Core Architecture**

- LangChain/LangGraph-based evaluation framework
- Parallel supervisor architecture for concurrent evaluation
- Support for 8 evaluation dimensions

**Enhanced Evaluators** (`judge_agent_enhanced_dimensions.py`)

- `PerformanceEvaluator`: Latency measurement (p50/p95/p99), throughput, behavioral consistency
- `UXEvaluator`: 5-factor model (clarity, helpfulness, tone, error handling, guidance)
- `BiasEvaluator`: Paired demographic testing across 6 categories
- `HarmPreventionEvaluator`: LLM vs Agent control attribution
- `GuardrailEvaluator`: Boundary consistency, bypass resistance testing

**Utilities Package** (`utils/`)

- `extract_json()`: Robust JSON extraction from LLM responses
- `retry_async()`: Async retry decorator with exponential backoff
- `SyntheticDataGenerator`: Generate fake PII for safe testing
- `PIISanitizer`: Redact PII from responses
- `AuditLogger`: JSONL audit logging with S3 support

**Test Generation**

- Synthetic PII in harm prevention and guardrail tests
- Synthetic demographic names in bias tests
- Multi-strategy JSON extraction fallback

### Key Innovation

- **Control Attribution**: Harm Prevention evaluator distinguishes between:
  - `llm_builtin`: Base LLM safety
  - `agent_specific`: Developer-added safety logic
  - `guardrail_service`: External guardrail services

---

## [2.0.0] - Previous

- Initial LangChain implementation
- Basic 8-dimension evaluation framework
- AWS Bedrock integration
