# Utils Package

Utility functions for Judge Agent v3.

## Modules

### `json_parser.py`

JSON extraction utilities for parsing LLM responses.

#### `extract_json(content: str) -> dict | list`

Robustly extracts and parses JSON from LLM response content.

**Extraction strategies (in order):**
1. Direct JSON parsing
2. Extract from ```json code blocks
3. Extract from ``` code blocks
4. Find first valid JSON object/array using bracket matching

**Usage:**
```python
from utils import extract_json

# Direct JSON
data = extract_json('{"key": "value"}')

# From code block
data = extract_json('```json\n{"key": "value"}\n```')

# Embedded in text
data = extract_json('Here is the data: {"result": [1, 2, 3]} more text')
```

**Error handling:**
```python
try:
    data = extract_json(llm_response)
except ValueError as e:
    # Error includes content preview for debugging
    logger.error(f"JSON extraction failed: {e}")
```

---

### `retry.py`

Async retry decorator with exponential backoff and error dict return pattern.

#### `@retry_async(max_attempts=3, base_delay=1.0, exponential_backoff=True)`

Retries async functions with configurable backoff. Returns error dict on failure instead of raising exception.

**Features:**
- Retries up to `max_attempts` times (default: 3)
- Exponential backoff: 1s, 2s, 4s (with base_delay=1.0)
- Logs each attempt with detailed error info
- Returns error dict on final failure: `{'error': str, 'attempts': int, 'last_exception': str, 'exception_type': str}`

**Usage:**
```python
from utils import retry_async, is_error_result, extract_error_message

# Decorate async function
@retry_async(max_attempts=3, base_delay=1.0)
async def call_agent(prompt: str):
    response = await agent_connector.invoke(prompt)
    return response

# Use with error checking
result = await call_agent("test prompt")

if is_error_result(result):
    logger.error(extract_error_message(result))
    # Handle error case
    return None
else:
    # Handle success case
    process_response(result)
```

**Custom configuration:**
```python
# More retries, faster backoff
@retry_async(max_attempts=5, base_delay=0.5, exponential_backoff=True)
async def critical_operation():
    ...

# Constant delay (no exponential backoff)
@retry_async(max_attempts=3, base_delay=2.0, exponential_backoff=False)
async def steady_retry():
    ...

# Use convenience decorator with defaults
from utils import retry_with_backoff

@retry_with_backoff  # 3 attempts, 1s/2s/4s backoff
async def quick_setup():
    ...
```

**Helper functions:**
```python
# Check if result is an error
if is_error_result(result):
    print("Operation failed")

# Extract formatted error message
message = extract_error_message(result)
# "Failed after 3 attempts. Last error: Connection timeout"
```

---

## Legacy Retry (json_parser.py)

The `json_parser.py` module also contains retry decorators that **raise exceptions** on failure (different behavior than `retry.py`):

- `retry_agent_call` - 3 attempts, 1s/2s/4s backoff, catches ConnectionError/TimeoutError
- `retry_llm_call` - 5 attempts, 0.5s/0.75s/1.125s backoff, catches all exceptions

**Usage:**
```python
from utils import retry_agent_call

@retry_agent_call
async def invoke_agent(prompt):
    return await connector.invoke(prompt)

# Raises exception on final failure
try:
    result = await invoke_agent("test")
except Exception as e:
    logger.error(f"Failed after retries: {e}")
```

---

### `pii_sanitizer.py`

PII data generation and sanitization utilities for testing and security.

#### `SyntheticDataGenerator`

Generate obviously fake synthetic PII data for testing purposes.

**Features:**
- All data uses reserved/invalid ranges (SSN starts with 999-, phone uses 555, etc.)
- Clearly synthetic to avoid matching real PII
- Reproducible with optional seed parameter

**Methods:**
```python
from utils import SyntheticDataGenerator

gen = SyntheticDataGenerator(seed=42)  # Optional seed for reproducibility

# Generate fake PII
ssn = gen.ssn()                    # '999-81-1824'
card = gen.credit_card('visa')     # '4000-1234-5678-9010'
phone = gen.phone('formatted')     # '(555) 123-4567'
email = gen.email()                # 'test.user123@example.com'
patient_id = gen.patient_id()      # 'TEST-P-123456789'
mrn = gen.medical_record_number()  # 'DEMO-MRN-123456'
dob = gen.date_of_birth()          # '01/01/1900'
```

**Card types:** `visa`, `mastercard`, `amex`, `discover`
**Phone formats:** `formatted`, `dashes`, `plain`

#### `PIISanitizer`

Detect and redact PII from text strings.

**Supported PII types:**
- Social Security Numbers (SSN)
- Credit card numbers
- Phone numbers
- Email addresses
- Patient IDs
- Medical record numbers
- Dates of birth (with context)

**Basic usage:**
```python
from utils import PIISanitizer, sanitize_text

# Quick sanitization
sanitized = sanitize_text("My SSN is 123-45-6789")
# "My SSN is [REDACTED-SSN]"

# Full control
sanitizer = PIISanitizer()
text = "Contact john@example.com or call 555-123-4567"
sanitized = sanitizer.sanitize(text)
# "Contact [REDACTED-EMAIL] or call [REDACTED-PHONE]"
```

**Detection without sanitization:**
```python
sanitizer = PIISanitizer()

# Detect PII occurrences
detections = sanitizer.detect_pii(text)
# [('email', 'john@example.com', 8, 24), ('phone', '555-123-4567', 33, 45)]

# Check if text contains PII
if sanitizer.has_pii(text):
    print("Warning: PII detected!")

# Count PII by type
counts = sanitizer.get_pii_count(text)
# {'email': 1, 'phone': 1}
```

**Format preservation:**
```python
# Preserve original length with X's
sanitizer = PIISanitizer(preserve_format=True)
sanitized = sanitizer.sanitize("SSN: 123-45-6789")
# "SSN: XXXXXXXXXXX"
```

**Generate test dataset:**
```python
from utils import generate_test_data

test_data = generate_test_data()
# {
#   'ssn': '999-12-3456',
#   'credit_card_visa': '4000-1234-5678-9010',
#   'phone_formatted': '(555) 123-4567',
#   'email': 'test.user456@example.com',
#   ...
# }
```

---

### `audit_log.py`

Audit logging for Judge Agent evaluations with JSONL format and date-partitioned storage.

#### `AuditLogger`

Immutable audit trail for agent evaluations stored in JSONL format (one JSON object per line).

**Features:**
- JSONL format for append-only, immutable logging
- Date-partitioned storage: `YYYY/MM/DD/evaluations.jsonl`
- Dual storage: local files and S3
- Query history with filtering
- Agent statistics and analytics

**Basic usage:**
```python
from utils import AuditLogger, create_audit_logger

# Create logger (local + S3)
logger = AuditLogger(
    local_log_dir="./audit-logs",
    s3_bucket="my-audit-bucket",
    s3_prefix="judge-agent/audit-logs"
)

# Or use convenience function
logger = create_audit_logger(s3_bucket="my-bucket")

# Log evaluation
await logger.log_evaluation(
    evaluation_id="eval-123",
    agent_id="agent-001",
    agent_name="Customer Service Bot",
    evaluator_user="engineer@company.com",
    framework="langchain",
    risk_level="high",
    overall_score=0.85,
    passed=True,
    dimension_scores={
        "security": {"score": 0.9, "passed": True, "test_count": 10},
        "privacy": {"score": 0.8, "passed": True, "test_count": 8}
    },
    critical_failures=0,
    deployment_stage="production",
    metadata={"version": "2.1.0"}
)
```

**Query history:**
```python
from datetime import datetime, timedelta

# Query all evaluations for an agent
entries = await logger.query_history(agent_id="agent-001")

# Query with date range
start = datetime.now() - timedelta(days=30)
end = datetime.now()
entries = await logger.query_history(
    agent_id="agent-001",
    start_date=start,
    end_date=end
)

# Query passed evaluations only
passed = await logger.query_history(
    agent_id="agent-001",
    passed_only=True
)

# Query by minimum score
high_performers = await logger.query_history(
    agent_id="agent-001",
    min_score=0.9
)
```

**Agent statistics:**
```python
# Get 30-day statistics summary
stats = await logger.get_agent_statistics("agent-001", days=30)

# Returns:
# {
#   'agent_id': 'agent-001',
#   'period_days': 30,
#   'total_evaluations': 45,
#   'passed_evaluations': 42,
#   'pass_rate': 0.933,
#   'average_score': 0.87,
#   'min_score': 0.65,
#   'max_score': 0.98,
#   'latest_evaluation': '2025-12-06T12:00:00Z',
#   'latest_score': 0.91,
#   'latest_passed': True
# }
```

**Storage configuration:**
```python
# Local only (no S3)
logger = create_audit_logger(local_only=True)

# S3 only (no local files)
logger = AuditLogger(
    s3_bucket="my-bucket",
    enable_local=False,
    enable_s3=True
)

# Custom S3 prefix and region
logger = AuditLogger(
    s3_bucket="my-bucket",
    s3_prefix="prod/audit-logs",
    aws_region="us-west-2"
)
```

**Date partitioning:**
Logs are automatically partitioned by date:
- Local: `./audit-logs/2025/12/06/evaluations.jsonl`
- S3: `s3://bucket/audit-logs/2025/12/06/evaluations.jsonl`

This enables:
- Efficient querying by date range
- Easy archival/deletion of old logs
- Cost optimization (S3 lifecycle policies)

**JSONL format:**
Each line is a complete JSON object:
```jsonl
{"timestamp":"2025-12-06T12:00:00Z","evaluation_id":"eval-1","agent_id":"agent-001",...}
{"timestamp":"2025-12-06T13:30:00Z","evaluation_id":"eval-2","agent_id":"agent-002",...}
```

Benefits:
- Append-only (immutable)
- Line-by-line parsing
- Compatible with log processing tools (Athena, BigQuery, etc.)

---

## Choosing Between Retry Implementations

**Use `utils.retry.retry_async`** when:
- You want to handle errors as return values (error dict pattern)
- You need custom error handling logic
- You want to continue execution after failure

**Use `utils.json_parser.retry_agent_call/retry_llm_call`** when:
- You want exceptions to propagate
- You're using try/except error handling
- You need pre-configured retry settings

---

## Installation

No additional dependencies required beyond the main `requirements_langchain.txt`.

## Testing

Run tests to verify utilities:
```bash
# Test JSON extraction
python test_json_parser.py

# Test retry decorator
python test_retry.py

# Test PII sanitizer
python test_pii_sanitizer.py

# Test audit logger
python test_audit_log.py
```
