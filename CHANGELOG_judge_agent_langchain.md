# Changelog: judge_agent_langchain.py Updates

## Summary of Changes

Updated `judge_agent_langchain.py` to integrate utility functions from `utils/` package for improved reliability and error handling.

## Changes Made

### 1. Added Utility Imports (Line 37-38)

```python
# Import utility functions
from utils import extract_json, retry_async
```

### 2. Enhanced Test Generation with JSON Extraction Fallback (Lines 475-526)

**Location:** `generate_tests_node()` → `generate_for_dimension()`

**Changes:**
- Added fallback JSON extraction when PydanticOutputParser fails
- Uses `extract_json()` utility for robust JSON parsing
- Three-tier fallback strategy:
  1. PydanticOutputParser (primary)
  2. Manual JSON extraction with `extract_json()` (fallback)
  3. Default mock test (last resort)

**Benefits:**
- Handles malformed LLM responses gracefully
- Extracts JSON from code blocks (```json) automatically
- Ensures test generation never completely fails

### 3. Added Retry Decorator to Agent Invocations (Lines 503-527)

**Location:** `execute_tests_node()` → `execute_single_test()`

**Changes:**
- Wrapped `connector.invoke()` with `@retry_async` decorator
- 3 retry attempts with exponential backoff (1s, 2s, 4s)
- Error dict detection using `is_error_result()` and `extract_error_message()`
- Graceful handling of failed invocations

**Code:**
```python
@retry_async(max_attempts=3, base_delay=1.0, exponential_backoff=True)
async def invoke_agent_with_retry(prompt: str) -> str:
    """Invoke agent with automatic retry on failure"""
    return await connector.invoke(prompt)
```

### 4. Added Retry to Quick Security Check (Lines 861-875)

**Location:** `JudgeAgent.quick_security_check()`

**Changes:**
- Same retry pattern as above
- Skips failed tests instead of crashing
- Continues evaluation even if some agent calls fail

## Technical Details

### Retry Configuration
- **Max attempts:** 3
- **Base delay:** 1.0 seconds
- **Backoff:** Exponential (1s → 2s → 4s)
- **Error handling:** Returns error dict instead of raising exception

### JSON Extraction Fallback Strategy

1. **Primary:** PydanticOutputParser attempts structured parsing
2. **Fallback:** If parsing fails, retry with `extract_json()`:
   - Extracts from ```json``` code blocks
   - Extracts from ``` generic blocks
   - Finds JSON objects/arrays in plain text
   - Handles nested structures
3. **Last Resort:** Returns minimal default test case

### Error Detection Pattern

```python
from utils import is_error_result, extract_error_message

response = await invoke_agent_with_retry(prompt)
if is_error_result(response):
    error_msg = extract_error_message(response)
    # Handle error gracefully
```

## Testing Recommendations

1. **Test retry behavior:**
   - Simulate transient failures
   - Verify 3 retry attempts
   - Check exponential backoff timing

2. **Test JSON extraction fallback:**
   - Provide malformed JSON in LLM responses
   - Test ```json code blocks
   - Test embedded JSON in text

3. **Test error handling:**
   - Verify error dicts are detected
   - Check that evaluation continues after failures
   - Ensure appropriate logging

## Backward Compatibility

✅ **Fully backward compatible**
- No breaking changes to API
- Existing code continues to work
- New features activate automatically on errors

## Performance Impact

- **Minimal overhead** when operations succeed
- **Automatic recovery** reduces manual intervention
- **Better completion rates** for evaluations
- **Clearer error messages** for debugging

## Dependencies

Requires `utils/` package with:
- `extract_json()` - Robust JSON extraction
- `retry_async()` - Async retry decorator
- `is_error_result()` - Error dict detection
- `extract_error_message()` - Error message formatting
