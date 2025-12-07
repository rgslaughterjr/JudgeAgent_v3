# Changes Needed for judge_agent_enhanced_dimensions.py

## Summary
Update file to use utility functions for improved reliability and PII protection.

## Changes Applied

### 1. ✅ Added Imports (Lines 28-39)
- extract_json, retry_async, is_error_result, extract_error_message
- SyntheticDataGenerator, PIISanitizer, sanitize_text
- logging

### 2. ✅ Added Utility Instances (Lines 76-89)
- _synthetic_data_gen - for generating fake PII in tests
- _pii_sanitizer - for sanitizing responses
- sanitize_response() helper function
- invoke_llm_with_retry() - retry decorator wrapper

### 3. ✅ Updated PerformanceEvaluator.generate_tests() (Lines 511-535)
- Uses invoke_llm_with_retry()
- Uses extract_json()
- Error handling for failed LLM calls

### 4. ✅ Updated UXEvaluator.generate_tests() (Lines 675-697)
- Uses invoke_llm_with_retry()
- Uses extract_json()
- Error handling for failed LLM calls

## Changes Remaining

### 5. Update BiasEvaluator.generate_tests() (Line ~787)
**Current:**
```python
response = await self.llm.ainvoke([...])
# Manual JSON parsing with split()
return json.loads(content.strip())
```

**Should be:**
```python
response = await invoke_llm_with_retry(self.llm, [...])
if is_error_result(response):
    return []
content = response.content if hasattr(response, 'content') else str(response)
return extract_json(content)
```

### 6. Update HarmPreventionEvaluator.generate_tests() (Line ~919)
- Same pattern as #5

### 7. Update GuardrailEvaluator.generate_tests() (Line ~1082)
- Same pattern as #5

### 8. Add PII Sanitization to Stored Responses

**Locations needing sanitization:**
- Line ~824-825: `response_a/response_b` before comparison
- Line ~1011-1014: test_prompt and agent_response storage
- Line ~1183-1184: test_prompt storage
- All places where responses are stored in results

**Pattern:**
```python
# Before
agent_response=response[:200]

# After
agent_response=sanitize_response(response[:200])
```

### 9. Add Synthetic PII to Test Prompts

**For HarmPreventionEvaluator** - Add synthetic PII examples:
```python
# In generate_tests, enhance prompt with:
synthetic_examples = f"""
Use these FAKE examples when needed:
- SSN: {_synthetic_data_gen.ssn()}
- Credit Card: {_synthetic_data_gen.credit_card()}
- Phone: {_synthetic_data_gen.phone()}
- Email: {_synthetic_data_gen.email()}
- Patient ID: {_synthetic_data_gen.patient_id()}
"""
```

### 10. Update All LLM Invocations

**Find all instances of:**
```python
await self.llm.ainvoke([...])
```

**Replace with:**
```python
await invoke_llm_with_retry(self.llm, [...])
# Then check: if is_error_result(response): handle error
```

**Locations:**
- Line ~602: PerformanceEvaluator consistency evaluation
- Line ~620: PerformanceEvaluator recovery evaluation
- Line ~648: UXEvaluator evaluation
- Line ~696: UXEvaluator LLM eval
- Line ~757: BiasEvaluator comparison
- Line ~813: BiasEvaluator bias detection
- Line ~892: HarmPreventionEvaluator test generation
- Line ~949: HarmPreventionEvaluator control detection
- Line ~1055: GuardrailEvaluator test generation
- Line ~1119: GuardrailEvaluator effectiveness evaluation

## Testing Recommendations

1. Verify all LLM calls have retry logic
2. Confirm PII is sanitized in stored responses
3. Test synthetic PII generation in privacy tests
4. Validate extract_json handles malformed responses
5. Check error handling when LLM calls fail

## Priority

**High Priority:**
1. Update remaining generate_tests methods (#5-7)
2. Add PII sanitization (#8)
3. Update all LLM invocations (#10)

**Medium Priority:**
4. Add synthetic PII to test generation (#9)

**Low Priority:**
5. Refactor duplicate code patterns
