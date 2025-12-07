"""Test async retry decorator"""

import asyncio
import time
from utils.retry import retry_async, is_error_result, extract_error_message


# Test 1: Function that succeeds on first attempt
@retry_async(max_attempts=3, base_delay=0.1)
async def succeeds_immediately():
    """Always succeeds"""
    return {"status": "success", "data": "test"}


# Test 2: Function that fails twice then succeeds
attempt_count = 0

@retry_async(max_attempts=3, base_delay=0.1)
async def succeeds_on_third():
    """Fails twice, succeeds on third attempt"""
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError(f"Simulated failure {attempt_count}")
    return {"status": "success", "attempt": attempt_count}


# Test 3: Function that always fails
@retry_async(max_attempts=3, base_delay=0.1)
async def always_fails():
    """Always fails"""
    raise ValueError("This always fails")


# Test 4: Test exponential backoff timing
@retry_async(max_attempts=3, base_delay=0.5, exponential_backoff=True)
async def test_backoff():
    """Always fails to test backoff timing"""
    raise RuntimeError("Testing backoff")


# Test 5: Test without exponential backoff
@retry_async(max_attempts=3, base_delay=0.5, exponential_backoff=False)
async def test_no_backoff():
    """Always fails to test constant delay"""
    raise RuntimeError("Testing constant delay")


async def run_tests():
    """Run all retry decorator tests"""

    print("Test 1: Immediate success")
    result = await succeeds_immediately()
    assert result["status"] == "success", "Should succeed immediately"
    assert not is_error_result(result), "Should not be error result"
    print("[PASS] Immediate success works")

    print("\nTest 2: Succeeds on third attempt")
    global attempt_count
    attempt_count = 0
    result = await succeeds_on_third()
    assert result["status"] == "success", "Should succeed on third attempt"
    assert result["attempt"] == 3, f"Should be attempt 3, got {result['attempt']}"
    assert not is_error_result(result), "Should not be error result"
    print("[PASS] Retry until success works")

    print("\nTest 3: All attempts fail")
    result = await always_fails()
    assert is_error_result(result), "Should return error dict"
    assert result["attempts"] == 3, f"Should have 3 attempts, got {result['attempts']}"
    assert "ValueError" in result["exception_type"], "Should capture exception type"
    assert "always fails" in result["last_exception"], "Should capture exception message"
    print(f"[PASS] Error dict returned: {extract_error_message(result)}")

    print("\nTest 4: Exponential backoff timing")
    start_time = time.time()
    result = await test_backoff()
    elapsed = time.time() - start_time
    # Expected: 0.5s + 1.0s + 2.0s = 3.5s total delay
    # (no delay after first attempt, 0.5s after first fail, 1.0s after second fail)
    assert is_error_result(result), "Should return error dict"
    assert 0.5 <= elapsed <= 2.5, f"Exponential backoff timing off: {elapsed:.1f}s (expected ~1.5s)"
    print(f"[PASS] Exponential backoff timing correct: {elapsed:.1f}s")

    print("\nTest 5: Constant delay timing")
    start_time = time.time()
    result = await test_no_backoff()
    elapsed = time.time() - start_time
    # Expected: 0.5s + 0.5s = 1.0s total delay
    assert is_error_result(result), "Should return error dict"
    assert 0.8 <= elapsed <= 1.5, f"Constant delay timing off: {elapsed:.1f}s (expected ~1.0s)"
    print(f"[PASS] Constant delay timing correct: {elapsed:.1f}s")

    print("\nTest 6: Helper functions")
    error_dict = {
        'error': 'Test error',
        'attempts': 3,
        'last_exception': 'Connection timeout'
    }
    assert is_error_result(error_dict), "Should identify error dict"
    message = extract_error_message(error_dict)
    assert "3 attempts" in message, "Should include attempt count"
    assert "Connection timeout" in message, "Should include exception"
    print(f"[PASS] Helper functions work: '{message}'")

    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(run_tests())
