"""Quick tests for extract_json utility"""

from utils.json_parser import extract_json

def test_extract_json():
    """Test various JSON extraction scenarios"""

    # Test 1: Direct JSON
    result = extract_json('{"key": "value"}')
    assert result == {"key": "value"}, "Direct JSON parsing failed"
    print("[PASS] Test 1: Direct JSON")

    # Test 2: JSON in ```json code block
    result = extract_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}, "```json block parsing failed"
    print("[PASS] Test 2: ```json code block")

    # Test 3: JSON in generic ``` code block
    result = extract_json('```\n[1, 2, 3]\n```')
    assert result == [1, 2, 3], "``` block parsing failed"
    print("[PASS] Test 3: Generic ``` code block")

    # Test 4: JSON embedded in text
    result = extract_json('Here is the data: {"result": [1, 2, 3]} and more text')
    assert result == {"result": [1, 2, 3]}, "Embedded JSON parsing failed"
    print("[PASS] Test 4: Embedded JSON in text")

    # Test 5: Array embedded in text
    result = extract_json('The results are: [{"a": 1}, {"b": 2}] as shown')
    assert result == [{"a": 1}, {"b": 2}], "Embedded array parsing failed"
    print("[PASS] Test 5: Embedded array in text")

    # Test 6: Nested objects
    result = extract_json('{"outer": {"inner": {"deep": "value"}}}')
    assert result == {"outer": {"inner": {"deep": "value"}}}, "Nested objects failed"
    print("[PASS] Test 6: Nested objects")

    # Test 7: Complex JSON with strings containing brackets
    result = extract_json('{"text": "This has { and } brackets", "data": [1, 2]}')
    assert result["text"] == "This has { and } brackets", "String with brackets failed"
    print("[PASS] Test 7: JSON with brackets in strings")

    # Test 8: Error case - no JSON
    try:
        extract_json("This is just plain text with no JSON")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No JSON object or array found" in str(e), "Wrong error message"
        print("[PASS] Test 8: Proper error on no JSON")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_extract_json()
