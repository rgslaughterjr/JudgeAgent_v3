"""JSON parsing and retry utilities for Judge Agent"""

import asyncio
import json
import re
import logging
from typing import TypeVar, Callable, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def extract_json(content: str) -> dict | list:
    """
    Robustly extract and parse JSON from LLM response content.

    Extraction strategy (in order):
    1. Try direct JSON parsing of the entire content
    2. Extract from ```json code blocks using regex
    3. Extract from ``` code blocks using regex
    4. Find first valid JSON object/array in text using bracket matching

    Args:
        content: Raw LLM response content that may contain JSON

    Returns:
        Parsed JSON as dict or list

    Raises:
        ValueError: If no valid JSON found, with content preview for debugging
        json.JSONDecodeError: If extracted content is not valid JSON

    Examples:
        >>> extract_json('{"key": "value"}')
        {'key': 'value'}

        >>> extract_json('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}

        >>> extract_json('Here is the data: [1, 2, 3] and more text')
        [1, 2, 3]
    """
    if not content or not isinstance(content, str):
        raise ValueError("Content must be a non-empty string")

    original_content = content
    content = content.strip()

    # Strategy 1: Try direct JSON parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parsing failed, trying extraction methods")

    # Strategy 2: Extract from ```json code blocks using regex
    json_block_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_block_pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            extracted = match.group(1).strip()
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON from ```json block failed to parse: {e}")

    # Strategy 3: Extract from generic ``` code blocks
    code_block_pattern = r'```[a-zA-Z]*\s*\n(.*?)\n```'
    match = re.search(code_block_pattern, content, re.DOTALL)
    if match:
        try:
            extracted = match.group(1).strip()
            # Remove language identifier if still present
            if extracted.startswith(('json\n', 'JSON\n')):
                extracted = extracted.split('\n', 1)[1].strip()
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON from ``` block failed to parse: {e}")

    # Strategy 4: Find first valid JSON object or array using bracket matching
    # Try to find JSON object
    obj_start = content.find('{')
    arr_start = content.find('[')

    # Determine which comes first
    candidates = []

    if obj_start != -1:
        candidates.append((obj_start, '{', '}'))
    if arr_start != -1:
        candidates.append((arr_start, '[', ']'))

    if not candidates:
        # No JSON structure found
        preview = content[:200] + "..." if len(content) > 200 else content
        raise ValueError(
            f"No JSON object or array found in content. "
            f"Content preview: {preview}"
        )

    # Sort by start position to try the first one
    candidates.sort(key=lambda x: x[0])

    # Try each candidate
    for start_pos, open_char, close_char in candidates:
        try:
            extracted = _extract_balanced_json(content, start_pos, open_char, close_char)
            if extracted:
                return json.loads(extracted)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Candidate at position {start_pos} failed: {e}")
            continue

    # All strategies failed
    preview = content[:200] + "..." if len(content) > 200 else content
    raise ValueError(
        f"Failed to extract valid JSON from content using all strategies. "
        f"Content preview: {preview}"
    )


def _extract_balanced_json(
    content: str,
    start: int,
    open_char: str,
    close_char: str
) -> str:
    """
    Extract balanced brackets from content starting at position.

    Args:
        content: Full content string
        start: Starting position of opening bracket
        open_char: Opening bracket character ('{' or '[')
        close_char: Closing bracket character ('}' or ']')

    Returns:
        Extracted string with balanced brackets

    Raises:
        ValueError: If balanced brackets cannot be found
    """
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(content)):
        char = content[i]

        # Handle string escaping
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        # Handle strings (to ignore brackets inside strings)
        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        # Count bracket depth
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1

            if depth == 0:
                # Found matching closing bracket
                return content[start:i + 1]

    raise ValueError(
        f"No matching closing bracket '{close_char}' found for '{open_char}' at position {start}"
    )


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Async retry decorator with exponential backoff.

    Retries async function calls on specified exceptions with exponential backoff.
    Useful for handling transient failures in agent invocations or API calls.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each attempt (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: (Exception,))

    Returns:
        Decorated async function with retry logic

    Raises:
        Last exception encountered if all retry attempts fail

    Examples:
        >>> @retry_async(max_attempts=3, delay=1.0, backoff=2.0)
        ... async def call_agent(prompt):
        ...     return await agent.invoke(prompt)

        >>> @retry_async(max_attempts=5, exceptions=(TimeoutError, ConnectionError))
        ... async def fetch_data():
        ...     return await api.get('/data')
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


# Convenience decorators for common retry scenarios
retry_agent_call = retry_async(
    max_attempts=3,
    delay=1.0,
    backoff=2.0,
    exceptions=(ConnectionError, TimeoutError, Exception)
)

retry_llm_call = retry_async(
    max_attempts=5,
    delay=0.5,
    backoff=1.5,
    exceptions=(Exception,)
)
