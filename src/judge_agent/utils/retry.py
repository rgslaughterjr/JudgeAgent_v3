"""Async retry utilities with error dict return pattern"""

import asyncio
import logging
from typing import TypeVar, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential_backoff: bool = True
):
    """
    Async retry decorator with exponential backoff and error dict return.

    On final failure, returns an error dict instead of raising exception:
    {'error': str, 'attempts': int, 'last_exception': str}

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
            - With exponential_backoff=True: 1s, 2s, 4s
            - With exponential_backoff=False: 1s, 1s, 1s
        exponential_backoff: Use exponential backoff (default: True)

    Returns:
        Decorated async function that returns result or error dict

    Examples:
        >>> @retry_async(max_attempts=3, base_delay=1.0)
        ... async def call_api(url):
        ...     response = await http_client.get(url)
        ...     return response.json()

        >>> result = await call_api("https://api.example.com/data")
        >>> if isinstance(result, dict) and 'error' in result:
        ...     print(f"Failed after {result['attempts']} attempts")
        ... else:
        ...     print(f"Success: {result}")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T | dict]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T | dict:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(f"{func.__name__} - Attempt {attempt}/{max_attempts}")
                    result = await func(*args, **kwargs)

                    if attempt > 1:
                        logger.info(
                            f"{func.__name__} - Succeeded on attempt {attempt}/{max_attempts}"
                        )

                    return result

                except Exception as e:
                    last_exception = e
                    exception_type = type(e).__name__
                    exception_msg = str(e)

                    logger.warning(
                        f"{func.__name__} - Attempt {attempt}/{max_attempts} failed: "
                        f"{exception_type}: {exception_msg}"
                    )

                    # If this was the last attempt, don't sleep
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} - All {max_attempts} attempts failed. "
                            f"Last error: {exception_type}: {exception_msg}"
                        )
                        break

                    # Calculate delay for next attempt
                    if exponential_backoff:
                        delay = base_delay * (2 ** (attempt - 1))
                    else:
                        delay = base_delay

                    logger.info(
                        f"{func.__name__} - Retrying in {delay:.1f}s... "
                        f"({max_attempts - attempt} attempts remaining)"
                    )
                    await asyncio.sleep(delay)

            # All attempts failed - return error dict
            error_dict = {
                'error': f"Failed after {max_attempts} attempts",
                'attempts': max_attempts,
                'last_exception': str(last_exception),
                'exception_type': type(last_exception).__name__ if last_exception else 'Unknown'
            }

            return error_dict

        return wrapper
    return decorator


def is_error_result(result: Any) -> bool:
    """
    Check if a result is an error dict from retry_async.

    Args:
        result: Result to check

    Returns:
        True if result is an error dict, False otherwise

    Examples:
        >>> result = await some_retried_function()
        >>> if is_error_result(result):
        ...     logger.error(f"Operation failed: {result['error']}")
        ... else:
        ...     process_success(result)
    """
    return (
        isinstance(result, dict) and
        'error' in result and
        'attempts' in result and
        'last_exception' in result
    )


def extract_error_message(result: dict) -> str:
    """
    Extract formatted error message from error dict.

    Args:
        result: Error dict from retry_async

    Returns:
        Formatted error message string

    Examples:
        >>> error_dict = {'error': 'Failed', 'attempts': 3, 'last_exception': 'Connection timeout'}
        >>> extract_error_message(error_dict)
        'Failed after 3 attempts. Last error: Connection timeout'
    """
    if not is_error_result(result):
        return "Invalid error result"

    return (
        f"{result['error']} after {result['attempts']} attempts. "
        f"Last error: {result['last_exception']}"
    )


# Convenience decorator with default retry settings (3 attempts, 1s/2s/4s backoff)
retry_with_backoff = retry_async(max_attempts=3, base_delay=1.0, exponential_backoff=True)
