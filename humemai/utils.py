"""General utility functions."""

import logging
from datetime import datetime


def is_iso8601_datetime(value: str) -> bool:
    """
    Check if the given string is in ISO 8601 datetime format with seconds precision.

    Args:
        value (str): The string to check.

    Returns:
        bool: True if the string is a valid ISO 8601 datetime, False otherwise.
    """
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        return True
    except ValueError:
        return False


def disable_logger(logger_name: str = None):
    """
    Disables a specific logger or the root logger.

    Args:
        logger_name (str | None): The name of the logger to disable. If None, disables
        the root logger.
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()  # Root logger

    logger.setLevel(logging.CRITICAL + 1)  # Set level above CRITICAL
    logger.propagate = False  # Prevent messages from propagating to parent loggers
    for handler in logger.handlers:
        logger.removeHandler(handler)  # Remove existing handlers
