"""General utility functions."""

import json
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
        logger_ = logging.getLogger(logger_name)
    else:
        logger_ = logging.getLogger()  # Root logger

    logger_.setLevel(logging.CRITICAL + 1)  # Set level above CRITICAL
    logger_.propagate = False  # Prevent messages from propagating to parent loggers
    for handler in logger_.handlers:
        logger_.removeHandler(handler)  # Remove existing handlers


def parse_file_by_paragraph(file_path: str, least_newlines: int = 2) -> list[str]:
    """
    Reads a .txt file and parses its content into paragraphs.
    Paragraphs are separated by a specified minimum number of consecutive newlines.

    Args:
        file_path (str): Path to the .txt file.
        least_newlines (int): Minimum number of consecutive newlines to define a paragraph break.

    Returns:
        list[str]: A list of strings, each representing a paragraph.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Create a regex pattern for splitting based on the least number of newlines
        newline_pattern = rf"\n{{{least_newlines},}}"  # Matches least_newlines or more consecutive newlines

        # Split content into paragraphs based on the pattern
        paragraphs = re.split(newline_pattern, content)

        # Clean up whitespace from each paragraph and filter out empty ones
        paragraphs = [
            paragraph.strip() for paragraph in paragraphs if paragraph.strip()
        ]

        return paragraphs
    except FileNotFoundError:
        logger.debug(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        logger.debug(f"An error occurred: {e}")
        return []


def write_json(data: dict, file_path: str):
    """
    Save a dictionary as a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path to save the JSON file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.debug(f"An error occurred: {e}")


def read_json(file_path: str) -> dict:
    """
    Load a JSON file as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The dictionary loaded from the JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logger.debug(f"Error: File not found at {file_path}")
        return {}
    except Exception as e:
        logger.debug(f"An error occurred: {e}")
        return {}


def chunk_by_tokens(
    filename: str, num_tokens: int, num_tokens_per_word: int = 2
) -> list[str]:
    """
    Reads a text file and returns a list of string chunks, each about `num_tokens` tokens long.
    Tokens are approximated as: tokens = words * num_tokens_per_word.

    Steps:
    1. Read the entire text file.
    2. Split the text into words by whitespace.
    3. Calculate how many words per chunk: words_per_chunk = num_tokens // num_tokens_per_word.
    4. Divide the list of words into chunks of that many words.
    5. Join each chunk's words into a single string.
    6. Return the list of string chunks.

    Args:
        filename (str): Path to the text file.
        num_tokens (int): Desired approximate number of tokens per chunk.
        num_tokens_per_word (int): Average tokens per word. Default is 2.

    Returns:
        list[str]: A list of chunks, each chunk is a single string containing about `num_tokens` tokens.
    """
    # Calculate how many words correspond to the desired number of tokens
    words_per_chunk = max(num_tokens // num_tokens_per_word, 1)

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into words
    words = text.split()

    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        # Join words into a single string for the chunk
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)

    return chunks
