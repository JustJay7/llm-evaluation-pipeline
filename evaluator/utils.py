"""Utility functions for text processing, timing, and token counting."""

import re
import time
from typing import List
from contextlib import contextmanager

import tiktoken


# Regex pattern to split sentences
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex. 

    Args:
        text: Input text to split.

    Returns:
        List of sentences.
    """
    if not text or not text.strip():
        return []

    sentences = SENTENCE_PATTERN.split(text.strip())
    # Filter out very short fragments (likely not real sentences)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def extract_claims(text: str) -> List[str]:
    """
    Extract individual claims from text. 

    Splits on sentence boundaries and common conjunctions
    to get more granular claims for fact-checking.

    Args:
        text: Input text to extract claims from.

    Returns:
        List of individual claims.
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    claims = []

    for sentence in sentences:
        # Split on conjunctions for more granular claims
        parts = re.split(r'\s+(?: and|but|however|also|additionally)\s+', sentence)
        for part in parts:
            part = part.strip()
            if len(part) > 15:  # Filter tiny fragments
                claims.append(part)

    return claims


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for.
        model: Model name for tokenizer selection.

    Returns:
        Number of tokens.
    """
    if not text: 
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float = 0.0015,
    output_cost_per_1k: float = 0.002
) -> float:
    """
    Calculate estimated cost based on token counts.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        input_cost_per_1k:  Cost per 1000 input tokens.
        output_cost_per_1k: Cost per 1000 output tokens.

    Returns:
        Estimated cost in USD.
    """
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return input_cost + output_cost


@contextmanager
def timer():
    """
    Context manager for timing code blocks. 

    Yields:
        A function that returns elapsed time in milliseconds. 

    Example:
        with timer() as get_time:
            # do something
            elapsed = get_time()
    """
    start = time.perf_counter()
    elapsed_holder = [0.0]

    def get_elapsed() -> float:
        elapsed_holder[0] = (time.perf_counter() - start) * 1000
        return elapsed_holder[0]

    yield get_elapsed


class Timer:
    """
    Timer class for tracking multiple timing measurements. 

    Useful for tracking latency across different pipeline stages.
    """

    def __init__(self):
        """Initialize empty timing records."""
        self._records: dict = {}
        self._start_times: dict = {}

    def start(self, name: str) -> None:
        """Start timing for a named operation."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Stop timing for a named operation. 

        Args:
            name:  Name of the operation. 

        Returns:
            Elapsed time in milliseconds.
        """
        if name not in self._start_times:
            return 0.0

        elapsed = (time.perf_counter() - self._start_times[name]) * 1000
        self._records[name] = elapsed
        return elapsed

    def get(self, name: str) -> float:
        """Get recorded time for a named operation."""
        return self._records.get(name, 0.0)

    def get_all(self) -> dict:
        """Get all timing records."""
        return self._records. copy()

    def total(self) -> float:
        """Get total time across all operations."""
        return sum(self._records.values())


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text. 

    Returns:
        Normalized lowercase text with extra whitespace removed.
    """
    if not text:
        return ""

    # Lowercase and normalize whitespace
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to max length with ellipsis.

    Args:
        text: Input text. 
        max_length: Maximum length before truncation.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."