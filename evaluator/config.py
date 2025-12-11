"""Configuration management for the evaluation pipeline."""

import os
from dataclasses import dataclass


@dataclass
class EvaluatorConfig:
    """Central configuration for all evaluation components."""

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "facebook/bart-large-mnli"

    # Thresholds (tune these based on your needs)
    relevance_threshold:  float = 0.6
    hallucination_threshold:  float = 0.5
    completeness_threshold: float = 0.5

    # Cost tracking (per 1K tokens - adjust for your LLM provider)
    input_token_cost: float = 0.0015
    output_token_cost:  float = 0.002

    # Performance settings
    batch_size: int = 32
    max_workers:  int = 4
    cache_embeddings: bool = True

    @classmethod
    def from_env(cls) -> "EvaluatorConfig":
        """Load config from environment variables with fallbacks."""
        return cls(
            embedding_model=os.getenv(
                "EVAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            relevance_threshold=float(
                os.getenv("EVAL_RELEVANCE_THRESHOLD", 0.6)
            ),
            hallucination_threshold=float(
                os.getenv("EVAL_HALLUCINATION_THRESHOLD", 0.5)
            ),
            input_token_cost=float(
                os.getenv("EVAL_INPUT_TOKEN_COST", 0.0015)
            ),
            output_token_cost=float(
                os.getenv("EVAL_OUTPUT_TOKEN_COST", 0.002)
            ),
            batch_size=int(os.getenv("EVAL_BATCH_SIZE", 32)),
            max_workers=int(os.getenv("EVAL_MAX_WORKERS", 4)),
        )