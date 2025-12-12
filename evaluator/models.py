"""Data models for the evaluation pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class ChatMessage: 
    """Single message in a conversation."""

    role: str
    content: str
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Normalize role to lowercase."""
        self.role = self.role.lower()


@dataclass
class ContextChunk:
    """A single context chunk from the vector database."""

    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None


@dataclass
class EvaluationInput:
    """Structured input for the evaluation pipeline."""

    conversation: List[ChatMessage]
    context_chunks: List[ContextChunk]

    @property
    def last_user_message(self) -> Optional[str]:
        """Get the most recent user query."""
        for msg in reversed(self.conversation):
            if msg.role == "user":
                return msg.content
        return None

    @property
    def last_ai_response(self) -> Optional[str]:
        """Get the most recent AI response."""
        for msg in reversed(self.conversation):
            if msg.role in ("ai", "assistant", "bot"):
                return msg.content
        return None

    @property
    def combined_context(self) -> str:
        """Merge all context chunks into a single string."""
        return "\n\n".join(chunk.content for chunk in self.context_chunks)


@dataclass
class RelevanceResult:
    """Results from relevance evaluation."""

    score: float
    is_relevant: bool
    query_response_similarity: float
    context_response_similarity: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HallucinationResult:
    """Results from hallucination detection."""

    score: float  # 0 = no hallucination, 1 = full hallucination
    is_hallucinated: bool
    unsupported_claims: List[str] = field(default_factory=list)
    supported_claims: List[str] = field(default_factory=list)
    entailment_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class CompletenessResult: 
    """Results from completeness evaluation."""

    score: float
    is_complete: bool
    covered_aspects: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)


@dataclass
class LatencyMetrics:
    """Latency breakdown for the evaluation."""

    total_ms: float
    relevance_ms: float
    hallucination_ms: float
    completeness_ms: float
    embedding_ms: float


@dataclass
class CostMetrics:
    """Cost tracking for the evaluation."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass
class EvaluationResult: 
    """Complete evaluation result combining all metrics."""

    relevance: RelevanceResult
    hallucination: HallucinationResult
    completeness: CompletenessResult
    latency: LatencyMetrics
    cost: CostMetrics
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    overall_score: float = 0.0
    passed: bool = False

    def __post_init__(self):
        """Calculate overall score after initialization."""
        self._calculate_overall_score()

    def _calculate_overall_score(self):
        """Compute weighted average score."""
        weights = {
            "relevance": 0.35,
            "hallucination": 0.40,
            "completeness": 0.25
        }

        # Invert hallucination (lower = better)
        hallucination_inverted = 1.0 - self.hallucination.score

        self.overall_score = (
            weights["relevance"] * self.relevance.score
            + weights["hallucination"] * hallucination_inverted
            + weights["completeness"] * self.completeness.score
        )

        self.passed = (
            self.relevance.is_relevant
            and not self.hallucination.is_hallucinated
            and self.completeness.is_complete
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "timestamp": self.timestamp,
            "relevance": {
                "score": round(self.relevance.score, 4),
                "is_relevant": self.relevance.is_relevant,
                "query_response_similarity": round(
                    self.relevance.query_response_similarity, 4
                ),
                "context_response_similarity": round(
                    self.relevance.context_response_similarity, 4
                ),
            },
            "hallucination": {
                "score": round(self.hallucination.score, 4),
                "is_hallucinated": self.hallucination.is_hallucinated,
                "unsupported_claims": self.hallucination.unsupported_claims[:5],
                "supported_claims_count": len(
                    self.hallucination.supported_claims
                ),
            },
            "completeness": {
                "score": round(self.completeness.score, 4),
                "is_complete": self.completeness.is_complete,
                "covered_aspects": self.completeness.covered_aspects,
                "missing_aspects": self.completeness.missing_aspects,
            },
            "latency": {
                "total_ms": round(self.latency.total_ms, 2),
                "relevance_ms": round(self.latency.relevance_ms, 2),
                "hallucination_ms": round(self.latency.hallucination_ms, 2),
                "completeness_ms": round(self.latency.completeness_ms, 2),
                "embedding_ms": round(self.latency.embedding_ms, 2),
            },
            "cost":  {
                "input_tokens": self.cost.input_tokens,
                "output_tokens": self.cost.output_tokens,
                "total_tokens": self.cost.total_tokens,
                "estimated_cost_usd": round(self.cost.estimated_cost_usd, 6),
            },
        }