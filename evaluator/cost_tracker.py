"""Cost and latency tracking for the evaluation pipeline."""

from typing import Optional, Dict, List
from dataclasses import dataclass, field

from .config import EvaluatorConfig
from .models import LatencyMetrics, CostMetrics
from .utils import count_tokens, calculate_cost, Timer


class CostTracker:
    """
    Tracks token usage and estimates costs for LLM operations.
    
    Supports multiple pricing tiers and models.
    """

    # Default pricing per 1K tokens in USD
    DEFAULT_PRICING = {
        "gpt-3.5-turbo": {"input":  0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "default": {"input": 0.0015, "output": 0.002},
    }

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize the cost tracker.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or EvaluatorConfig()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._records: List[Dict] = []

    def count_and_track(
        self,
        input_text: str,
        output_text: str,
        model: str = "default"
    ) -> CostMetrics:
        """
        Count tokens and calculate cost for a single operation.
        
        Args:
            input_text: The input/prompt text.
            output_text: The generated output text.
            model: Model name for pricing lookup.
            
        Returns:
            CostMetrics with token counts and cost. 
        """
        input_tokens = count_tokens(input_text)
        output_tokens = count_tokens(output_text)
        total_tokens = input_tokens + output_tokens

        # Get pricing for model
        pricing = self.DEFAULT_PRICING.get(
            model,
            self.DEFAULT_PRICING["default"]
        )

        cost = calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_1k=pricing["input"],
            output_cost_per_1k=pricing["output"]
        )

        # Update running totals
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost

        # Record this operation
        self._records.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "model": model
        })

        return CostMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost
        )

    def get_totals(self) -> Dict: 
        """
        Get cumulative totals across all tracked operations.
        
        Returns:
            Dictionary with total tokens and cost.
        """
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_cost_usd": self._total_cost,
            "operation_count": len(self._records)
        }

    def reset(self) -> None:
        """Reset all tracking counters."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._records. clear()

    def get_average_cost(self) -> float:
        """
        Get average cost per operation.
        
        Returns:
            Average cost in USD. 
        """
        if not self._records:
            return 0.0
        return self._total_cost / len(self._records)


class LatencyTracker: 
    """
    Tracks latency metrics across pipeline stages.
    
    Provides detailed timing breakdowns for optimization.
    """

    def __init__(self):
        """Initialize the latency tracker."""
        self._timer = Timer()
        self._history: List[LatencyMetrics] = []

    def get_timer(self) -> Timer:
        """
        Get the internal timer for use in evaluators.
        
        Returns:
            Timer instance. 
        """
        return self._timer

    def build_metrics(self) -> LatencyMetrics: 
        """
        Build LatencyMetrics from current timer state.
        
        Returns:
            LatencyMetrics with all timing breakdowns.
        """
        times = self._timer.get_all()

        metrics = LatencyMetrics(
            total_ms = self._timer.total(),
            relevance_ms =times.get("relevance", 0.0),
            hallucination_ms = times.get("hallucination", 0.0),
            completeness_ms = times.get("completeness", 0.0),
            embedding_ms = times.get("embedding", 0.0)
        )

        self._history.append(metrics)
        return metrics

    def reset_timer(self) -> None:
        """Reset the timer for a new evaluation."""
        self._timer = Timer()

    def get_average_latency(self) -> Dict[str, float]:
        """
        Calculate average latency across all tracked evaluations.
        
        Returns:
            Dictionary with average times per stage.
        """
        if not self._history:
            return {
                "total_ms": 0.0,
                "relevance_ms": 0.0,
                "hallucination_ms": 0.0,
                "completeness_ms": 0.0,
                "embedding_ms": 0.0
            }

        count = len(self._history)

        return {
            "total_ms": sum(m.total_ms for m in self._history) / count,
            "relevance_ms": sum(m.relevance_ms for m in self._history) / count,
            "hallucination_ms": sum(
                m.hallucination_ms for m in self._history
            ) / count,
            "completeness_ms": sum(
                m.completeness_ms for m in self._history
            ) / count,
            "embedding_ms": sum(m.embedding_ms for m in self._history) / count
        }

    def get_percentile_latency(self, percentile: int = 95) -> Dict[str, float]:
        """
        Get percentile latency metrics.
        
        Args:
            percentile: Percentile to calculate (e.g., 95 for p95).
            
        Returns:
            Dictionary with percentile times per stage.
        """
        if not self._history:
            return {"total_ms": 0.0}

        import numpy as np

        total_times = [m.total_ms for m in self._history]

        return {
            "total_ms": float(np.percentile(total_times, percentile)),
            "percentile": percentile
        }

    def clear_history(self) -> None:
        """Clear latency history."""
        self._history.clear()