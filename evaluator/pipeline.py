"""Main evaluation pipeline that orchestrates all components."""

import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import EvaluatorConfig
from .models import (
    EvaluationInput,
    EvaluationResult,
    ChatMessage,
    ContextChunk,
)
from .relevance import RelevanceEvaluator
from .completeness import CompletenessEvaluator
from .hallucination import HallucinationDetector
from .cost_tracker import CostTracker, LatencyTracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Main pipeline for evaluating LLM responses. 

    Combines relevance, hallucination, and completeness checks
    with latency and cost tracking.
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize the evaluation pipeline. 

        Args:
            config: Configuration object.  Uses defaults if None.
        """
        self.config = config or EvaluatorConfig()

        # Initialize evaluators (lazy loaded internally)
        self._relevance_evaluator:  Optional[RelevanceEvaluator] = None
        self._completeness_evaluator: Optional[CompletenessEvaluator] = None
        self._hallucination_detector: Optional[HallucinationDetector] = None

        # Initialize trackers
        self.cost_tracker = CostTracker(self.config)
        self.latency_tracker = LatencyTracker()

        self._initialized = False
        logger.info("Evaluation pipeline created")

    @property
    def relevance_evaluator(self) -> RelevanceEvaluator: 
        """Lazy load relevance evaluator."""
        if self._relevance_evaluator is None:
            self._relevance_evaluator = RelevanceEvaluator(self.config)
        return self._relevance_evaluator

    @property
    def completeness_evaluator(self) -> CompletenessEvaluator:
        """Lazy load completeness evaluator."""
        if self._completeness_evaluator is None:
            self._completeness_evaluator = CompletenessEvaluator(self.config)
        return self._completeness_evaluator

    @property
    def hallucination_detector(self) -> HallucinationDetector: 
        """Lazy load hallucination detector."""
        if self._hallucination_detector is None: 
            self._hallucination_detector = HallucinationDetector(self.config)
        return self._hallucination_detector

    def _parse_conversation_json(self, data: Dict) -> List[ChatMessage]:
        """
        Parse conversation JSON into ChatMessage objects.

        Handles BeyondChats format with conversation_turns. 

        Args:
            data: Raw JSON data containing conversation. 

        Returns:
            List of ChatMessage objects.
        """
        messages = []

        # Handle BeyondChats format:  conversation_turns
        if "conversation_turns" in data:
            for turn in data["conversation_turns"]:
                role = turn.get("role", "").lower()
                # Normalize role names
                if role in ("ai/chatbot", "ai", "assistant", "bot", "chatbot"):
                    role = "ai"
                elif role == "user":
                    role = "user"

                content = turn.get("message", "")
                timestamp = turn.get("created_at", None)

                if content: 
                    messages.append(ChatMessage(
                        role=role,
                        content=content,
                        timestamp=timestamp
                    ))
            return messages

        # Handle other formats
        conversation_data = data.get("conversation", data.get("messages", []))

        if isinstance(conversation_data, list):
            for msg in conversation_data:
                if isinstance(msg, dict):
                    role = msg.get("role", msg.get("sender", "user"))
                    content = msg.get("content", msg.get("message", ""))
                    timestamp = msg.get("timestamp", None)

                    messages.append(ChatMessage(
                        role=role,
                        content=content,
                        timestamp=timestamp
                    ))

        return messages

    def _parse_context_json(self, data: Dict) -> List[ContextChunk]: 
        """
        Parse context JSON into ContextChunk objects.

        Handles BeyondChats format with data.vector_data.

        Args:
            data: Raw JSON data containing context vectors.

        Returns:
            List of ContextChunk objects. 
        """
        chunks = []

        # Handle BeyondChats format: data.vector_data
        if "data" in data and isinstance(data["data"], dict):
            vector_data = data["data"]. get("vector_data", [])
            if vector_data:
                for item in vector_data:
                    content = item.get("text", "")
                    # Use tokens as a proxy for score if no score provided
                    score = item.get("score", 0.0)
                    source = item.get("source_url", None)
                    metadata = {
                        "id": item.get("id"),
                        "tokens": item.get("tokens"),
                        "created_at": item.get("created_at")
                    }

                    if content:
                        chunks.append(ContextChunk(
                            content=content,
                            score=score,
                            metadata=metadata,
                            source=source
                        ))
                return chunks

        # Handle other formats
        context_data = data.get(
            "context",
            data.get("vectors", data.get("chunks", []))
        )

        if isinstance(context_data, list):
            for item in context_data:
                if isinstance(item, dict):
                    content = item.get("content", item.get("text", ""))
                    score = item.get("score", item.get("similarity", 0.0))
                    metadata = item.get("metadata", {})
                    source = item.get("source", item.get("url", None))

                    if content:
                        chunks.append(ContextChunk(
                            content=content,
                            score=score,
                            metadata=metadata,
                            source=source
                        ))
                elif isinstance(item, str):
                    chunks.append(ContextChunk(content=item))

        elif isinstance(context_data, str):
            chunks.append(ContextChunk(content=context_data))

        return chunks

    def load_input(
        self,
        conversation_json: Dict,
        context_json: Dict
    ) -> EvaluationInput:
        """
        Load and parse input JSONs into EvaluationInput. 

        Args:
            conversation_json: JSON containing chat conversation.
            context_json: JSON containing context vectors.

        Returns:
            Structured EvaluationInput object. 
        """
        conversation = self._parse_conversation_json(conversation_json)
        context_chunks = self._parse_context_json(context_json)

        logger.info(
            f"Loaded {len(conversation)} messages and "
            f"{len(context_chunks)} context chunks"
        )

        return EvaluationInput(
            conversation=conversation,
            context_chunks=context_chunks
        )

    def load_from_files(
        self,
        conversation_path: str,
        context_path: str
    ) -> EvaluationInput:
        """
        Load input from JSON files.

        Args:
            conversation_path: Path to conversation JSON file.
            context_path: Path to context JSON file. 

        Returns:
            Structured EvaluationInput object.
        """
        with open(conversation_path, "r", encoding="utf-8") as f:
            conversation_json = json.load(f)

        with open(context_path, "r", encoding="utf-8") as f:
            context_json = json.load(f)

        return self.load_input(conversation_json, context_json)

    def evaluate(self, eval_input: EvaluationInput) -> EvaluationResult: 
        """
        Run full evaluation on the input.

        Args:
            eval_input: Structured evaluation input.

        Returns:
            Complete EvaluationResult with all metrics.
        """
        logger.info("Starting evaluation")

        # Reset latency tracker for this evaluation
        self.latency_tracker.reset_timer()
        timer = self.latency_tracker.get_timer()

        # Extract key components
        query = eval_input.last_user_message or ""
        response = eval_input.last_ai_response or ""
        context = eval_input.combined_context

        logger.debug(f"Query: {query[: 100]}...")
        logger.debug(f"Response: {response[:100]}...")
        logger.debug(f"Context length: {len(context)} chars")

        # Run relevance evaluation
        relevance_result = self.relevance_evaluator.evaluate(
            query=query,
            response=response,
            context=context,
            timer=timer
        )
        logger.info(f"Relevance score: {relevance_result.score:.4f}")

        # Run hallucination detection
        hallucination_result = self.hallucination_detector.evaluate(
            response=response,
            context=context,
            timer=timer
        )
        logger.info(f"Hallucination score: {hallucination_result.score:.4f}")

        # Run completeness evaluation
        completeness_result = self.completeness_evaluator.evaluate(
            query=query,
            response=response,
            context=context,
            timer=timer
        )
        logger.info(f"Completeness score: {completeness_result.score:.4f}")

        # Build latency metrics
        latency_metrics = self.latency_tracker.build_metrics()

        # Calculate cost metrics
        input_text = query + context
        cost_metrics = self.cost_tracker.count_and_track(
            input_text=input_text,
            output_text=response
        )

        # Build final result
        result = EvaluationResult(
            relevance=relevance_result,
            hallucination=hallucination_result,
            completeness=completeness_result,
            latency=latency_metrics,
            cost=cost_metrics
        )

        logger.info(
            f"Evaluation complete. Overall score: {result.overall_score:.4f}"
        )
        logger.info(f"Passed: {result.passed}")

        return result

    def evaluate_from_json(
        self,
        conversation_json: Dict,
        context_json: Dict
    ) -> EvaluationResult:
        """
        Convenience method to evaluate directly from JSON dicts.

        Args:
            conversation_json: JSON containing chat conversation.
            context_json: JSON containing context vectors.

        Returns:
            Complete EvaluationResult. 
        """
        eval_input = self.load_input(conversation_json, context_json)
        return self.evaluate(eval_input)

    def evaluate_batch(
        self,
        inputs: List[EvaluationInput]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple inputs. 

        Args:
            inputs:  List of EvaluationInput objects.

        Returns:
            List of EvaluationResult objects. 
        """
        results = []
        total = len(inputs)

        for idx, eval_input in enumerate(inputs, 1):
            logger.info(f"Evaluating {idx}/{total}")
            result = self.evaluate(eval_input)
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics from all evaluations.

        Returns:
            Dictionary with cost and latency statistics.
        """
        return {
            "cost": self.cost_tracker.get_totals(),
            "latency": {
                "average": self.latency_tracker.get_average_latency(),
                "p95": self.latency_tracker.get_percentile_latency(95)
            }
        }

    def reset_statistics(self) -> None:
        """Reset all tracked statistics."""
        self.cost_tracker.reset()
        self.latency_tracker.clear_history()