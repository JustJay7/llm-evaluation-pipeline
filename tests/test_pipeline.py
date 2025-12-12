"""Unit tests for the LLM evaluation pipeline."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.config import EvaluatorConfig
from evaluator.models import (
    ChatMessage,
    ContextChunk,
    EvaluationInput,
)
from evaluator.utils import (
    split_into_sentences,
    extract_claims,
    count_tokens,
    calculate_cost,
    Timer,
)
from evaluator.relevance import RelevanceEvaluator, CompletenessEvaluator
from evaluator.hallucination import HallucinationDetector
from evaluator.cost_tracker import CostTracker, LatencyTracker
from evaluator.pipeline import EvaluationPipeline


class TestConfig(unittest.TestCase):
    """Tests for configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluatorConfig()
        self.assertEqual(config.embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(config.relevance_threshold, 0.6)
        self.assertEqual(config.hallucination_threshold, 0.5)

    def test_config_from_env(self):
        """Test configuration can be loaded from environment."""
        config = EvaluatorConfig.from_env()
        self.assertIsInstance(config, EvaluatorConfig)


class TestModels(unittest.TestCase):
    """Tests for data models."""

    def test_chat_message_normalization(self):
        """Test that role is normalized to lowercase."""
        msg = ChatMessage(role="USER", content="Hello")
        self.assertEqual(msg.role, "user")

    def test_context_chunk_defaults(self):
        """Test context chunk default values."""
        chunk = ContextChunk(content="Test content")
        self.assertEqual(chunk. score, 0.0)
        self.assertEqual(chunk.metadata, {})
        self.assertIsNone(chunk.source)

    def test_evaluation_input_properties(self):
        """Test EvaluationInput helper properties."""
        messages = [
            ChatMessage(role="user", content="What is Python?"),
            ChatMessage(role="ai", content="Python is a programming language."),
        ]
        chunks = [ContextChunk(content="Python info here.")]

        eval_input = EvaluationInput(
            conversation=messages,
            context_chunks=chunks
        )

        self.assertEqual(eval_input.last_user_message, "What is Python?")
        self.assertEqual(
            eval_input.last_ai_response,
            "Python is a programming language."
        )
        self.assertEqual(eval_input.combined_context, "Python info here.")


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        text = "This is sentence one. This is sentence two."
        sentences = split_into_sentences(text)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)

    def test_split_empty_text(self):
        """Test sentence splitting with empty text."""
        self.assertEqual(split_into_sentences(""), [])
        self.assertEqual(split_into_sentences(None), [])

    def test_extract_claims(self):
        """Test claim extraction."""
        text = "Python is fast. It has many libraries."
        claims = extract_claims(text)
        self.assertIsInstance(claims, list)

    def test_count_tokens(self):
        """Test token counting."""
        text = "Hello, how are you today?"
        tokens = count_tokens(text)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)

    def test_count_tokens_empty(self):
        """Test token counting with empty text."""
        self.assertEqual(count_tokens(""), 0)
        self.assertEqual(count_tokens(None), 0)

    def test_calculate_cost(self):
        """Test cost calculation."""
        cost = calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002
        )
        expected = (1000 / 1000 * 0.001) + (500 / 1000 * 0.002)
        self.assertAlmostEqual(cost, expected)

    def test_timer(self):
        """Test Timer class."""
        timer = Timer()
        timer.start("test_op")
        # Small delay
        for _ in range(1000):
            pass
        elapsed = timer.stop("test_op")
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(timer.get("test_op"), elapsed)


class TestCostTracker(unittest.TestCase):
    """Tests for cost tracking."""

    def test_count_and_track(self):
        """Test token counting and cost tracking."""
        tracker = CostTracker()
        metrics = tracker.count_and_track(
            input_text="Hello world",
            output_text="Hi there"
        )

        self.assertGreater(metrics. input_tokens, 0)
        self.assertGreater(metrics.output_tokens, 0)
        self.assertGreater(metrics.estimated_cost_usd, 0)

    def test_get_totals(self):
        """Test getting cumulative totals."""
        tracker = CostTracker()
        tracker.count_and_track("Input 1", "Output 1")
        tracker.count_and_track("Input 2", "Output 2")

        totals = tracker.get_totals()
        self.assertEqual(totals["operation_count"], 2)
        self.assertGreater(totals["total_tokens"], 0)

    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker()
        tracker.count_and_track("Input", "Output")
        tracker.reset()

        totals = tracker.get_totals()
        self.assertEqual(totals["operation_count"], 0)
        self.assertEqual(totals["total_tokens"], 0)


class TestLatencyTracker(unittest.TestCase):
    """Tests for latency tracking."""

    def test_timer_operations(self):
        """Test timer start/stop operations."""
        tracker = LatencyTracker()
        timer = tracker.get_timer()

        timer.start("operation")
        timer.stop("operation")

        metrics = tracker.build_metrics()
        self.assertGreaterEqual(metrics.total_ms, 0)

    def test_reset_timer(self):
        """Test timer reset."""
        tracker = LatencyTracker()
        timer = tracker.get_timer()
        timer.start("op")
        timer.stop("op")

        tracker.reset_timer()
        new_timer = tracker.get_timer()
        self.assertEqual(new_timer.total(), 0)


class TestRelevanceEvaluator(unittest. TestCase):
    """Tests for relevance evaluation."""

    @classmethod
    def setUpClass(cls):
        """Set up evaluator once for all tests."""
        cls. evaluator = RelevanceEvaluator()

    def test_relevant_response(self):
        """Test evaluation of a relevant response."""
        result = self.evaluator.evaluate(
            query="What is Python?",
            response="Python is a programming language used for web development and data science.",
            context="Python is a high-level programming language.  It is used for web development, data analysis, and automation."
        )

        self.assertGreater(result.score, 0)
        self.assertIsInstance(result.is_relevant, bool)
        self.assertGreaterEqual(result.query_response_similarity, 0)
        self.assertGreaterEqual(result.context_response_similarity, 0)

    def test_empty_response(self):
        """Test evaluation with empty response."""
        result = self.evaluator.evaluate(
            query="What is Python?",
            response="",
            context="Python is a programming language."
        )

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.is_relevant)


class TestCompletenessEvaluator(unittest.TestCase):
    """Tests for completeness evaluation."""

    @classmethod
    def setUpClass(cls):
        """Set up evaluator once for all tests."""
        cls.evaluator = CompletenessEvaluator()

    def test_complete_response(self):
        """Test evaluation of a complete response."""
        result = self.evaluator.evaluate(
            query="What are the benefits of Python?",
            response="Python has many benefits including easy syntax, large community, and extensive libraries.",
            context="Python benefits include readability and versatility."
        )

        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 1)
        self.assertIsInstance(result.is_complete, bool)

    def test_empty_response(self):
        """Test evaluation with empty response."""
        result = self.evaluator.evaluate(
            query="What is Python?",
            response="",
            context="Python is a programming language."
        )

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.is_complete)


class TestHallucinationDetector(unittest. TestCase):
    """Tests for hallucination detection."""

    @classmethod
    def setUpClass(cls):
        """Set up detector once for all tests."""
        cls.detector = HallucinationDetector()

    def test_supported_response(self):
        """Test response that is supported by context."""
        result = self.detector.evaluate(
            response="Python is a programming language.",
            context="Python is a high-level programming language created by Guido van Rossum."
        )

        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 1)
        self.assertIsInstance(result.is_hallucinated, bool)

    def test_empty_response(self):
        """Test with empty response."""
        result = self.detector.evaluate(
            response="",
            context="Some context here."
        )

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result. is_hallucinated)

    def test_no_context(self):
        """Test with no context provided."""
        result = self.detector.evaluate(
            response="Python is great.",
            context=""
        )

        self.assertEqual(result. score, 1.0)
        self.assertTrue(result.is_hallucinated)


class TestEvaluationPipeline(unittest.TestCase):
    """Tests for the main evaluation pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up pipeline once for all tests."""
        cls.pipeline = EvaluationPipeline()

    def test_evaluate_from_json(self):
        """Test full evaluation from JSON input."""
        conversation_json = {
            "conversation": [
                {"role": "user", "content":  "What is machine learning?"},
                {"role":  "ai", "content": "Machine learning is a subset of AI that enables systems to learn from data. "}
            ]
        }

        context_json = {
            "context": [
                {"content": "Machine learning is a branch of artificial intelligence. It allows computers to learn from data without being explicitly programmed."}
            ]
        }

        result = self.pipeline.evaluate_from_json(
            conversation_json,
            context_json
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)
        self.assertIsInstance(result.passed, bool)

    def test_result_to_dict(self):
        """Test that results can be converted to dictionary."""
        conversation_json = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "ai", "content": "Hi there! "}
            ]
        }
        context_json = {"context": [{"content": "Greeting context. "}]}

        result = self. pipeline.evaluate_from_json(
            conversation_json,
            context_json
        )

        result_dict = result.to_dict()
        self.assertIn("overall_score", result_dict)
        self.assertIn("passed", result_dict)
        self.assertIn("relevance", result_dict)
        self.assertIn("hallucination", result_dict)
        self.assertIn("completeness", result_dict)
        self.assertIn("latency", result_dict)
        self.assertIn("cost", result_dict)

    def test_load_input(self):
        """Test loading and parsing input JSONs."""
        conversation_json = {
            "conversation": [
                {"role": "user", "content": "Test question"}
            ]
        }
        context_json = {
            "context": [{"content": "Test context"}]
        }

        eval_input = self.pipeline.load_input(
            conversation_json,
            context_json
        )

        self.assertEqual(len(eval_input.conversation), 1)
        self.assertEqual(len(eval_input.context_chunks), 1)
        self.assertEqual(eval_input.last_user_message, "Test question")

    def test_get_statistics(self):
        """Test getting aggregate statistics."""
        stats = self.pipeline.get_statistics()
        self.assertIn("cost", stats)
        self.assertIn("latency", stats)


class TestIntegration(unittest.TestCase):
    """Integration tests with sample data files."""

    def test_sample_files_if_exist(self):
        """Test with actual sample files if they exist."""
        data_dir = Path(__file__).parent.parent / "data"
        conv_file = data_dir / "sample-chat-conversation-01.json"
        ctx_file = data_dir / "sample_context_vectors-01.json"

        if not conv_file.exists() or not ctx_file.exists():
            self.skipTest("Sample files not found in data/ folder")

        import json

        with open(conv_file, "r") as f:
            conversation_json = json.load(f)
        with open(ctx_file, "r") as f:
            context_json = json.load(f)

        pipeline = EvaluationPipeline()
        result = pipeline.evaluate_from_json(conversation_json, context_json)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)