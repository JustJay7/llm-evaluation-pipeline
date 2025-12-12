"""Relevance evaluation using semantic similarity."""

from typing import Optional, List
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import EvaluatorConfig
from .models import RelevanceResult, CompletenessResult
from .utils import extract_claims, Timer


class RelevanceEvaluator: 
    """
    Evaluates response relevance using embedding similarity.
    
    Compares the AI response against both the user query
    and the retrieved context to determine relevance.
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize the relevance evaluator.
        
        Args:
            config: Configuration object.  Uses defaults if None.
        """
        self.config = config or EvaluatorConfig()
        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: dict = {}

    @property
    def model(self) -> SentenceTransformer: 
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with optional caching.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        if not text:
            return np.zeros(384)  # Default dimension for MiniLM

        if self.config.cache_embeddings:
            cache_key = hash(text[:500])  # Use prefix for long texts
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

        embedding = self.model.encode(text, convert_to_numpy=True)

        if self.config.cache_embeddings:
            self._embedding_cache[cache_key] = embedding

        return embedding

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0

        emb1 = self._get_embedding(text1).reshape(1, -1)
        emb2 = self._get_embedding(text2).reshape(1, -1)

        similarity = cosine_similarity(emb1, emb2)[0][0]
        # Ensure result is between 0 and 1
        return float(max(0.0, min(1.0, similarity)))

    def evaluate(
        self,
        query: str,
        response: str,
        context: str,
        timer: Optional[Timer] = None
    ) -> RelevanceResult: 
        """
        Evaluate response relevance.
        
        Args:
            query: User's question.
            response: AI's response.
            context: Retrieved context from vector DB.
            timer: Optional timer for latency tracking.
            
        Returns:
            RelevanceResult with scores and details.
        """
        if timer:
            timer.start("relevance")

        # Handle empty inputs
        if not response: 
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                query_response_similarity=0.0,
                context_response_similarity=0.0,
                details={"error": "Empty response"}
            )

        # Calculate similarity scores
        query_sim = self._compute_similarity(query, response)
        context_sim = self._compute_similarity(context, response)

        # Combined score:  weight context higher since RAG relies on it
        combined_score = (0.4 * query_sim) + (0.6 * context_sim)

        is_relevant = combined_score >= self.config.relevance_threshold

        if timer:
            timer.stop("relevance")

        return RelevanceResult(
            score=combined_score,
            is_relevant=is_relevant,
            query_response_similarity=query_sim,
            context_response_similarity=context_sim,
            details={
                "threshold": self.config.relevance_threshold,
                "query_weight": 0.4,
                "context_weight": 0.6
            }
        )

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()