"""Hallucination detection using NLI and semantic analysis."""

from typing import Optional, List, Tuple
import numpy as np

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from . config import EvaluatorConfig
from .models import HallucinationResult
from .utils import extract_claims, Timer


class HallucinationDetector:
    """
    Detects hallucinations in AI responses. 
    
    Uses a combination of: 
    1. Natural Language Inference (NLI) to check entailment
    2. Semantic similarity to verify claims against context
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize the hallucination detector.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self. config = config or EvaluatorConfig()
        self._nli_pipeline = None
        self._embedding_model = None

    @property
    def nli_pipeline(self):
        """Lazy load the NLI model."""
        if self._nli_pipeline is None:
            self._nli_pipeline = pipeline(
                "zero-shot-classification",
                model=self.config.nli_model,
                device=-1  # CPU, use 0 for GPU
            )
        return self._nli_pipeline

    @property
    def embedding_model(self) -> SentenceTransformer: 
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.config.embedding_model
            )
        return self._embedding_model

    def _check_claim_support(
        self,
        claim: str,
        context: str
    ) -> Tuple[bool, float]:
        """
        Check if a claim is supported by the context.
        
        Uses semantic similarity as primary check.
        
        Args:
            claim: The claim to verify.
            context: The context to check against.
            
        Returns:
            Tuple of (is_supported, confidence_score).
        """
        if not claim or not context:
            return False, 0.0

        # Get embeddings
        claim_emb = self. embedding_model.encode(
            claim, convert_to_numpy=True
        ).reshape(1, -1)
        
        # Split context into chunks for better matching
        context_sentences = context.split('.')
        context_sentences = [s.strip() for s in context_sentences if s.strip()]
        
        if not context_sentences:
            return False, 0.0

        # Get embeddings for all context sentences
        context_embs = self.embedding_model.encode(
            context_sentences, convert_to_numpy=True
        )

        # Find max similarity with any context sentence
        similarities = cosine_similarity(claim_emb, context_embs)[0]
        max_similarity = float(np.max(similarities))

        # Threshold for considering a claim supported
        support_threshold = 0.5
        is_supported = max_similarity >= support_threshold

        return is_supported, max_similarity

    def _check_entailment(
        self,
        premise: str,
        hypothesis: str
    ) -> Tuple[str, float]:
        """
        Check entailment relationship using NLI.
        
        Args:
            premise: The context/premise text.
            hypothesis: The claim to check. 
            
        Returns:
            Tuple of (label, confidence).
        """
        if not premise or not hypothesis: 
            return "neutral", 0.0

        try:
            # Truncate to avoid memory issues
            premise = premise[:1000]
            hypothesis = hypothesis[:200]

            result = self.nli_pipeline(
                hypothesis,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="{}",
                multi_label=False
            )

            top_label = result["labels"][0]
            top_score = result["scores"][0]

            return top_label, top_score

        except Exception as e:
            # Fallback on error
            return "neutral", 0.0

    def evaluate(
        self,
        response: str,
        context: str,
        timer: Optional[Timer] = None
    ) -> HallucinationResult:
        """
        Evaluate response for hallucinations.
        
        Args:
            response: The AI response to check.
            context: The ground truth context.
            timer: Optional timer for latency tracking.
            
        Returns:
            HallucinationResult with detailed findings.
        """
        if timer:
            timer.start("hallucination")

        # Handle empty inputs
        if not response: 
            return HallucinationResult(
                score=0.0,
                is_hallucinated=False,
                unsupported_claims=[],
                supported_claims=[],
                entailment_scores={}
            )

        if not context:
            # No context means everything could be hallucinated
            return HallucinationResult(
                score=1.0,
                is_hallucinated=True,
                unsupported_claims=["No context provided for verification"],
                supported_claims=[],
                entailment_scores={}
            )

        # Extract claims from response
        claims = extract_claims(response)

        if not claims:
            # No extractable claims, use full response
            claims = [response]

        supported_claims = []
        unsupported_claims = []
        entailment_scores = {}

        for claim in claims: 
            # Check semantic similarity support
            is_supported, similarity = self._check_claim_support(claim, context)

            # Store score
            claim_key = claim[: 50] + "..." if len(claim) > 50 else claim
            entailment_scores[claim_key] = similarity

            if is_supported:
                supported_claims. append(claim)
            else:
                unsupported_claims. append(claim)

        # Calculate hallucination score
        total_claims = len(claims)
        if total_claims == 0:
            hallucination_score = 0.0
        else:
            hallucination_score = len(unsupported_claims) / total_claims

        is_hallucinated = hallucination_score >= self.config.hallucination_threshold

        if timer:
            timer. stop("hallucination")

        return HallucinationResult(
            score=hallucination_score,
            is_hallucinated=is_hallucinated,
            unsupported_claims=unsupported_claims,
            supported_claims=supported_claims,
            entailment_scores=entailment_scores
        )


class FactualityChecker:
    """
    Additional factuality verification layer.
    
    Cross-references specific facts against context.
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        """
        Initialize the factuality checker.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self. config = config or EvaluatorConfig()
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer: 
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def extract_facts(self, text: str) -> List[str]:
        """
        Extract factual statements from text.
        
        Args:
            text: Input text. 
            
        Returns:
            List of factual statements.
        """
        import re

        facts = []

        # Pattern for statements with numbers
        number_pattern = r'[^. ]*\d+[^.]*\.'
        number_matches = re.findall(number_pattern, text)
        facts.extend([m.strip() for m in number_matches])

        # Pattern for definitive statements
        definitive_patterns = [
            r'[^.]*\bis\b[^.]*\.',
            r'[^.]*\bare\b[^.]*\.',
            r'[^.]*\bwas\b[^.]*\.',
            r'[^. ]*\bwere\b[^.]*\.',
        ]

        for pattern in definitive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts.extend([m.strip() for m in matches[: 3]])  # Limit per pattern

        # Remove duplicates while preserving order
        seen = set()
        unique_facts = []
        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique_facts.append(fact)

        return unique_facts[: 10]  # Limit total facts

    def verify_facts(
        self,
        facts: List[str],
        context: str
    ) -> Tuple[List[str], List[str]]:
        """
        Verify facts against context.
        
        Args:
            facts: List of facts to verify.
            context: Context to verify against.
            
        Returns:
            Tuple of (verified_facts, unverified_facts).
        """
        if not facts or not context:
            return [], facts if facts else []

        verified = []
        unverified = []

        context_emb = self.model.encode(context, convert_to_numpy=True)

        for fact in facts:
            fact_emb = self.model.encode(fact, convert_to_numpy=True)
            similarity = cosine_similarity(
                fact_emb. reshape(1, -1),
                context_emb.reshape(1, -1)
            )[0][0]

            if similarity >= 0.45: 
                verified.append(fact)
            else:
                unverified.append(fact)

        return verified, unverified