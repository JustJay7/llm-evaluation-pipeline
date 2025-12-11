"""Confidence scoring for evaluation results."""


def calculate_confidence(result) -> float:
    """
    Calculate confidence in the evaluation result.
    
    Higher confidence when:
    - More context was available
    - Scores are not borderline (near thresholds)
    - Multiple claims were checkable
    
    Args:
        result: EvaluationResult object
        
    Returns:
        Confidence score between 0 and 1
    """
    confidence = 1.0
    
    # Lower confidence for borderline relevance scores
    relevance = result.relevance.score
    if 0.5 <= relevance <= 0.7:  # Near threshold
        confidence *= 0.8
    
    # Lower confidence for borderline hallucination scores
    hallucination = result.hallucination.score
    if 0.4 <= hallucination <= 0.6:
        confidence *= 0.8
    
    # Get claim counts from hallucination result
    unsupported_count = len(result.hallucination.unsupported_claims)
    
    # Try to get supported count - handle different attribute names
    supported_count = 0
    if hasattr(result.hallucination, 'supported_claims_count'):
        supported_count = result.hallucination.supported_claims_count
    elif hasattr(result.hallucination, 'supported_count'):
        supported_count = result.hallucination.supported_count
    elif hasattr(result.hallucination, 'details') and isinstance(result.hallucination.details, dict):
        supported_count = result.hallucination.details.get('supported_claims_count', 0)
    
    total_claims = unsupported_count + supported_count
    
    # Lower confidence if few claims were checked
    if total_claims < 2:
        confidence *= 0.7
    elif total_claims > 5:
        confidence *= 1.1  # More data = more confident
    
    # Lower confidence for borderline completeness
    completeness = result.completeness.score
    if 0.4 <= completeness <= 0.6:
        confidence *= 0.85
    
    # Ensure confidence is between 0 and 1
    return min(1.0, max(0.0, confidence))