"""Generate human-readable explanations and improvement suggestions."""

from typing import List, Dict, Any


def generate_explanation(result) -> Dict[str, Any]:
    """
    Generate human-readable explanations for evaluation results.
    
    Args:
        result: EvaluationResult object
        
    Returns:
        Dictionary with explanations and suggestions
    """
    explanations = []
    suggestions = []
    
    # --- Relevance Explanations ---
    rel = result.relevance
    if rel.score < 0.6:
        if rel.query_response_similarity < 0.5:
            explanations.append(
                f"Response is not directly addressing the user's question.  "
                f"(Query-Response similarity: {rel.query_response_similarity:.0%})"
            )
            suggestions.append(
                "💡 Improve the LLM prompt to focus more on answering the specific question asked."
            )
        if rel.context_response_similarity < 0.5:
            explanations.append(
                f"Response does not align well with the retrieved context. "
                f"(Context-Response similarity: {rel.context_response_similarity:.0%})"
            )
            suggestions.append(
                "Consider improving RAG retrieval to fetch more relevant context chunks."
            )
    else:
        explanations.append(
            f"Response is relevant to both the query and context. "
            f"(Score: {rel.score:.0%})"
        )
    
    # --- Hallucination Explanations ---
    hall = result.hallucination
    unsupported_count = len(hall.unsupported_claims or [])
    supported_count = len(hall.supported_claims or [])

    if hall.is_hallucinated:
        explanations.append(
            f"Response contains hallucinated content. "
            f"({unsupported_count} unsupported claims detected)"
        )
        suggestions.append(
            "Add more context vectors to support the claims, or constrain the LLM to only use provided context."
        )
    elif hall.unsupported_claims:
        total_claims = unsupported_count + supported_count
        explanations.append(
            f"Some claims lack direct support in context: {unsupported_count} unsupported "
            f"out of {total_claims} total claims."
        )
        for claim in hall.unsupported_claims[:3]:  # Show max 3
            truncated = claim[:100] + "..." if len(claim) > 100 else claim
            explanations.append(f"   • Unsupported: \"{truncated}\"")
    else:
        explanations.append(
            f"All claims in the response are supported by context. "
            f"({supported_count} claims verified)"
        )

    # --- Completeness Explanations ---
    comp = result.completeness
    if comp.score < 0.5:
        explanations.append(
            f"Response is incomplete. Missing aspects of the user's question."
        )
        if comp.missing_aspects:
            for aspect in comp.missing_aspects[: 3]:
                explanations.append(f"   • Not addressed: \"{aspect[:80]}\"")
        suggestions.append(
            "Ensure the LLM addresses all parts of multi-part questions."
        )
    elif comp.score < 1.0:
        explanations.append(
            f"Response partially addresses the question.  (Score: {comp.score:.0%})"
        )
    else:
        explanations.append(
            f"Response fully addresses the user's question."
        )
    
    # --- Latency Explanations ---
    lat = result.latency
    if lat.total_ms > 5000:
        explanations.append(
            f"High latency detected: {lat.total_ms:.0f}ms total."
        )
        # Find bottleneck
        stages = {
            "Relevance": lat.relevance_ms,
            "Hallucination": lat.hallucination_ms,
            "Completeness": lat.completeness_ms
        }
        bottleneck = max(stages, key=stages.get)
        suggestions.append(
            f"Latency bottleneck is in {bottleneck} evaluation ({stages[bottleneck]:.0f}ms). "
            f"Consider optimizing or caching."
        )
    
    # --- Cost Explanations ---
    cost = result.cost
    if cost.total_tokens > 10000:
        explanations.append(
            f"High token usage: {cost.total_tokens:,} tokens (${cost.estimated_cost_usd:.4f})"
        )
        suggestions.append(
            "Consider truncating context or summarizing before evaluation to reduce costs."
        )
    
    # --- Overall Summary ---
    if result.passed:
        summary = f"PASSED - Overall score: {result.overall_score:.0%}"
    else:
        summary = f"FAILED - Overall score: {result.overall_score:.0%}"
        
        # Why did it fail?
        failure_reasons = []
        if result.overall_score < 0.7:
            failure_reasons.append("overall score below 70%")
        if rel.score < 0.6:
            failure_reasons.append("relevance below threshold")
        if hall.is_hallucinated:
            failure_reasons.append("hallucination detected")
            
        if failure_reasons: 
            summary += f" (Failed due to: {', '.join(failure_reasons)})"
    
    return {
        "summary": summary,
        "explanations": explanations,
        "suggestions": suggestions,
        "passed": result.passed,
        "overall_score": result.overall_score
    }


def format_explanation_text(explanation_dict: Dict[str, Any]) -> str:
    """Format explanation dictionary as readable text."""
    lines = []
    
    lines.append("EVALUATION EXPLANATION")
    lines.append("")
    lines.append(explanation_dict["summary"])
    lines.append("")
    
    lines.append("Details:")
    for exp in explanation_dict["explanations"]:
        lines.append(exp)
    
    if explanation_dict["suggestions"]: 
        lines.append("")
        lines.append("Suggestions for Improvement:")
        for sug in explanation_dict["suggestions"]:
            lines.append(sug)
    
    lines.append("")
    
    return "\n".join(lines)