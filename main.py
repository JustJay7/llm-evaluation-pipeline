#!/usr/bin/env python3
"""
LLM Response Evaluation Pipeline - Main Entry Point.  

Usage: 
    python main.py --demo
    python main.py --samples
    python main.py --samples --report
    python main.py -c <conversation. json> -x <context.json>
"""

import argparse
import json
import sys
import random
import time
import statistics
from pathlib import Path

from evaluator.explainer import generate_explanation, format_explanation_text
from evaluator.pipeline import EvaluationPipeline
from evaluator.config import EvaluatorConfig
from evaluator.report import generate_html_report
from evaluator.confidence import calculate_confidence
from evaluator.history import log_evaluation, get_stats


def load_json_file(filepath: str) -> dict:
    """Load JSON from file path."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def print_results(result, show_confidence: bool = True):
    """Print evaluation results in a formatted way."""
    result_dict = result.to_dict()
    
    # Calculate confidence
    confidence = calculate_confidence(result)
    
    print()
    print("EVALUATION RESULTS")
    print(json.dumps(result_dict, indent=2))

    print()
    print("SUMMARY")
    print(f"Overall Score:     {result.overall_score:.4f}")
    print(f"Passed:           {result.passed}")
    if show_confidence:
        print(f"Confidence:       {confidence:.1%}")
    print(f"Relevance:        {result.relevance.score:.4f} ({'PASS' if result.relevance.is_relevant else 'FAIL'})")
    print(f"Hallucination:    {result.hallucination.score:.4f} ({'FAIL' if result.hallucination.is_hallucinated else 'PASS'})")
    print(f"Completeness:     {result.completeness.score:.4f} ({'PASS' if result.completeness.is_complete else 'FAIL'})")
    print(f"Total Latency:    {result.latency.total_ms:.2f} ms")
    print(f"Estimated Cost:   ${result.cost.estimated_cost_usd:.6f}")
    print()
    
    # Highlight unsupported claims
    if result.hallucination.unsupported_claims:
        print("UNSUPPORTED CLAIMS DETECTED:")
        for claim in result.hallucination.unsupported_claims:
            truncated = claim[:100] + "..." if len(claim) > 100 else claim
            print(f"  • {truncated}")
        print()
    
    # Generate and print explanation
    explanation = generate_explanation(result)
    print(format_explanation_text(explanation))

def analyze_hallucinations(result):
    """Analyze hallucinations for known issues."""
    print()
    print("HALLUCINATION ANALYSIS")
    
    known_issues_found = False
    
    for claim in result.hallucination.unsupported_claims:
        claim_lower = claim.lower()
        
        # Check for known hallucination patterns
        if "subsidized" in claim_lower or ("room" in claim_lower and "clinic" in claim_lower):
            print(f"CAUGHT KNOWN ISSUE: {claim[:80]}...")
            known_issues_found = True
        elif "book" in claim_lower and "consultation" in claim_lower:
            print(f"Marketing claim (may not be in context): {claim[:60]}...")
    
    if not known_issues_found and not result.hallucination.unsupported_claims:
        print("No hallucinations detected!")
    elif not known_issues_found:
        print(f"Found {len(result.hallucination.unsupported_claims)} unsupported claim(s)")
    
    print()

def run_demo():
    """Run a demonstration with sample data."""
    print("🚀 LLM Evaluation Pipeline - Demo Mode")
    print()

    conversation_json = {
        "conversation_turns": [
            {
                "role": "User",
                "message": "What are the benefits of using Python for data science?"
            },
            {
                "role": "AI/Chatbot",
                "message": (
                    "Python is excellent for data science due to several reasons:  "
                    "1) It has powerful libraries like NumPy, Pandas, and Scikit-learn.  "
                    "2) It has a simple and readable syntax that makes it easy to learn. "
                    "3) It has strong community support with extensive documentation. "
                    "4) It integrates well with other languages and tools."
                )
            }
        ]
    }

    context_json = {
        "data": {
            "vector_data": [
                {
                    "text": (
                        "Python is a popular programming language for data science.  "
                        "It offers libraries such as NumPy for numerical computing, "
                        "Pandas for data manipulation, and Scikit-learn for machine learning.  "
                        "Python's syntax is clean and readable, making it beginner-friendly."
                    ),
                    "source_url": "https://example.com/python-docs"
                },
                {
                    "text": (
                        "The Python community is very active and supportive. "
                        "There are many tutorials, documentation, and forums available. "
                        "Python can integrate with C, C++, and Java for performance needs."
                    ),
                    "source_url": "https://example.com/community-guide"
                }
            ]
        }
    }

    print("Input Conversation:")
    for turn in conversation_json["conversation_turns"]:
        role = turn["role"]. upper()
        content = turn["message"][: 100] + "..." if len(turn["message"]) > 100 else turn["message"]
        print(f"{role}: {content}")
    print()

    print("Running evaluation...")

    pipeline = EvaluationPipeline()
    result = pipeline.evaluate_from_json(conversation_json, context_json)
    log_evaluation(result, source="demo")

    print_results(result)
    analyze_hallucinations(result)


def run_samples(generate_report: bool = False):
    """Run evaluation on the sample files from data folder."""
    print("LLM Evaluation Pipeline - Sample Files Mode")
    print()

    data_dir = Path("data")

    sample_pairs = [
        ("sample-chat-conversation-01.json", "sample_context_vectors-01.json"),
        ("sample-chat-conversation-02.json", "sample_context_vectors-02.json"),
    ]

    pipeline = EvaluationPipeline()
    results = []
    result_dicts = []

    for conv_file, ctx_file in sample_pairs: 
        conv_path = data_dir / conv_file
        ctx_path = data_dir / ctx_file

        if not conv_path.exists():
            print(f"Warning: {conv_path} not found, skipping...")
            continue
        if not ctx_path.exists():
            print(f"Warning: {ctx_path} not found, skipping...")
            continue

        print(f"\nEvaluating:  {conv_file}")
        print(f"Context:     {ctx_file}")

        conversation_json = load_json_file(str(conv_path))
        context_json = load_json_file(str(ctx_path))

        result = pipeline.evaluate_from_json(conversation_json, context_json)
        log_evaluation(result, source="samples")
        results.append(result)
        result_dicts.append(result. to_dict())

        print_results(result)
        analyze_hallucinations(result)

    # Print aggregate statistics
    if results:
        print("AGGREGATE STATISTICS")
        stats = pipeline.get_statistics()
        print(json.dumps(stats, indent=2))
        
        # Summary table
        print()
        print("SUMMARY TABLE")
        print(f"{'Sample':<10} {'Overall': >10} {'Relevance':>10} {'Halluc. ':>10} {'Complete':>10} {'Status':>10}")
        for i, r in enumerate(results, 1):
            status = "PASS" if r.passed else "FAIL"
            print(f"#{i:<9} {r.overall_score:>10.2%} {r.relevance.score:>10.2%} {r.hallucination.score:>10.2%} {r.completeness.score:>10.2%} {status:>10}")
        print()

    # Generate HTML report if requested
    if generate_report and result_dicts:
        print("Generating HTML report...")
        report_path = generate_html_report(result_dicts, "evaluation_report.html")
        print(f"Open {report_path} in your browser to view the visual report!")
        print()

    return results

def run_evaluation(conversation_path: str, context_path: str, output_path: str = None, generate_report: bool = False):
    """Run evaluation on provided JSON files."""
    print("Loading input files...")
    conversation_json = load_json_file(conversation_path)
    context_json = load_json_file(context_path)

    print("Initializing pipeline...")
    pipeline = EvaluationPipeline()

    print("Running evaluation...")
    result = pipeline.evaluate_from_json(conversation_json, context_json)
    log_evaluation(result, source="single")
    result_dict = result.to_dict()

    if output_path: 
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results saved to: {output_path}")

    print_results(result)
    analyze_hallucinations(result)

    if generate_report:
        report_path = generate_html_report([result_dict], "evaluation_report.html")
        print(f"Report generated: {report_path}")

    return result

def print_stats_summary(limit: int = 20):
    """Print high-level metrics from SQLite history."""
    stats = get_stats(limit=limit)

    if stats.get("count", 0) == 0:
        print("No evaluation history found.")
        return

    print(f"Last {stats['count']} Evaluations")
    print("------------------------------")
    print(f"Avg Overall Score:     {stats['avg_overall'] * 100:.1f}%")
    print(f"Avg Relevance:         {stats['avg_relevance'] * 100:.1f}%")
    print(f"Avg Hallucination:     {stats['avg_hallucination'] * 100:.1f}%")
    print(f"Avg Completeness:      {stats['avg_completeness'] * 100:.1f}%")
    print(f"P95 Latency:           {stats['p95_latency']:.2f} ms")

    failure = stats.get("most_common_failure")
    if failure:
        print(f"Most Frequent Failure: {failure.capitalize()}")
    else:
        print("Most Frequent Failure: None")


def run_stress_test(n: int):
    """Run N randomized evaluations to measure latency distribution and stability."""

    print(f"Stress Test ({n} runs)")
    print("----------------------")

    pipeline = EvaluationPipeline()

    # Load the same sample files you use in run_samples()
    data_dir = Path("data")
    sample_pairs = [
        ("sample-chat-conversation-01.json", "sample_context_vectors-01.json"),
        ("sample-chat-conversation-02.json", "sample_context_vectors-02.json"),
    ]

    pairs = []
    for conv, ctx in sample_pairs:
        cpath = data_dir / conv
        xpath = data_dir / ctx
        if cpath.exists() and xpath.exists():
            pairs.append((conv, ctx))

    if not pairs:
        print("No sample pairs found; cannot run stress test.")
        return

    latencies = []
    rel_scores = []
    hall_scores = []
    comp_scores = []

    for i in range(n):
        conv_file, ctx_file = random.choice(pairs)

        conversation_json = load_json_file(data_dir / conv_file)
        context_json = load_json_file(data_dir / ctx_file)

        start = time.perf_counter()
        result = pipeline.evaluate_from_json(conversation_json, context_json)
        elapsed = (time.perf_counter() - start) * 1000

        # Prefer pipeline latency if filled
        elapsed = result.latency.total_ms

        latencies.append(elapsed)
        rel_scores.append(result.relevance.score)
        hall_scores.append(result.hallucination.score)
        comp_scores.append(result.completeness.score)

        # Persist run
        log_evaluation(result, source="stress")

    # Compute metrics
    def pct(values, p):
        k = int(round((p * (len(values) - 1)) / 100))
        return sorted(values)[k]

    mean_lat = statistics.mean(latencies)
    p50 = pct(latencies, 50)
    p90 = pct(latencies, 90)
    p95 = pct(latencies, 95)
    p99 = pct(latencies, 99)

    # Stability (standard deviation)
    variations = {
        "relevance": statistics.pstdev(rel_scores),
        "hallucination": statistics.pstdev(hall_scores),
        "completeness": statistics.pstdev(comp_scores),
    }
    unstable = max(variations, key=variations.get)

    print(f"Mean latency: {mean_lat:.2f} ms")
    print(f"P50 latency:  {p50:.2f} ms")
    print(f"P90 latency:  {p90:.2f} ms")
    print(f"P95 latency:  {p95:.2f} ms")
    print(f"P99 latency:  {p99:.2f} ms")
    print()
    print("Metric variation (std dev):")
    print(f"  Relevance:     {variations['relevance']:.4f}")
    print(f"  Hallucination: {variations['hallucination']:.4f}")
    print(f"  Completeness:  {variations['completeness']:.4f}")
    print()
    print(f"Most unstable metric: {unstable}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Response Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples: 
    python main.py --demo
    python main.py --samples
    python main.py --samples --report
    python main.py -c conversation.json -x context.json
    python main.py -c conversation.json -x context. json -o results.json --report
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with sample data"
    )

    parser.add_argument(
        "--samples",
        action="store_true",
        help="Run evaluation on sample files in data/ folder"
    )

    parser.add_argument(
        "-c", "--conversation",
        type=str,
        help="Path to conversation JSON file"
    )

    parser.add_argument(
        "-x", "--context",
        type=str,
        help="Path to context JSON file"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to save output JSON (optional)"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML visual report"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show aggregated statistics from SQLite history"
    )

    parser.add_argument(
        "--stress",
        type=int,
        metavar="N",
        help="Run stress-test with N randomized evaluations"
    )

    args = parser.parse_args()

    if args.stats:
        print_stats_summary()
        sys.exit(0)
    
    if args.stress:
        run_stress_test(args.stress)
        sys.exit(0)

    if args.demo:
        run_demo()
    elif args.samples:
        run_samples(generate_report=args.report)
    elif args.conversation and args.context:
        run_evaluation(
            args.conversation,
            args.context,
            args.output,
            generate_report=args.report
        )
    else:
        parser.print_help()
        print()
        print("Error: Provide --demo, --samples, OR both --conversation and --context")
        sys.exit(1)


if __name__ == "__main__":
    main()