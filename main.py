#!/usr/bin/env python3
"""
LLM Response Evaluation Pipeline - Main Entry Point. 

Usage:
    python main.py --conversation <path> --context <path>
    python main.py --demo
    python main.py --samples
"""

import argparse
import json
import sys
from pathlib import Path

from evaluator.pipeline import EvaluationPipeline
from evaluator.config import EvaluatorConfig


def load_json_file(filepath: str) -> dict:
    """
    Load JSON from file path.

    Args:
        filepath: Path to JSON file.

    Returns:
        Parsed JSON as dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json. load(f)
    except json.JSONDecodeError as e:
        print(f"Error:  Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def print_results(result):
    """Print evaluation results in a formatted way."""
    result_dict = result.to_dict()
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(result_dict, indent=2))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Score:    {result.overall_score:.4f}")
    print(f"Passed:            {result.passed}")
    print(f"Relevance:         {result.relevance.score:.4f} ({'PASS' if result.relevance.is_relevant else 'FAIL'})")
    print(f"Hallucination:    {result.hallucination.score:.4f} ({'FAIL' if result.hallucination.is_hallucinated else 'PASS'})")
    print(f"Completeness:     {result.completeness.score:.4f} ({'PASS' if result.completeness.is_complete else 'FAIL'})")
    print(f"Total Latency:    {result.latency.total_ms:.2f} ms")
    print(f"Estimated Cost:   ${result.cost.estimated_cost_usd:.6f}")
    print()


def run_demo():
    """Run a demonstration with sample data."""
    print("=" * 60)
    print("LLM Evaluation Pipeline - Demo Mode")
    print("=" * 60)
    print()

    # Sample conversation JSON
    conversation_json = {
        "conversation": [
            {
                "role": "user",
                "content": "What are the benefits of using Python for data science?"
            },
            {
                "role": "ai",
                "content": (
                    "Python is excellent for data science due to several reasons:  "
                    "1) It has powerful libraries like NumPy, Pandas, and Scikit-learn.  "
                    "2) It has a simple and readable syntax that makes it easy to learn. "
                    "3) It has strong community support with extensive documentation. "
                    "4) It integrates well with other languages and tools."
                )
            }
        ]
    }

    # Sample context JSON (simulating vector DB retrieval)
    context_json = {
        "context": [
            {
                "content": (
                    "Python is a popular programming language for data science.  "
                    "It offers libraries such as NumPy for numerical computing, "
                    "Pandas for data manipulation, and Scikit-learn for machine learning.  "
                    "Python's syntax is clean and readable, making it beginner-friendly."
                ),
                "score": 0.92,
                "source": "python_docs"
            },
            {
                "content": (
                    "The Python community is very active and supportive. "
                    "There are many tutorials, documentation, and forums available. "
                    "Python can integrate with C, C++, and Java for performance needs."
                ),
                "score": 0.87,
                "source": "community_guide"
            }
        ]
    }

    print("Input Conversation:")
    print("-" * 40)
    for msg in conversation_json["conversation"]: 
        role = msg["role"]. upper()
        content = msg["content"][: 100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"{role}: {content}")
    print()

    print("Running evaluation...")
    print("-" * 40)

    pipeline = EvaluationPipeline()
    result = pipeline.evaluate_from_json(conversation_json, context_json)

    print_results(result)


def run_samples():
    """Run evaluation on the sample files from data folder."""
    print("=" * 60)
    print("LLM Evaluation Pipeline - Sample Files Mode")
    print("=" * 60)
    print()

    data_dir = Path("data")

    # Define sample file pairs
    sample_pairs = [
        ("sample-chat-conversation-01.json", "sample_context_vectors-01.json"),
        ("sample-chat-conversation-02.json", "sample_context_vectors-02.json"),
    ]

    pipeline = EvaluationPipeline()
    results = []

    for conv_file, ctx_file in sample_pairs: 
        conv_path = data_dir / conv_file
        ctx_path = data_dir / ctx_file

        if not conv_path.exists():
            print(f"Warning: {conv_path} not found, skipping...")
            continue
        if not ctx_path.exists():
            print(f"Warning: {ctx_path} not found, skipping...")
            continue

        print(f"\nEvaluating:  {conv_file} + {ctx_file}")
        print("-" * 40)

        conversation_json = load_json_file(str(conv_path))
        context_json = load_json_file(str(ctx_path))

        result = pipeline.evaluate_from_json(conversation_json, context_json)
        results.append({
            "conversation_file": conv_file,
            "context_file": ctx_file,
            "result": result. to_dict()
        })

        print_results(result)

    # Print aggregate statistics
    if results:
        print("=" * 60)
        print("AGGREGATE STATISTICS")
        print("=" * 60)
        stats = pipeline.get_statistics()
        print(json.dumps(stats, indent=2))

    return results


def run_evaluation(conversation_path:  str, context_path: str, output_path: str = None):
    """
    Run evaluation on provided JSON files.

    Args:
        conversation_path: Path to conversation JSON. 
        context_path: Path to context JSON.
        output_path: Optional path to save results.
    """
    print("Loading input files...")
    conversation_json = load_json_file(conversation_path)
    context_json = load_json_file(context_path)

    print("Initializing pipeline...")
    pipeline = EvaluationPipeline()

    print("Running evaluation...")
    result = pipeline.evaluate_from_json(conversation_json, context_json)

    result_dict = result.to_dict()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results saved to: {output_path}")

    print_results(result)

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Response Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples: 
    python main.py --demo
    python main.py --samples
    python main.py -c conversation. json -x context.json
    python main.py -c conversation.json -x context. json -o results.json
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

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.samples:
        run_samples()
    elif args.conversation and args.context:
        run_evaluation(args.conversation, args.context, args.output)
    else:
        parser.print_help()
        print()
        print("Error:  Provide --demo, --samples, OR both --conversation and --context")
        sys.exit(1)


if __name__ == "__main__": 
    main()