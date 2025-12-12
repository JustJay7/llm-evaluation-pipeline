#!/usr/bin/env python3
"""Benchmark the evaluation pipeline performance."""

import time
import statistics
from evaluator.pipeline import EvaluationPipeline

def run_benchmark(iterations:  int = 10):
    """Run performance benchmark."""
    
    print("LLM Evaluation Pipeline Benchmark")
    
    # Sample data
    conversation = {
        "conversation_turns": [
            {"role": "User", "message": "What is IVF treatment?"},
            {"role": "AI/Chatbot", "message": "IVF is in vitro fertilization, a process where eggs are fertilized outside the body. "}
        ]
    }
    context = {
        "data": {
            "vector_data": [
                {"text": "IVF (In Vitro Fertilization) is an assisted reproductive technology where eggs are retrieved and fertilized with sperm in a laboratory."}
            ]
        }
    }
    
    pipeline = EvaluationPipeline()
    
    # Warm-up run
    print("\nWarming up (loading models)...")
    pipeline.evaluate_from_json(conversation, context)
    
    # Benchmark runs
    print(f"\nRunning {iterations} iterations...")
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        result = pipeline.evaluate_from_json(conversation, context)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.1f}ms")
    
    # Statistics
    print("\nRESULTS")
    print("\n")
    print(f"Mean:    {statistics.mean(times):.1f}ms")
    print(f"Median:  {statistics.median(times):.1f}ms")
    print(f"Std Dev: {statistics.stdev(times):.1f}ms")
    print(f"Min:     {min(times):.1f}ms")
    print(f"Max:     {max(times):.1f}ms")
    print(f"p95:     {sorted(times)[int(len(times)*0.95)]:.1f}ms")
    
    # Throughput
    avg_time = statistics.mean(times)
    throughput = 1000 / avg_time * 60  # per minute
    daily = throughput * 60 * 24
    
    print("\nTHROUGHPUT (single instance)")
    print(f"Per minute: {throughput:.0f} evaluations")
    print(f"Per day:    {daily: ,.0f} evaluations")
    print(f"For 1M/day: {1_000_000/daily:.0f} instances needed")

if __name__ == "__main__":
    run_benchmark()