# LLM Response Evaluation Pipeline

A production-grade evaluation pipeline that measures the reliability of LLM/chatbot responses using semantic relevance scoring, hallucination detection, aspect-based completeness analysis, and real-time latency/cost instrumentation. The system also includes historical performance tracking and stress-test benchmarking, enabling deep insight into model behavior, quality drift, and operational performance.

Built for the BeyondChats AI Internship Assignment.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Scalability & Performance](#scalability--performance)
- [Benchmark Results](#benchmark-results)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output Format](#inputoutput-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Running Tests](#running-tests)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Features

- **Relevance Evaluation**: Measures semantic similarity between response, query, and retrieved context
- **Hallucination Detection**: Identifies claims in responses not supported by the provided context
- **Completeness Checking (Semantic Coverage)**: Extracts key aspects from the user query using regex-based aspect mining, computes semantic similarity via sentence-transformers, combines it with keyword overlap, and determines whether each aspect was fully addressed.
- **Confidence Scoring**: Indicates how confident the system is in its evaluation
- **Latency Tracking**: Detailed timing breakdown for each evaluation stage
- **Cost Estimation**: Token counting and cost calculation for LLM operations
- **HTML Reports**: Beautiful visual dashboards for evaluation results
- **Benchmarking**: Performance testing for scalability verification
- **Flexible Input**: Supports BeyondChats JSON format and generic conversation/context formats
- **Real-time Ready**: Optimized for low-latency evaluation in production environments

---

### Advanced Features (New Additions)

**Explainability Layer**  
Generates human-readable explanations that describe *why* a response passed or failed.  
Includes:
- Evaluation reasoning summary  
- Metric-by-metric breakdown  
- Unsupported claim reporting  
- Confidence interpretation  

**Auto-Suggestions for Improvement**  
Provides actionable recommendations such as:
- Improving RAG retrieval  
- Adding more context vectors  
- Reducing unsupported claims  

**Enhanced CLI Output**  
Adds structured sections to command-line output:
- Explanation block  
- Detailed diagnostic breakdown  
- Suggested next steps  

**Historical Evaluation Memory (SQLite)**  
Automatically stores every evaluation run (demo, samples, custom files, stress tests) in a lightweight SQLite database.

Each entry logs:
- Overall score  
- Relevance, hallucination, completeness scores  
- Latency in milliseconds  
- Pass or fail status  
- Timestamp  
- Failure reason (automatically inferred)

View aggregated stats with:

```bash
python main.py --stats
```

| **Metric** | **Last 20 Evaluations** |
|:-------------|:------------|
| Avg Overall Score | 72.4% |
| Avg Relevance | 81.2% |
| Avg Hallucination | 11.2% |
| Avg Completeness | 91.5% |
| P95 Latency | 47.9 ms |
| Most Frequent Failure | Hallucination |

**Stress-Test Mode (Performance & Robustness Testing)**
Evaluate how the pipeline behaves under repeated randomized evaluations, helping identify:

- Latency distribution (p50, p90, p95, p99)
- Scoring variation across metrics
- Most unstable metric (e.g., hallucination vs relevance)
- Potential bottlenecks in embedding or NLI computation

Run stress-test:
```bash
python main.py --stress 50
```

Stress Test (50 runs)
- Mean latency: 44.1 ms
- P50 latency: 42.8 ms
- P90 latency: 47.2 ms
- P95 latency: 49.9 ms
- P99 latency: 54.3 ms

Metric variation (std dev):
| Metric | Std Dev |
|---------|---------|
| Relevance | 0.0128 |
| Hallucination | 0.0762 |
| Completeness | 0.4000 |

Most unstable metric:
Completeness

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              ┌──────────────┐    ┌──────────────┐               │
│              │ Conversation │    │   Context    │               │
│              │     JSON     │    │    JSON      │               │
│              └──────┬───────┘    └──────┬───────┘               │
│                     │                   │                       │
│                     ▼                   ▼                       │
│              ┌─────────────────────────────────────┐            │
│              │            INPUT PARSER             │            │
│              │  • Parse conversation_turns         │            │
│              │  • Extract vector_data              │            │
│              │  • Normalize roles & content        │            │
│              └─────────────────┬───────────────────┘            │
│                                │                                │
│                                ▼                                │
│              ┌─────────────────────────────────────┐            │
│              │          EVALUATION INPUT           │            │
│              │  • Last user message (query)        │            │
│              │  • Last AI response                 │            │
│              │  • Combined context                 │            │
│              └─────────────────┬───────────────────┘            │
│                                │                                │
│        ┌─────────────┬─────────┼─┬──────────────────┐           │
│        ▼             ▼           ▼                  ▼           │
│   ┌──────────┐ ┌──────────┐ ┌────────────┐   ┌──────────────┐   │
│   │RELEVANCE │ │COMPLETE  │ │HALLUCINATE │   │   LATENCY    │   │
│   │EVALUATOR │ │EVALUATOR │ │  DETECTOR  │   │   TRACKER    │   │
│   ├──────────┤ ├──────────┤ ├────────────┤   ├──────────────┤   │
│   │Sentence  │ │Aspect    │ │Claim       │   │Per-stage     │   │
│   │Embeddings│ │Extraction│ │Extraction  │   │timing        │   │
│   │Cosine    │ │Coverage  │ │Context     │   │              │   │
│   │Similarity│ │Check     │ │Verification│   │              │   │
│   └────┬─────┘ └─────┬────┘ └─────┬──────┘   └──────┬───────┘   │
│        │             │            │                 │           │
│        └─────────────┴────────────┴─────────────────┘           │
│                               │                                 │
│                               ▼                                 │
│                ┌─────────────────────────────┐                  │
│                │      RESULT AGGREGATOR      │                  │
│                │  • Overall score            │                  │
│                │  • Confidence score         │                  │
│                │  • Pass/Fail determination  │                  │
│                │  • Detailed breakdowns      │                  │
│                │  • Cost metrics             │                  │
│                └──────────────┬──────────────┘                  │
│                               │                                 │
│                               ▼                                 │
│                ┌─────────────────────────────┐                  │
│                │  JSON OUTPUT + HTML REPORT  │                  │
│                └─────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Responsibility | Key Technology |
|-----------|---------------|----------------|
| **Input Parser** | Parse BeyondChats JSON format, normalize data | Python dict parsing |
| **Relevance Evaluator** | Semantic similarity scoring | sentence-transformers |
| **Hallucination Detector** | Claim extraction & verification | sentence-transformers |
| **Completeness Evaluator** | Query coverage analysis | Regex + embeddings |
| **Confidence Scorer** | Evaluation reliability assessment | Statistical analysis |
| **Latency Tracker** | Per-stage timing | Python time. perf_counter |
| **Cost Tracker** | Token counting & cost estimation | tiktoken |
| **Report Generator** | Visual HTML dashboards | HTML/CSS |

---

## Design Decisions

### Why This Approach?

#### 1. **Embedding-Based Similarity over LLM-as-Judge**

**Decision**: Use sentence-transformers (`all-MiniLM-L6-v2`) for semantic similarity instead of calling GPT/Claude to judge responses.

**Why**:
- **10-100x faster**: Embeddings compute in ~20ms vs 1-2s for LLM API calls
- **Near-zero marginal cost**: No per-evaluation API fees
- **No external dependencies**: Works offline, no API keys needed
- **Deterministic**: Same input always produces same output (important for testing)

**Trade-off**: Slightly less nuanced than LLM-as-Judge, but acceptable for real-time evaluation.

#### 2. **Claim-Based Hallucination Detection**

**Decision**: Extract individual claims from responses and verify each against context.

**Why**: 
- **Granular detection**: Identifies exactly which claims are unsupported
- **Actionable feedback**: "Claim X is not supported" vs "Response has hallucinations"
- **Debuggable**: Easy to trace why something was flagged

**Alternative considered**: Full NLI (Natural Language Inference) between response and context.  Rejected because it's computationally expensive and less interpretable.

#### 3. **Lazy Loading for Models**

**Decision**: Load ML models only when first needed, not at initialization.

```python
@property
def model(self) -> SentenceTransformer:
    if self._model is None:
        self._model = SentenceTransformer(self.config.embedding_model)
    return self._model
```

**Why**:
- **Faster startup**: Pipeline initializes instantly
- **Memory efficient**: Models loaded only if needed
- **Reusable**: Once loaded, cached for subsequent evaluations

#### 4. **Modular Architecture**

**Decision**: Separate evaluators for each metric, orchestrated by a central pipeline.

**Why**:
- **Testable**: Each component can be unit tested independently
- **Maintainable**: Easy to modify one evaluator without affecting others
- **Extensible**: Add new evaluation metrics by creating new evaluators
- **Configurable**: Enable/disable specific evaluations as needed

#### 5. **Weighted Overall Score**

**Formula**: `Relevance (35%) + (1 - Hallucination) (40%) + Completeness (25%)`

**Why this weighting**:
- **Hallucination (40%)**: Most critical - factual errors destroy trust
- **Relevance (35%)**: Important - irrelevant answers waste user time
- **Completeness (25%)**: Valuable but partial answers can still be useful


#### 6. **Confidence Scoring**

**Decision**: Add a confidence score to indicate evaluation reliability.

**Why**: 
- **Transparency**: Users know when to trust results
- **Edge case handling**: Borderline scores get flagged
- **Better decisions**: Low confidence = human review needed

### 7. **Historical Memory for Long-Term Observability (NEW)**

**Decision**: Persist every evaluation into SQLite.

**Why**:
- Enables longitudinal analysis
- Detects model drift or retrieval degradation
- Supports aggregate statistics and monitoring
- Zero external dependencies

This mirrors how production-grade evaluation systems store scoring metadata.

### 8. **Stress-Testing for Reliability (NEW)**

**Decision**: Add a CLI-driven stress test that evaluates many randomized prompts and computes latency distribution + metric stability.

**Why**:
- Reveals worst-case latency spikes
- Highlights unstable metrics
- Provides real performance guarantees
- Demonstrates awareness of scaling concerns

This is a feature commonly found in serious ML evaluation workflows.

---

## Scalability & Performance

### How We Handle Millions of Daily Conversations

#### 1. **Lightweight Models**

| Model | Size | Inference Time | Memory |
|-------|------|----------------|--------|
| all-MiniLM-L6-v2 | 80MB | ~20ms | ~200MB |

We use the **smallest effective model** for embeddings.  The MiniLM model achieves 90%+ of the quality of larger models at 1/20th the latency.

#### 2. **Embedding Caching**

```python
if self. config.cache_embeddings:
    cache_key = hash(text[: 500])
    if cache_key in self._embedding_cache:
        return self._embedding_cache[cache_key]
```

**Impact**: Repeated text (common greetings, standard responses) are computed once. 

#### 3. **Batch Processing Ready**

```python
def evaluate_batch(self, inputs: List[EvaluationInput]) -> List[EvaluationResult]:
    # Process multiple evaluations efficiently
```

For high throughput, batch multiple conversations together to maximize GPU utilization.

#### 4. **Lazy Evaluation**

Models are loaded on-demand, not at startup. In a microservice architecture: 
- Cold start:  ~3s (model loading)
- Warm evaluation: ~45ms

#### 5. **Cost Optimization Strategies**

| Strategy | Implementation | Savings |
|----------|---------------|---------|
| **Local Models** | No API calls for evaluation | ~$0.01-0.05 per evaluation |
| **Token Counting** | tiktoken (local) vs API tokenizer | 100% |
| **Caching** | Embedding cache for repeated text | 20-40% |
| **Early Exit** | Skip completeness if hallucination=100% | Variable |

#### 6. **Production Deployment Recommendations**

```yaml
# Kubernetes deployment for scale
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 10  # Scale based on load
  template:
    spec:
      containers:
      - name: evaluator
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

**Recommended Infrastructure**:
- **Load Balancer**: Distribute across multiple evaluator instances
- **Redis Cache**: Shared embedding cache across instances
- **GPU Nodes**: For high-throughput batch processing
- **Async Queue**: Kafka/RabbitMQ for non-blocking evaluation

---

## Benchmark Results

Actual performance measurements on MacBook Air (M1):

### Latency

| Metric | Value |
|--------|-------|
| **Mean** | 44.7ms |
| **Median** | 44.7ms |
| **Std Dev** | 1.4ms |
| **Min** | 41.7ms |
| **Max** | 46.8ms |
| **p95** | 46.8ms |

### Throughput (Single Instance)

| Metric | Value |
|--------|-------|
| **Per minute** | 1,341 evaluations |
| **Per day** | 1,931,374 evaluations |
| **For 1M/day** | 1 instance needed |

### Latency Breakdown By Stage (Actual Measurements)

From our test runs: 

| Stage | Time | % of Total |
|-------|------|------------|
| Relevance | ~10ms | 22% |
| Hallucination | ~25ms | 56% |
| Completeness | ~10ms | 22% |
| **Total** | ~45ms | 100% |

> **Note**: Benchmark uses simple test data. Real-world performance with larger contexts may vary.  Run `python benchmark.py` to test on your hardware.

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Local Setup

1. **Clone the repository**: 
```bash
git clone https://github.com/JustJay7/llm-evaluation-pipeline.git
cd llm-evaluation-pipeline
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python main.py --demo
```

---

## Usage

### Quick Demo
Run the built-in demo with sample data:
```bash
python main.py --demo
```

### Evaluate Sample Files
Run evaluation on the BeyondChats sample files in the `data/` folder:
```bash
python main.py --samples
```

### Generate HTML Report
Run evaluation and generate a visual HTML report:
```bash
python main.py --samples --report
```
Then open `evaluation_report.html` in your browser.

### Run Benchmark
Test performance on your machine:
```bash
python benchmark.py
```

### View Historical Evaluation Statistics
```bash
python main.py --stats
```

Displays aggregated metrics such as:
- Average overall score
- Average relevance / hallucination / completeness
- Latency percentiles
- Most frequent failure reason

Useful for tracking model drift and long-term behavior.

### Run Stress Test (Performance Profiling)
```bash
python main.py --stress 50
```

Runs 50 randomized evaluations and prints:
- Mean latency
- p50 / p90 / p95 / p99 latencies
- Score variance across metrics
- Most unstable metric

Ideal for load testing, micro-batching experiments, and profiling evaluation bottlenecks.

### Evaluate Custom Files
Provide your own conversation and context JSON files:
```bash
python main.py -c path/to/conversation.json -x path/to/context.json
```

Save results to a file:
```bash
python main.py -c conversation.json -x context.json -o results.json --report
```

### Programmatic Usage

```python
from evaluator. pipeline import EvaluationPipeline
from evaluator.confidence import calculate_confidence

# Initialize pipeline
pipeline = EvaluationPipeline()

# Prepare input data (BeyondChats format)
conversation_json = {
    "conversation_turns": [
        {"role": "User", "message": "What is IVF?"},
        {"role":  "AI/Chatbot", "message": "IVF is in vitro fertilization... "}
    ]
}

context_json = {
    "data":  {
        "vector_data": [
            {"text": "IVF (In Vitro Fertilization) is a medical procedure..."}
        ]
    }
}

# Run evaluation
result = pipeline.evaluate_from_json(conversation_json, context_json)

# Access results
print(f"Overall Score: {result.overall_score}")
print(f"Passed: {result.passed}")
print(f"Confidence:  {calculate_confidence(result):. 1%}")
print(f"Hallucination Score: {result.hallucination. score}")
print(f"Unsupported Claims: {result.hallucination.unsupported_claims}")

# Export to JSON
result_dict = result.to_dict()

# Generate HTML report
from evaluator.report import generate_html_report
generate_html_report([result_dict], "my_report.html")
```

---

## Input/Output Format

### Input:  Conversation JSON (BeyondChats Format)
```json
{
    "chat_id": 12345,
    "conversation_turns": [
        {
            "turn":  1,
            "role": "User",
            "message": "What is the cost of IVF? ",
            "created_at": "2025-01-01T10:00:00.000000Z"
        },
        {
            "turn": 2,
            "role": "AI/Chatbot",
            "message": "The cost of IVF at our clinic is.. .",
            "created_at":  "2025-01-01T10:00:05.000000Z"
        }
    ]
}
```

### Input: Context JSON (BeyondChats Format)
```json
{
    "status": "success",
    "data": {
        "vector_data": [
            {
                "id": 123,
                "text": "IVF treatment costs approximately.. .",
                "source_url": "https://example.com/ivf-costs",
                "tokens": 150
            }
        ]
    }
}
```

### Output: Evaluation Result
```json
{
    "overall_score": 0.85,
    "passed": true,
    "timestamp": "2025-01-01T12:00:00.000000",
    "relevance":  {
        "score": 0.82,
        "is_relevant": true,
        "query_response_similarity": 0.75,
        "context_response_similarity": 0.88
    },
    "hallucination": {
        "score":  0.1,
        "is_hallucinated": false,
        "unsupported_claims": [],
        "supported_claims_count": 5
    },
    "completeness": {
        "score": 1.0,
        "is_complete": true,
        "covered_aspects": ["cost of ivf treatment"],
        "missing_aspects": []
    },
    "latency": {
        "total_ms": 45.5,
        "relevance_ms": 10.2,
        "hallucination_ms": 25.1,
        "completeness_ms": 10.2
    },
    "cost":  {
        "input_tokens": 500,
        "output_tokens":  100,
        "total_tokens":  600,
        "estimated_cost_usd": 0.001
    }
}
```

### Sample Evaluation Results

From running on BeyondChats sample data:

| Sample | Overall | Relevance | Hallucination | Completeness | Status |
|--------|---------|-----------|---------------|--------------|--------|
| #1 (IVF Cost) | 69.3% | 50. 3% | 33.3% | 100.0% | FAIL |
| #2 (Donor Egg) | 53.0% | 53.5% | 14.3% | 0.0% | FAIL |

**Key Findings**:
- Sample #1: Caught unsupported marketing claim (consultation booking link)
- Sample #2: Detected incomplete response to user's question about donor eggs

---

## Evaluation Metrics

### Relevance Score (0-1)
Measures how well the response relates to both the user query and retrieved context. 

- **Method**: Sentence embeddings + cosine similarity
- **Formula**: `0.4 × query_similarity + 0.6 × context_similarity`
- **Threshold**: 0.6 (configurable)

### Hallucination Score (0-1)
Detects unsupported claims by checking each statement against the context.

- **Method**: Claim extraction → semantic matching against context
- **0.0** = No hallucination (all claims supported)
- **1.0** = Complete hallucination (no claims supported)
- **Threshold**: 0.5 (above = hallucinated)

### Completeness Score (0-1)
Evaluates whether key aspects of the question are addressed.

- **Method**: Query aspect extraction → coverage verification
- **1.0** = All aspects covered
- **0.0** = No aspects covered

### Confidence Score (0-1)
Indicates how reliable the evaluation results are.

- **Method**: Statistical analysis of score distributions
- **Factors that reduce confidence**:
  - Borderline scores (near thresholds)
  - Few claims to verify
  - Ambiguous queries
- **Usage**: Low confidence → consider human review

### Overall Score
```
overall = 0.35 × relevance + 0.40 × (1 - hallucination) + 0.25 × completeness
```

### Pass/Fail Criteria
- Overall score ≥ 0.7
- No critical hallucinations (hallucination score < 0.5)
- Minimum relevance (relevance score ≥ 0.6)

---

## Running Tests

```bash
# Run all tests
python -m unittest tests.test_pipeline -v

# Run specific test class
python -m unittest tests.test_pipeline.TestHallucinationDetector -v
```

---

## Results

Below are real outputs from the pipeline so reviewers can understand the system’s behavior without running the code.  
All screenshots are located in the `results/` folder for easy reference.

### 1. Demo Evaluation  
Command:
```bash
python main.py --demo
```
This runs a built-in conversation and shows full evaluation including relevance, hallucination, completeness, and explanations.

**Screenshot:**  
![Demo Output](results/demo-output-1.png)
![Demo Output](results/demo-output-2.png)

### 2. Sample Evaluation with HTML Report  
Command:
```bash
python main.py --samples --report
open evaluation_report.html
```
Runs evaluations on curated sample conversations stored in `data/` and prints results + aggregate stats and generates a rich HTML dashboard with score breakdowns, completeness explanations, hallucination lists, and pass/fail diagnostics.

**HTML Report Preview:**  
![Sample Output](results/sample-output-1.png)
![Sample Output](results/sample-output-2.png)
![Sample Output](results/sample-output-3.png)
![Sample Output](results/sample-output-4.png)
![Sample Output](results/sample-output-5.png)
![Sample Output](results/sample-output-6.png)
![Sample Output](results/sample-output-7.png)
![Report Preview](results/report-screenshot-1.png)
![Report Preview](results/report-screenshot-2.png)

### 3. Stress Test Mode (50 Evaluations)  
Command:
```bash
python main.py --stress 50
```
Runs multiple randomized evaluations to measure latency distribution and metric stability.  
Outputs mean latency, p50–p99, and variation in relevance/hallucination/completeness.

**Screenshot:**  
![Stress Test](results/stress-test-output.png)

### 4. Historical Analytics (via SQLite Memory)  
Command:
```bash
python main.py --stats
```
Prints aggregated insights from the last 20 evaluations:
- Avg overall score  
- Avg hallucination  
- Avg completeness  
- P95 latency  
- Most frequent failure reason  

This mimics real LLM evaluation telemetry dashboards.

**Screenshot:**  
![Stats Output](results/stats-output.png)

---

## Project Structure

```
llm-evaluation-pipeline/
├── evaluator/
│   ├── __init__.py          # Package exports
│   ├── completeness.py      # Semantic completeness evaluator (NEW)
│   ├── confidence.py        # Confidence scoring
│   ├── config.py            # Configuration management
│   ├── cost_tracker.py      # Cost and latency tracking
│   ├── explainer.py         # Generate human-readable explanations
│   ├── hallucination.py     # Hallucination detection
│   ├── history.py           # SQLite historical evaluator memory (NEW)
│   ├── models.py            # Data models (dataclasses)
│   ├── pipeline.py          # Main orchestration pipeline
│   ├── relevance.py         # Relevance & completeness evaluators
│   └── report.py            # HTML report generation
│   ├── utils.py             # Text processing, timing utilities
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Comprehensive unit tests
├── data/
│   ├── sample-chat-conversation-01.json
│   ├── sample-chat-conversation-02.json
│   ├── sample_context_vectors-01.json
│   └── sample_context_vectors-02.json
├── results/
│   ├── demo-output-1.png         # Dashboard screenshot
│   ├── demo-output-2.png         # Dashboard screenshot
│   ├── report-screenshot-1.png   # Report screenshot
│   ├── report-screenshot-2.png   # Report screenshot
│   ├── sample-output-1.png       # Sample Output screenshot
│   ├── sample-output-2.png       # Sample Output screenshot
│   ├── sample-output-3.png       # Sample Output screenshot
│   ├── sample-output-4.png       # Sample Output screenshot
│   ├── sample-output-5.png       # Sample Output screenshot
│   ├── sample-output-6.png       # Sample Output screenshot
│   ├── sample-output-7.png       # Sample Output screenshot
│   ├── stats-output.png          # Stats Output screenshot
│   └── stress-test-output-2.png  # Stress Test Output screenshot
├── main.py                  # CLI entry point
├── benchmark.py             # Performance benchmarking
├── evaluation_history.db    # Database to store past evaluations
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Configuration

Environment variables for customization:

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EVAL_RELEVANCE_THRESHOLD` | `0.6` | Minimum relevance score |
| `EVAL_HALLUCINATION_THRESHOLD` | `0.5` | Maximum hallucination score |
| `EVAL_INPUT_TOKEN_COST` | `0.0015` | Cost per 1K input tokens |
| `EVAL_OUTPUT_TOKEN_COST` | `0.002` | Cost per 1K output tokens |

---

## Future Improvements

1. **GPU Acceleration**: Add CUDA support for batch processing
2. **Async Evaluation**: Non-blocking evaluation for real-time systems
3. **Custom Thresholds**: Per-domain threshold configuration
4. **Explanation Generation**: Human-readable explanations for failures
5. **Dashboard**: Real-time monitoring visualization
6. **A/B Testing**: Compare different LLM configurations
7. **Multi-language Support**: Evaluate responses in different languages

---

## Technologies Used

- **sentence-transformers**: Semantic similarity with pre-trained models
- **scikit-learn**: Cosine similarity calculations
- **tiktoken**: Token counting for cost estimation
- **NumPy**: Numerical operations

---

## Author

**JustJay7** - BeyondChats AI Internship Assignment

---

## License

MIT License