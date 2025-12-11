# LLM Response Evaluation Pipeline

A Python pipeline for evaluating LLM/chatbot responses in RAG (Retrieval-Augmented Generation) systems. Evaluates responses for **relevance**, **hallucination**, and **completeness** while tracking **latency** and **cost** metrics in real-time.

Built for the BeyondChats AI Internship Assignment.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Scalability & Performance](#scalability--performance)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output Format](#inputoutput-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)

---

## Features

- **Relevance Evaluation**: Measures semantic similarity between response, query, and retrieved context
- **Hallucination Detection**: Identifies claims in responses not supported by the provided context
- **Completeness Checking**: Verifies if the response adequately addresses the user's question
- **Latency Tracking**: Detailed timing breakdown for each evaluation stage
- **Cost Estimation**: Token counting and cost calculation for LLM operations
- **Flexible Input**: Supports BeyondChats JSON format and generic conversation/context formats
- **Real-time Ready**: Optimized for low-latency evaluation in production environments

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
│   │EVALUATOR │ │EVALUATOR │ │ DETECTOR   │   │   TRACKER    │   │
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
│                │  • Pass/Fail determination  │                  │
│                │  • Detailed breakdowns      │                  │
│                │  • Cost metrics             │                  │
│                └──────────────┬──────────────┘                  │
│                               │                                 │
│                               ▼                                 │
│                ┌─────────────────────────────┐                  │
│                │         JSON OUTPUT         │                  │
│                └─────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Responsibility | Key Technology |
|-----------|---------------|----------------|
| **Input Parser** | Parse BeyondChats JSON format, normalize data | Python dict parsing |
| **Relevance Evaluator** | Semantic similarity scoring | sentence-transformers |
| **Hallucination Detector** | Claim extraction & verification | sentence-transformers + NLI |
| **Completeness Evaluator** | Query coverage analysis | Regex + embeddings |
| **Latency Tracker** | Per-stage timing | Python time. perf_counter |
| **Cost Tracker** | Token counting & cost estimation | tiktoken |

---

## Design Decisions

### Why This Approach?

#### 1. **Embedding-Based Similarity over LLM-as-Judge**

**Decision**: Use sentence-transformers (`all-MiniLM-L6-v2`) for semantic similarity instead of calling GPT/Claude to judge responses.

**Why**:
- **10-100x faster**:  Embeddings compute in ~50ms vs 1-2s for LLM API calls
- **Near-zero marginal cost**: No per-evaluation API fees
- **No external dependencies**: Works offline, no API keys needed
- **Deterministic**:  Same input always produces same output (important for testing)

**Trade-off**: Slightly less nuanced than LLM-as-Judge, but acceptable for real-time evaluation.

#### 2. **Claim-Based Hallucination Detection**

**Decision**:  Extract individual claims from responses and verify each against context.

**Why**: 
- **Granular detection**:  Identifies exactly which claims are unsupported
- **Actionable feedback**:  "Claim X is not supported" vs "Response has hallucinations"
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

**Decision**:  Separate evaluators for each metric, orchestrated by a central pipeline.

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

---

## Scalability & Performance

### How We Handle Millions of Daily Conversations

#### 1. **Lightweight Models**

| Model | Size | Inference Time | Memory |
|-------|------|----------------|--------|
| all-MiniLM-L6-v2 | 80MB | ~20ms | ~200MB |
| facebook/bart-large-mnli | 1.6GB | ~200ms | ~2GB |

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
- Warm evaluation: ~200ms

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
- **Load Balancer**:  Distribute across multiple evaluator instances
- **Redis Cache**:  Shared embedding cache across instances
- **GPU Nodes**: For high-throughput batch processing
- **Async Queue**:  Kafka/RabbitMQ for non-blocking evaluation

#### 7. **Latency Breakdown (Actual Measurements)**

From our test runs: 

| Stage | Time | % of Total |
|-------|------|------------|
| Relevance | ~300ms | 20% |
| Hallucination | ~1200ms | 65% |
| Completeness | ~200ms | 15% |
| **Total** | ~1700ms | 100% |

**Optimization Opportunity**: Hallucination detection is the bottleneck. For real-time requirements (<500ms), consider:
- Reducing claim extraction depth
- Using simpler similarity threshold instead of NLI
- Async evaluation with results pushed via webhook

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

### Evaluate Custom Files
Provide your own conversation and context JSON files:
```bash
python main.py -c path/to/conversation.json -x path/to/context.json
```

Save results to a file:
```bash
python main.py -c conversation.json -x context.json -o results.json
```

### Programmatic Usage

```python
from evaluator. pipeline import EvaluationPipeline

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
print(f"Overall Score: {result. overall_score}")
print(f"Passed:  {result.passed}")
print(f"Hallucination Score: {result.hallucination. score}")
print(f"Unsupported Claims: {result.hallucination.unsupported_claims}")

# Export to JSON
result_dict = result.to_dict()
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
        "missing_aspects":  []
    },
    "latency": {
        "total_ms": 1500.5,
        "relevance_ms": 500.2,
        "hallucination_ms": 800.1,
        "completeness_ms": 200.2
    },
    "cost":  {
        "input_tokens": 500,
        "output_tokens":  100,
        "total_tokens":  600,
        "estimated_cost_usd": 0.001
    }
}
```

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

### Overall Score
```
overall = 0.35 × relevance + 0.40 × (1 - hallucination) + 0.25 × completeness
```

---

## Running Tests

```bash
# Run all tests
python -m unittest tests. test_pipeline -v

# Run specific test class
python -m unittest tests.test_pipeline.TestHallucinationDetector -v
```

---

## Project Structure

```
llm-evaluation-pipeline/
├── evaluator/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration management
│   ├── models. py           # Data models (dataclasses)
│   ├── utils.py             # Text processing, timing utilities
│   ├── relevance.py         # Relevance & completeness evaluators
│   ├── hallucination.py     # Hallucination detection
│   ├── cost_tracker.py      # Cost and latency tracking
│   └── pipeline.py          # Main orchestration pipeline
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Comprehensive unit tests
├── data/
│   ├── sample-chat-conversation-01.json
│   ├── sample-chat-conversation-02.json
│   ├── sample_context_vectors-01.json
│   └── sample_context_vectors-02.json
├── main.py                  # CLI entry point
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

---

## Technologies Used

- **sentence-transformers**: Semantic similarity with pre-trained models
- **transformers**: NLI model for entailment checking
- **scikit-learn**: Cosine similarity calculations
- **tiktoken**: Token counting for cost estimation
- **NumPy**: Numerical operations

---

## Author

**JustJay7** - BeyondChats AI Internship Assignment

---

## License

MIT License