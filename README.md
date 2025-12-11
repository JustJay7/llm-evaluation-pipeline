# LLM Response Evaluation Pipeline

A Python pipeline for evaluating LLM/chatbot responses in RAG (Retrieval-Augmented Generation) systems.  Evaluates responses for **relevance**, **hallucination**, and **completeness** while tracking **latency** and **cost** metrics.

Built for the BeyondChats AI Internship Assignment.

## Features

- **Relevance Evaluation**: Measures semantic similarity between response, query, and retrieved context
- **Hallucination Detection**: Identifies claims in responses not supported by the provided context
- **Completeness Checking**: Verifies if the response adequately addresses the user's question
- **Latency Tracking**: Detailed timing breakdown for each evaluation stage
- **Cost Estimation**: Token counting and cost calculation for LLM operations
- **Flexible Input**: Supports BeyondChats JSON format and generic conversation/context formats

## Project Structure

```
llm-evaluation-pipeline/
├── evaluator/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── models.py           # Data models and structures
│   ├── utils.py            # Utility functions
│   ├── relevance.py        # Relevance & completeness evaluation
│   ├── hallucination.py    # Hallucination detection
│   ├── cost_tracker.py     # Cost and latency tracking
│   └── pipeline.py         # Main evaluation pipeline
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # Unit tests
├── data/
│   ├── sample-chat-conversation-01.json
│   ├── sample-chat-conversation-02.json
│   ├── sample_context_vectors-01.json
│   └── sample_context_vectors-02.json
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/llm-evaluation-pipeline.git
cd llm-evaluation-pipeline
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

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
from evaluator.pipeline import EvaluationPipeline

# Initialize pipeline
pipeline = EvaluationPipeline()

# Prepare input data
conversation_json = {
    "conversation_turns": [
        {"role": "User", "message": "What is IVF?"},
        {"role": "AI/Chatbot", "message": "IVF is in vitro fertilization... "}
    ]
}

context_json = {
    "data":  {
        "vector_data": [
            {"text": "IVF (In Vitro Fertilization) is a medical procedure... "}
        ]
    }
}

# Run evaluation
result = pipeline.evaluate_from_json(conversation_json, context_json)

# Access results
print(f"Overall Score: {result. overall_score}")
print(f"Passed: {result.passed}")
print(f"Hallucination Score: {result.hallucination.score}")
print(f"Relevance Score: {result.relevance.score}")

# Get detailed results as dictionary
result_dict = result.to_dict()
```

## Input Format

### Conversation JSON (BeyondChats Format)
```json
{
    "chat_id": 12345,
    "conversation_turns": [
        {
            "turn":  1,
            "role":  "User",
            "message":  "What is the cost of IVF? ",
            "created_at": "2025-01-01T10:00:00.000000Z"
        },
        {
            "turn": 2,
            "role": "AI/Chatbot",
            "message":  "The cost of IVF at our clinic is.. .",
            "created_at":  "2025-01-01T10:00:05.000000Z"
        }
    ]
}
```

### Context JSON (BeyondChats Format)
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

## Output Format

```json
{
    "overall_score": 0.85,
    "passed": true,
    "timestamp": "2025-01-01T12:00:00.000000",
    "relevance": {
        "score": 0.82,
        "is_relevant":  true,
        "query_response_similarity": 0.75,
        "context_response_similarity": 0.88
    },
    "hallucination":  {
        "score": 0.1,
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
        "total_ms": 1500. 5,
        "relevance_ms": 500.2,
        "hallucination_ms": 800.1,
        "completeness_ms": 200.2
    },
    "cost": {
        "input_tokens": 500,
        "output_tokens":  100,
        "total_tokens": 600,
        "estimated_cost_usd": 0.001
    }
}
```

## Evaluation Metrics

### Relevance Score (0-1)
Measures how well the response relates to both the user query and retrieved context using semantic similarity (sentence embeddings + cosine similarity).

- **Weight**: Query (40%) + Context (60%)
- **Threshold**: 0.6 (configurable)

### Hallucination Score (0-1)
Detects unsupported claims by checking each statement against the context.

- **0.0** = No hallucination (all claims supported)
- **1.0** = Complete hallucination (no claims supported)
- **Threshold**: 0.5 (above = hallucinated)

### Completeness Score (0-1)
Evaluates whether key aspects of the question are addressed in the response.

- **1.0** = All aspects covered
- **0.0** = No aspects covered
- **Threshold**: 0.5 (configurable)

### Overall Score
Weighted combination:  `Relevance (35%) + (1 - Hallucination) (40%) + Completeness (25%)`

## Running Tests

```bash
python -m unittest tests.test_pipeline -v
```

## Configuration

Environment variables for customization: 

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EVAL_RELEVANCE_THRESHOLD` | `0.6` | Minimum relevance score |
| `EVAL_HALLUCINATION_THRESHOLD` | `0.5` | Maximum hallucination score |
| `EVAL_INPUT_TOKEN_COST` | `0.0015` | Cost per 1K input tokens |
| `EVAL_OUTPUT_TOKEN_COST` | `0.002` | Cost per 1K output tokens |

## Technologies Used

- **sentence-transformers**:  Semantic similarity with pre-trained models
- **transformers**: NLI model for entailment checking
- **scikit-learn**:  Cosine similarity calculations
- **tiktoken**: Token counting for cost estimation
- **NumPy**: Numerical operations

## Author

**JustJay7** - BeyondChats AI Internship Assignment

## License

MIT License