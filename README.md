# rageval: RAG Pipeline Evaluator & LoRA Fine-tuner

A comprehensive CLI harness to evaluate RAG pipelines and continuously fine-tune domain-specific models.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![RAGAS Eval](https://img.shields.io/badge/eval-RAGAS-orange.svg)
![HuggingFace PEFT](https://img.shields.io/badge/finetune-HuggingFace-purple.svg)

## The Problem
RAG pipelines often fail silently: context chunks are missed, models hallucinate due to misaligned system prompts, or token-limits clip important answers. **Rageval** fixes this by providing rigorous objective evaluation metrics combined with an automated fine-tuning flywheel.

## What it measures
Rageval integrates directly with RAGAS to compute three fundamental metrics:
* **Faithfulness**: Measures if the generated answer can be entirely inferred from the retrieved chunks (detects factual hallucination).
* **Answer Relevancy**: Assesses how well the final answer addresses the user's original query (detects tangential output).
* **Context Recall**: Determines whether the retrieved chunks actually contained the necessary information required by the ground truth (detects poor embedding/retrieval).

## Quickstart

```bash
# Clone the repository
git clone https://github.com/yourusername/rageval.git
cd rageval

# Install with all dependencies
make install

# Export required API keys
export OPENAI_API_KEY="sk-..."       # For default GPT-4o-mini generation and RAGAS
export ANTHROPIC_API_KEY="sk-..."    # Only if generating synthetic fine-tuning data

# Run full ingest and evaluation
make ingest
make eval
```

## Example output

```text
Evaluating: What is DSO?
Evaluating: What is three-way matching?
...
                             RAG Evaluation Results                              
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric           ┃    Score ┃                                                ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Answer Relevancy │     0.97 │ ███████████████████░                           │
│ Context Recall   │     0.91 │ ██████████████████░░                           │
│ Faithfulness     │     0.88 │ █████████████████░░░                           │
└──────────────────┴──────────┴────────────────────────────────────────────────┘
```

## Results

| Metric | Score | What it means |
|--------|-------|---------------|
| Context Recall | `0.91` | The retrieval mechanism successfully found 91% of ground truth sentences in the ingested documents. |
| Faithfulness | `0.88` | High factual fidelity; very low instance of independent hallucination relative to the provided context. |
| Relevancy | `0.97` | Model output was highly conversational and precisely mapped back to the user query constraint. |

## Fine-tuning

Rageval supports creating targeted QLoRA fine-tunes for models (e.g. `microsoft/Phi-3-mini-4k-instruct`) using automatically generated synthetic financial QA datasets. This resolves systematic gaps discovered during pipeline evaluation.

To fine-tune and benchmark the performance delta on holdback questions:
```bash
make generate-data
make finetune
make benchmark
```
For more detailed documentation, see the [finetune/README](finetune/) and the architectural deep-dive at `docs/architecture.md`.

## Reproducing Results

To natively reproduce scores, pull the repository, supply data in `data/raw/` and `data/golden/`, then:
1. `make install`
2. `export ANTHROPIC_API_KEY=...` & `export OPENAI_API_KEY=...`
3. `make all`

## Version History
Rageval automatically git tags versions upon each `make eval` run. You can view metric trends internally with:
```bash
rageval history
```

---
*Built by Muhammed Nehan — open to feedback and contributions*
