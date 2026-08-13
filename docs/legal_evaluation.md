# Legal-Reference-Aware Evaluation Metrics

This module provides three domain-specific evaluation metrics for the Indonesian Educational Legal Corpus (ASVRI-PSKP-Legal-SFT):

1. **Faithfulness**
2. **Context Recall**
3. **Legal Reference Accuracy**

These metrics utilize LLM-as-a-judge methodologies to evaluate RAG/GraphRAG experiments robustly without altering existing pipelines.

## Definitions

### 1. Faithfulness
Measures whether the factual claims made in the model's generated answer are fully supported by the retrieved context provided to it. 
- A claim is `SUPPORTED` (1.0), `PARTIALLY_SUPPORTED` (0.5), or `NOT_SUPPORTED` (0.0).
- **Formula**: `(SUPPORTED + 0.5 * PARTIALLY_SUPPORTED) / TOTAL_CLAIMS`

### 2. Context Recall
Measures whether the retrieved context contains all the necessary information to reconstruct the ground-truth/reference answer.
- A reference claim is `FOUND` (1.0), `PARTIAL` (0.5), or `NOT_FOUND` (0.0).
- **Formula**: `(FOUND + 0.5 * PARTIAL) / TOTAL_REFERENCE_CLAIMS`

### 3. Legal Reference Accuracy
This is a stringent, multi-stage domain-specific metric that evaluates whether legal citations in the generated answer are accurate and factually valid.
It checks:
- **Regulation Accuracy**: Did the model mention a valid regulation that actually exists in the corpus?
- **Article/Paragraph Accuracy**: Are the specific articles and paragraphs correct and existent?
- **Full Reference Accuracy**: A reference is only fully accurate if the Regulation, Article, Paragraph are correct/existent AND the actual text of that article/paragraph in the corpus *supports* the claim made by the model. If a model hallucinates the contents of a real article, it scores 0.

## Architecture

- **`corpus_manager.py`**: Loads JSONL chunks from `data/chunks_v2` to create a fast-lookup index for checking legal reference existence. Normalizes raw text into canonical representations (e.g., `UU|20|2003|5|1`).
- **`legal_eval_prompts.py`**: Strict system prompts ensuring the LLM returns structured JSON with evidence spans and reasons.
- **`legal_evaluator.py`**: The core LLM driver containing the API call logic, caching, and scoring logic.
- **`run_legal_metrics.py`**: CLI tool to iterate over prediction CSVs and output detailed/aggregate metrics.

## Requirements

1. `openai` python package.
2. An OpenAI API Key configured in your `.env` or system environment (`OPENAI_API_KEY`).
3. (Optional) Customization variables:
   - `EVAL_MODEL`: defaults to `gpt-4o-mini`
   - `EVAL_TEMPERATURE`: defaults to `0.0`
   - `EVAL_MAX_TOKENS`: defaults to `2048`

## Usage

To evaluate a prediction CSV file:

```bash
# Full evaluation
python -m scripts.evaluation.run_legal_metrics \
    --input outputs/predictions/lora_graphrag_predictions_sample_wide.csv \
    --output-dir results/legal_eval

# Smoke test (run on first 5 samples only)
python -m scripts.evaluation.run_legal_metrics \
    --input outputs/predictions/lora_graphrag_predictions_sample_wide.csv \
    --output-dir results/legal_eval \
    --smoke-test 5
```

## Outputs

The evaluator will create a directory (e.g., `results/legal_eval`) containing:
1. `detailed_results.jsonl`: Line-by-line detailed results for every evaluated sample, including the exact claims extracted, evidence, and support checks.
2. `aggregate_results.csv`: A summary containing Mean, Median, and Standard Deviation for each metric across the evaluated file.
3. `cache.json`: Caches LLM responses based on prompt hashes to avoid redundant API costs and allow easy resumption.

## Limitations
- **Cost**: Processing thousands of samples with an LLM judge incurs cost. Caching is used extensively to mitigate this.
- **Normalizer Rules**: The regex normalizer inside `CorpusManager` covers major Indonesian legal formats (UU, PP, Permendikbud) but might fail on extremely obscure or malformed citations.
