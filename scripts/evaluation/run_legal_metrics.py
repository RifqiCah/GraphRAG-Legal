import argparse
import csv
import json
import logging
from pathlib import Path
import statistics
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from scripts.evaluation.legal_evaluator import LegalEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Legal-Reference-Aware Evaluation Metrics")
    parser.add_argument("--input", required=True, type=str, help="Path to prediction CSV file")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save evaluation results")
    parser.add_argument("--smoke-test", type=int, default=0, help="Run on N samples only")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_results_path = output_dir / "detailed_results.jsonl"
    aggregate_results_path = output_dir / "aggregate_results.csv"
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # Try to determine model, condition, retrieval_mode from filename
    filename = input_path.name.lower()
    
    model_name = "unknown"
    if "llama2" in filename: model_name = "llama2_7b"
    elif "wizardlm" in filename: model_name = "wizardlm_13b"
    
    condition = "unknown"
    if "finetune" in filename or "lora" in filename: condition = "fine-tuned"
    elif "zeroshot" in filename: condition = "zero-shot"
    
    retrieval_mode = "unknown"
    if "graphrag" in filename: retrieval_mode = "graphrag"
    elif "rag" in filename or "vector" in filename: retrieval_mode = "rag"
    else: retrieval_mode = "no_retrieval"

    evaluator = LegalEvaluator(cache_file=str(output_dir / "cache.json"))
    
    logger.info(f"Starting evaluation for {input_path}")
    logger.info(f"Detected context: Model={model_name}, Condition={condition}, Retrieval={retrieval_mode}")

    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.smoke_test > 0:
        rows = rows[:args.smoke_test]
        logger.info(f"Running in smoke test mode with {args.smoke_test} samples.")

    # Prepare Dataset for RAGAS
    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for row in rows:
        dataset_dict["question"].append(row.get("input", ""))
        dataset_dict["answer"].append(row.get("prediction", ""))
        context = row.get("graph_context", row.get("retrieved_context", row.get("vector_context", "")))
        dataset_dict["contexts"].append([context] if context else [])
        dataset_dict["ground_truth"].append(row.get("reference", row.get("ground_truth", "")))
        
    ds = Dataset.from_dict(dataset_dict)
    
    logger.info("Running RAGAS evaluation (Faithfulness, Context Recall)...")
    
    from langchain_openai import ChatOpenAI
    llm_model = os.getenv("EVAL_MODEL", "gpt-4o-mini")
    eval_llm = ChatOpenAI(model=llm_model)

    # Setting up ragas evaluation
    ragas_result = evaluate(
        dataset=ds,
        metrics=[faithfulness, context_recall],
        llm=eval_llm,
        embeddings=None
    )
    
    ragas_df = ragas_result.to_pandas()

    from tqdm import tqdm

    results = []
    logger.info("Running Custom Legal Reference Accuracy evaluation...")
    with open(detailed_results_path, 'w', encoding='utf-8') as f_out:
        for i, row in enumerate(tqdm(rows, desc="Evaluating Legal Ref Acc")):
            sample_id = row.get("index", row.get("id", str(i)))
            task = row.get("label", "unknown")
            generated_answer = row.get("prediction", "")
            reference_answer = row.get("reference", row.get("ground_truth", ""))
            
            # Custom metric: Context Legal Recall
            context_text = row.get("graph_context", row.get("retrieved_context", row.get("vector_context", "")))
            lra_res = evaluator.evaluate_context_legal_accuracy(context_text, reference_answer)
            
            # RAGAS metrics
            f_score = ragas_df.iloc[i].get("faithfulness", 0.0)
            cr_score = ragas_df.iloc[i].get("context_recall", 0.0)
            
            result_row = {
                "sample_id": sample_id,
                "task": task,
                "model": model_name,
                "condition": condition,
                "retrieval_mode": retrieval_mode,
                "faithfulness": {"score": f_score},
                "context_recall": {"score": cr_score},
                "legal_reference_accuracy": lra_res
            }
            results.append(result_row)
            f_out.write(json.dumps(result_row) + "\n")
            
            # Print intermediate score every 10 samples
            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                current_fa = statistics.mean([r["legal_reference_accuracy"].get("context_legal_recall", 0.0) for r in results])
                logger.info(f"--- [Intermediate Score {i+1}/{len(rows)}] Current Context Legal Ref Recall: {current_fa:.4f} ---")

    # Calculate Aggregates
    logger.info("Calculating aggregates...")
    
    metrics = {
        "faithfulness": [],
        "context_recall": [],
        "context_legal_recall": []
    }
    
    for r in results:
        # Check if RAGAS returned NaN/None for some rows
        f_val = r["faithfulness"].get("score", 0.0)
        cr_val = r["context_recall"].get("score", 0.0)
        
        metrics["faithfulness"].append(f_val if f_val == f_val else 0.0) # handle NaN
        metrics["context_recall"].append(cr_val if cr_val == cr_val else 0.0)
        
        metrics["context_legal_recall"].append(r["legal_reference_accuracy"].get("context_legal_recall", 0.0))

    def get_stats(vals):
        if not vals: return 0.0, 0.0, 0.0
        return statistics.mean(vals), statistics.median(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    f_mean, f_med, f_std = get_stats(metrics["faithfulness"])
    cr_mean, cr_med, cr_std = get_stats(metrics["context_recall"])
    fa_mean, fa_med, fa_std = get_stats(metrics["context_legal_recall"])

    with open(aggregate_results_path, 'w', encoding='utf-8', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Metric", "Mean", "Median", "StdDev"])
        writer.writerow(["Faithfulness", f_mean, f_med, f_std])
        writer.writerow(["Context Recall", cr_mean, cr_med, cr_std])
        writer.writerow(["Context Legal Reference Recall", fa_mean, fa_med, fa_std])

    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    logger.info(f"Mean Faithfulness: {f_mean:.4f}")
    logger.info(f"Mean Context Recall: {cr_mean:.4f}")
    logger.info(f"Mean Context Legal Ref Recall: {fa_mean:.4f}")

if __name__ == "__main__":
    main()
