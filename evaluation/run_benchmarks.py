#!/usr/bin/env python3
"""
Official Benchmark Evaluation for LocalMod

Tests LocalMod classifiers against established public benchmarks:

1. TOXICITY: Jigsaw Toxic Comments (Civil Comments subset)
   - Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
   - HuggingFace: google/civil_comments

2. SPAM: SMS Spam Collection (UCI)
   - Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
   - HuggingFace: sms_spam

3. PROMPT INJECTION: Deepset Prompt Injection Dataset
   - Source: https://huggingface.co/datasets/deepset/prompt-injections
   - HuggingFace: deepset/prompt-injections

Usage:
    python evaluation/run_benchmarks.py
    python evaluation/run_benchmarks.py --benchmark toxicity
    python evaluation/run_benchmarks.py --samples 500
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_dataset_safe(name: str, split: str, config: str = None):
    """Load dataset with error handling."""
    try:
        from datasets import load_dataset
        # Try with trust_remote_code first (newer versions), fallback without
        try:
            if config:
                return load_dataset(name, config, split=split, trust_remote_code=True)
            return load_dataset(name, split=split, trust_remote_code=True)
        except TypeError:
            # Older datasets version doesn't support trust_remote_code
            if config:
                return load_dataset(name, config, split=split)
            return load_dataset(name, split=split)
    except Exception as e:
        print(f"   ❌ Failed to load {name}: {e}")
        return None


def evaluate_toxicity(samples: int = 500) -> Dict:
    """
    Evaluate on Jigsaw Civil Comments dataset.
    
    Labels: toxicity score 0-1 (we use threshold 0.5)
    """
    print("\n" + "="*70)
    print("📊 TOXICITY BENCHMARK: Jigsaw Civil Comments")
    print("="*70)
    print(f"   Dataset: google/civil_comments")
    print(f"   Samples: {samples}")
    
    # Load dataset
    print("\n   Loading dataset...")
    dataset = load_dataset_safe("google/civil_comments", split="test")
    if dataset is None:
        return {"error": "Failed to load dataset"}
    
    # Sample subset
    if len(dataset) > samples:
        dataset = dataset.shuffle(seed=42).select(range(samples))
    
    print(f"   Loaded {len(dataset)} samples")
    
    # Load classifier
    print("   Loading classifier...")
    from localmod.classifiers.toxicity import ToxicityClassifier
    clf = ToxicityClassifier(device="cpu")
    clf.load()
    
    # Evaluate
    print("   Running evaluation...")
    results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    predictions = []
    
    start_time = time.time()
    for i, item in enumerate(dataset):
        text = item["text"]
        # Civil comments uses toxicity score 0-1
        is_toxic = item["toxicity"] >= 0.5
        
        pred = clf.predict(text)
        pred_toxic = pred.flagged
        
        if pred_toxic and is_toxic:
            results["tp"] += 1
        elif not pred_toxic and not is_toxic:
            results["tn"] += 1
        elif pred_toxic and not is_toxic:
            results["fp"] += 1
        else:
            results["fn"] += 1
        
        predictions.append({
            "text": text[:100],
            "label": is_toxic,
            "pred": pred_toxic,
            "conf": pred.confidence
        })
        
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(dataset)}")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    metrics["samples"] = len(dataset)
    metrics["time_sec"] = round(elapsed, 2)
    metrics["samples_per_sec"] = round(len(dataset) / elapsed, 1)
    
    print_metrics("Toxicity", metrics)
    return metrics


def evaluate_spam(samples: int = 500) -> Dict:
    """
    Evaluate on UCI SMS Spam Collection.
    
    Labels: ham (0) or spam (1)
    """
    print("\n" + "="*70)
    print("📊 SPAM BENCHMARK: UCI SMS Spam Collection")
    print("="*70)
    print(f"   Dataset: sms_spam")
    print(f"   Samples: {samples}")
    
    # Load dataset
    print("\n   Loading dataset...")
    dataset = load_dataset_safe("sms_spam", split="train")
    if dataset is None:
        return {"error": "Failed to load dataset"}
    
    # Sample subset
    if len(dataset) > samples:
        dataset = dataset.shuffle(seed=42).select(range(samples))
    
    print(f"   Loaded {len(dataset)} samples")
    
    # Load classifier
    print("   Loading classifier...")
    from localmod.classifiers.spam import SpamClassifier
    clf = SpamClassifier(device="cpu")
    clf.load()
    
    # Evaluate
    print("   Running evaluation...")
    results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    
    start_time = time.time()
    for i, item in enumerate(dataset):
        text = item["sms"]
        is_spam = item["label"] == 1  # 1 = spam, 0 = ham
        
        pred = clf.predict(text)
        pred_spam = pred.flagged
        
        if pred_spam and is_spam:
            results["tp"] += 1
        elif not pred_spam and not is_spam:
            results["tn"] += 1
        elif pred_spam and not is_spam:
            results["fp"] += 1
        else:
            results["fn"] += 1
        
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(dataset)}")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    metrics["samples"] = len(dataset)
    metrics["time_sec"] = round(elapsed, 2)
    metrics["samples_per_sec"] = round(len(dataset) / elapsed, 1)
    
    print_metrics("Spam", metrics)
    return metrics


def evaluate_prompt_injection(samples: int = 500) -> Dict:
    """
    Evaluate on Deepset Prompt Injection dataset.
    
    Labels: 0 = safe, 1 = injection
    """
    print("\n" + "="*70)
    print("📊 PROMPT INJECTION BENCHMARK: Deepset Prompt Injections")
    print("="*70)
    print(f"   Dataset: deepset/prompt-injections")
    print(f"   Samples: {samples}")
    
    # Load dataset
    print("\n   Loading dataset...")
    dataset = load_dataset_safe("deepset/prompt-injections", split="train")
    if dataset is None:
        return {"error": "Failed to load dataset"}
    
    # Sample subset
    if len(dataset) > samples:
        dataset = dataset.shuffle(seed=42).select(range(samples))
    
    print(f"   Loaded {len(dataset)} samples")
    
    # Load classifier
    print("   Loading classifier...")
    from localmod.classifiers.prompt_injection import PromptInjectionDetector
    clf = PromptInjectionDetector(device="cpu")
    clf.load()
    
    # Evaluate
    print("   Running evaluation...")
    results = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    
    start_time = time.time()
    for i, item in enumerate(dataset):
        text = item["text"]
        is_injection = item["label"] == 1
        
        pred = clf.predict(text)
        pred_injection = pred.flagged
        
        if pred_injection and is_injection:
            results["tp"] += 1
        elif not pred_injection and not is_injection:
            results["tn"] += 1
        elif pred_injection and not is_injection:
            results["fp"] += 1
        else:
            results["fn"] += 1
        
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(dataset)}")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    metrics["samples"] = len(dataset)
    metrics["time_sec"] = round(elapsed, 2)
    metrics["samples_per_sec"] = round(len(dataset) / elapsed, 1)
    
    print_metrics("Prompt Injection", metrics)
    return metrics


def calculate_metrics(results: Dict) -> Dict:
    """Calculate precision, recall, F1, accuracy from confusion matrix."""
    tp, tn, fp, fn = results["tp"], results["tn"], results["fp"], results["fn"]
    total = tp + tn + fp + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(name: str, metrics: Dict):
    """Print metrics in a nice format."""
    if "error" in metrics:
        print(f"\n   ❌ Error: {metrics['error']}")
        return
    
    print(f"\n   {'─'*50}")
    print(f"   📈 {name} Results")
    print(f"   {'─'*50}")
    print(f"   Accuracy:  {metrics['accuracy']:5.1f}%")
    print(f"   Precision: {metrics['precision']:5.1f}%")
    print(f"   Recall:    {metrics['recall']:5.1f}%")
    print(f"   F1 Score:  {metrics['f1']:5.1f}%")
    print(f"   {'─'*50}")
    print(f"   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Pos     Neg")
    print(f"   Actual Pos    {metrics['tp']:4d}    {metrics['fn']:4d}")
    print(f"   Actual Neg    {metrics['fp']:4d}    {metrics['tn']:4d}")
    print(f"   {'─'*50}")
    print(f"   Throughput: {metrics['samples_per_sec']} samples/sec")
    print(f"   Total time: {metrics['time_sec']}s for {metrics['samples']} samples")


def main():
    parser = argparse.ArgumentParser(description="Run official benchmarks on LocalMod")
    parser.add_argument(
        "--benchmark", "-b",
        choices=["toxicity", "spam", "prompt_injection", "all"],
        default="all",
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=500,
        help="Number of samples per benchmark (default: 500)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file for results"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🏆 LocalMod Official Benchmark Evaluation")
    print("="*70)
    print(f"\n   Samples per benchmark: {args.samples}")
    
    # Check for datasets library
    try:
        from datasets import load_dataset
        print("   ✅ HuggingFace datasets library available")
    except ImportError:
        print("   ❌ HuggingFace datasets library not found")
        print("   Install with: pip install datasets")
        return 1
    
    results = {}
    
    if args.benchmark in ["toxicity", "all"]:
        results["toxicity"] = evaluate_toxicity(args.samples)
    
    if args.benchmark in ["spam", "all"]:
        results["spam"] = evaluate_spam(args.samples)
    
    if args.benchmark in ["prompt_injection", "all"]:
        results["prompt_injection"] = evaluate_prompt_injection(args.samples)
    
    # Summary
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n   {'Benchmark':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"   {'-'*60}")
    
    for name, metrics in results.items():
        if "error" not in metrics:
            print(f"   {name:<20} {metrics['accuracy']:>9.1f}% {metrics['precision']:>9.1f}% {metrics['recall']:>9.1f}% {metrics['f1']:>9.1f}%")
        else:
            print(f"   {name:<20} {'ERROR':>10}")
    
    print(f"   {'-'*60}")
    
    # Average (excluding errors)
    valid_results = [m for m in results.values() if "error" not in m]
    if valid_results:
        avg_acc = sum(m["accuracy"] for m in valid_results) / len(valid_results)
        avg_f1 = sum(m["f1"] for m in valid_results) / len(valid_results)
        print(f"   {'AVERAGE':<20} {avg_acc:>9.1f}%{' '*10}{' '*10} {avg_f1:>9.1f}%")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n   📄 Results saved to: {args.output}")
    
    print("\n" + "="*70)
    print("✅ Benchmark evaluation complete!")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

