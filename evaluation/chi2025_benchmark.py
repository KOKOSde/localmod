#!/usr/bin/env python3
"""
CHI 2025 Benchmark for LocalMod Toxicity Detection.

This script evaluates LocalMod's toxicity detection against the CHI 2025
"Lost in Moderation" benchmark methodology using:
- HateXplain dataset
- Civil Comments dataset  
- SBIC (Social Bias Inference Corpus)

Results are compared with published scores from commercial APIs.

Usage:
    python evaluation/chi2025_benchmark.py
    python evaluation/chi2025_benchmark.py --samples 500  # Quick test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    # Optional: only needed for non-toxicity benchmarks (LocalMod classifiers).
    # Import may fail if script is run without `PYTHONPATH=src`.
    from localmod.classifiers.nsfw import NSFWClassifier
    from localmod.classifiers.pii import PIIDetector
    from localmod.classifiers.prompt_injection import PromptInjectionDetector
    from localmod.classifiers.spam import SpamClassifier
except Exception:
    NSFWClassifier = None
    PIIDetector = None
    PromptInjectionDetector = None
    SpamClassifier = None


# Published commercial API scores from CHI 2025 paper
# Reference: Hartmann et al., "Lost in Moderation", CHI 2025
# https://arxiv.org/html/2503.01623
PUBLISHED_SCORES = {
    "OpenAI Content Moderation API": 0.83,
    "Microsoft Azure Content Moderation": 0.81,
    "Amazon Comprehend": 0.74,
    "Perspective API": 0.62,
    "Google Natural Language API": 0.59,
}

# LocalMod toxicity ensemble configuration
ENSEMBLE_CONFIG = {
    "toxicity": {"weight": 0.50, "type": "multilabel", "path": "~/.cache/localmod/models/toxicity"},
    "toxicity_dehatebert": {"weight": 0.20, "type": "binary", "path": "~/.cache/localmod/models/toxicity_dehatebert"},
    "toxicity_snlp": {"weight": 0.15, "type": "binary", "path": "~/.cache/localmod/models/toxicity_snlp"},
    "toxicity_facebook": {"weight": 0.15, "type": "binary", "path": "~/.cache/localmod/models/toxicity_facebook"},
}

# Alternative paths if models are saved with different names
ALTERNATIVE_PATHS = {
    "toxicity_dehatebert": "~/.cache/localmod/models/dehatebert",
    "toxicity_snlp": "~/.cache/localmod/models/snlp_toxicity",
    "toxicity_facebook": "~/.cache/localmod/models/fb_hate_speech",
}

def _confusion_counts(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Binary confusion matrix counts with positive label = 1."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def _binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Reader-friendly binary metrics.

    Positive class (1) is the "harmful / should-be-flagged" class:
    - Precision highlights false positives (benign flagged).
    - Recall highlights false negatives (harmful missed).
    """
    counts = _confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0  # specificity

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tnr),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "support": float(len(y_true)),
    }


def _device_for_arg(device: str) -> str:
    d = (device or "cpu").lower()
    if d == "cuda" and not torch.cuda.is_available():
        print("  ⚠️  CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return d


def _print_batch_progress(done: int, total: int, every: int = 10) -> None:
    """Lightweight progress logging (keeps runs debuggable on CPU)."""
    if total <= 0:
        return
    if done == total or (every > 0 and done % every == 0):
        print(f"    ... {done}/{total} batches")


def _make_synthetic_pii_dataset(n: int, seed: int = 42) -> List[Tuple[str, int]]:
    """
    Synthetic PII benchmark dataset.

    Positive label (1) means: text contains PII that should be flagged.
    This is not a public benchmark; it's intended to quantify false positives vs
    false negatives for the shipped deterministic detector.
    """
    rng = np.random.RandomState(seed)

    safe_templates = [
        "Let's meet tomorrow at the office.",
        "The quick brown fox jumps over the lazy dog.",
        "Can you review this PR before EOD?",
        "Dinner was great, thanks for hosting.",
        "Reminder: standup at 10am.",
    ]

    pii_templates = [
        "Email me at {email} about the invoice.",
        "Call me at {phone} when you arrive.",
        "My SSN is {ssn} (do not share).",
        "Card number: {cc} exp 11/29.",
        "Reach me: {email} or {phone}.",
    ]

    def rand_email(i: int) -> str:
        return f"user{i}@example.com"

    def rand_phone(i: int) -> str:
        return f"555-{1000 + (i % 9000):04d}"

    def rand_ssn(i: int) -> str:
        a = 100 + (i % 899)  # avoid 000 prefix
        b = 10 + (i % 89)
        c = 1000 + (i % 8999)
        return f"{a:03d}-{b:02d}-{c:04d}"

    def rand_cc(_: int) -> str:
        # Common Visa test number; passes Luhn.
        return "4111 1111 1111 1111"

    data: List[Tuple[str, int]] = []
    n_pos = n // 2
    n_neg = n - n_pos

    for i in range(n_neg):
        data.append((safe_templates[i % len(safe_templates)], 0))
    for i in range(n_pos):
        t = pii_templates[i % len(pii_templates)].format(
            email=rand_email(i),
            phone=rand_phone(i),
            ssn=rand_ssn(i),
            cc=rand_cc(i),
        )
        data.append((t, 1))

    rng.shuffle(data)
    return data


def run_pii_benchmark(max_samples: int = 2000) -> Dict:
    """Benchmark LocalMod PII detector on a synthetic labeled set."""
    if PIIDetector is None:
        raise RuntimeError(
            "LocalMod is not importable. Run with: PYTHONPATH=src python evaluation/chi2025_benchmark.py ..."
        )

    print("=" * 70)
    print("Benchmark - LocalMod PII Detection (Synthetic PII vs Safe)")
    print("=" * 70)

    n = int(max_samples) if max_samples else 2000
    data = _make_synthetic_pii_dataset(n=n)
    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    detector = PIIDetector()
    detector.load()

    preds: List[int] = []
    start = time.time()
    for t in texts:
        preds.append(1 if detector.predict(t).flagged else 0)
    elapsed = time.time() - start

    metrics = _binary_metrics(labels, preds)
    print(f"\nSamples: {len(texts)} (pos={sum(labels)}, neg={len(labels) - sum(labels)})")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"FP / FN:           {int(metrics['fp'])} / {int(metrics['fn'])}")
    print(f"Time: {elapsed:.1f}s")

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": "synthetic_pii_v1",
        "samples": int(metrics["support"]),
        "metrics": metrics,
        "time_seconds": elapsed,
        "notes": [
            "Synthetic dataset (balanced) used to quantify false positives vs false negatives for the deterministic detector.",
            "Positive label = contains PII (1).",
        ],
    }


def run_nsfw_image_benchmark(
    max_samples: int = 500,
    device: str = "cpu",
    batch_size: int = 16,
    threshold: float = 0.5,
) -> Dict:
    """
    Benchmark LocalMod NSFW image classifier on a proxy labeled image benchmark.

    Many NSFW image datasets on the Hub are either huge or lack explicit labels.
    This benchmark uses a small, reproducible proxy:
    - positive (1): `x1101/nsfw` images (assumed NSFW)
    - negative (0): `cifar10` test images (assumed SFW)
    """
    try:
        from localmod.classifiers.nsfw_image import ImageNSFWClassifier
    except Exception as e:
        raise RuntimeError(
            "LocalMod image classifier not importable. Run with: PYTHONPATH=src python evaluation/chi2025_benchmark.py ..."
        ) from e

    print("=" * 70)
    print("Benchmark - LocalMod NSFW Image Detection (Proxy: x1101/nsfw vs CIFAR-10)")
    print("=" * 70)

    # Some community image datasets contain truncated/corrupt files.
    # Allow PIL to load truncated images so we can skip/score them rather than crash.
    try:
        from PIL import ImageFile  # type: ignore
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    except Exception:
        pass

    n = int(max_samples) if max_samples else 200
    n_pos = max(1, n // 2)
    n_neg = n - n_pos

    pos_ds = load_dataset("x1101/nsfw", split=f"train[:{n_pos}]")
    neg_ds = load_dataset("cifar10", split=f"test[:{n_neg}]")

    images: List = []
    labels: List[int] = []

    # x1101/nsfw: column is "image"
    for row in pos_ds:
        img = row.get("image")
        if img is not None:
            images.append(img)
            labels.append(1)

    # cifar10: columns are "img" and "label"
    for row in neg_ds:
        img = row.get("img") or row.get("image")
        if img is not None:
            images.append(img)
            labels.append(0)

    if not images:
        raise RuntimeError("Failed to load proxy images for NSFW image benchmark.")

    clf = ImageNSFWClassifier(device=device, threshold=threshold)
    clf.load()

    preds: List[int] = []
    start = time.time()
    total_batches = (len(images) + batch_size - 1) // batch_size
    batch_i = 0
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        # classifier has predict_batch for images
        batch_res = clf.predict_batch(batch_imgs)
        preds.extend([1 if r.flagged else 0 for r in batch_res])
        batch_i += 1
        _print_batch_progress(batch_i, total_batches, every=10)
    elapsed = time.time() - start

    metrics = _binary_metrics(labels, preds)
    pos = int(sum(labels))
    print(f"\nSamples: {len(labels)} (pos={pos}, neg={len(labels) - pos})")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"FP / FN:           {int(metrics['fp'])} / {int(metrics['fn'])}")
    print(f"Time: {elapsed:.1f}s")

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": "proxy: x1101/nsfw (pos) vs cifar10 (neg)",
        "samples": int(metrics["support"]),
        "threshold": threshold,
        "metrics": metrics,
        "time_seconds": elapsed,
        "notes": [
            "Positive label = NSFW (1).",
            "Proxy benchmark: all x1101/nsfw images treated as NSFW positives; CIFAR-10 treated as SFW negatives.",
        ],
    }



def load_datasets(max_samples: int = None) -> Dict[str, List[Tuple[str, int]]]:
    """Load benchmark datasets."""
    print("\nLoading datasets...")
    datasets = {}
    
    # HateXplain
    from collections import Counter
    ds = load_dataset("hatexplain", split="test")
    hatexplain = []
    for item in ds:
        text = " ".join(item["post_tokens"])
        labels = item["annotators"]["label"]
        majority_label = Counter(labels).most_common(1)[0][0]
        hatexplain.append((text, 1 if majority_label > 0 else 0))
        if max_samples and len(hatexplain) >= max_samples:
            break
    datasets["HateXplain"] = hatexplain
    print(f"  HateXplain: {len(hatexplain)} samples")
    
    # Civil Comments
    sample_size = max_samples or 2000
    ds = load_dataset("google/civil_comments", split=f"test[:{sample_size}]")
    civil = [(item["text"][:512], 1 if item["toxicity"] >= 0.5 else 0) for item in ds]
    datasets["Civil Comments"] = civil
    print(f"  Civil Comments: {len(civil)} samples")
    
    # SBIC
    ds = load_dataset("social_bias_frames", split=f"test[:{sample_size}]")
    sbic = []
    for item in ds:
        try:
            score = float(item["offensiveYN"]) if item["offensiveYN"] else 0.0
        except:
            score = 0.0
        sbic.append((item["post"][:512], 1 if score >= 0.5 else 0))
    datasets["SBIC"] = sbic
    print(f"  SBIC: {len(sbic)} samples")
    
    return datasets


def load_ensemble_models(device: str = "cpu") -> Dict[str, Dict]:
    """Load all toxicity ensemble models."""
    print("\nLoading ensemble models...")
    models = {}
    
    for name, config in ENSEMBLE_CONFIG.items():
        path = os.path.expanduser(config["path"])
        
        # Try alternative path if primary doesn't exist
        if not os.path.exists(path) and name in ALTERNATIVE_PATHS:
            path = os.path.expanduser(ALTERNATIVE_PATHS[name])
        
        if not os.path.exists(path):
            print(f"  ⚠️  {name}: Not found at {path}")
            continue
        
        try:
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(device)
            model.eval()
            models[name] = {
                "tokenizer": tok,
                "model": model,
                "type": config["type"],
                "weight": config["weight"],
            }
            print(f"  ✅ {name}: Loaded ({config['weight']*100:.0f}% weight)")
        except Exception as e:
            print(f"  ❌ {name}: Failed - {e}")
    
    return models


def get_ensemble_predictions(
    models: Dict[str, Dict],
    texts: List[str],
    threshold: float = 0.17,
    batch_size: int = 64,
    device: str = "cpu",
) -> List[int]:
    """Get predictions using weighted ensemble."""
    n = len(texts)
    weighted_probs = np.zeros(n)
    total_weight = 0.0
    
    for name, m in models.items():
        all_probs = []

        total_batches = (n + batch_size - 1) // batch_size
        batch_i = 0
        for i in range(0, n, batch_size):
            batch = texts[i:i+batch_size]
            inputs = m["tokenizer"](
                batch, return_tensors="pt", truncation=True, max_length=256, padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = m["model"](**inputs)
            
            if m["type"] == "multilabel":
                probs = torch.sigmoid(out.logits).max(dim=-1).values
            else:
                probs = torch.softmax(out.logits, dim=-1)[:, 1]
            
            all_probs.extend(probs.cpu().tolist())
            batch_i += 1
            _print_batch_progress(batch_i, total_batches, every=10)
        
        weighted_probs += np.array(all_probs) * m["weight"]
        total_weight += m["weight"]
    
    # Normalize if not all models loaded
    if total_weight > 0 and total_weight < 1.0:
        weighted_probs /= total_weight
    
    return (weighted_probs >= threshold).astype(int).tolist()


def run_chi2025_toxicity_benchmark(
    max_samples: int = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict:
    """Run the CHI 2025 "Lost in Moderation" toxicity benchmark."""
    print("=" * 70)
    print("CHI 2025 Benchmark - LocalMod Toxicity Detection")
    print("=" * 70)
    
    # Load data
    datasets = load_datasets(max_samples)
    
    # Load models
    models = load_ensemble_models(device)
    
    if not models:
        print("\n❌ No models loaded. Run 'python scripts/download_models.py' first.")
        return {}
    
    # Evaluate on each dataset
    results = {}
    
    for ds_name, data in datasets.items():
        print(f"\nEvaluating on {ds_name}...")
        start = time.time()
        
        texts = [t for t, _ in data]
        labels = [l for _, l in data]
        
        preds = get_ensemble_predictions(models, texts, batch_size=batch_size, device=device)
        
        metrics = _binary_metrics(labels, preds)
        
        elapsed = time.time() - start
        
        results[ds_name] = {
            **{k: v for k, v in metrics.items() if k not in ("support",)},
            "samples": len(data),
            "time_seconds": elapsed,
        }
        
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"  Precision:         {metrics['precision']:.3f}")
        print(f"  Recall:            {metrics['recall']:.3f}")
        print(f"  F1 Score:          {metrics['f1_score']:.3f}")
        print(f"  FP / FN:           {int(metrics['fp'])} / {int(metrics['fn'])}")
        print(f"  Time: {elapsed:.1f}s")
    
    # Calculate average
    avg_bal_acc = float(np.mean([r["balanced_accuracy"] for r in results.values()]))
    avg_f1 = float(np.mean([r["f1_score"] for r in results.values()]))
    
    results["average"] = {
        "balanced_accuracy": avg_bal_acc,
        "f1_score": avg_f1,
    }
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH COMMERCIAL APIs")
    print("=" * 70)
    
    all_scores = {**PUBLISHED_SCORES, "LocalMod (Ensemble)": avg_bal_acc}
    for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        marker = "⭐" if "LocalMod" in name else ""
        print(f"{name:35} {score:.2f} {marker}")
    
    print(f"\n🏆 LocalMod achieved {avg_bal_acc:.3f} balanced accuracy")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(models.keys()),
        "threshold": 0.17,
        "datasets": results,
        "comparison": all_scores,
    }


def run_uci_sms_spam_benchmark(max_samples: int = None, device: str = "cpu", batch_size: int = 64) -> Dict:
    """Benchmark LocalMod spam classifier on the UCI SMS Spam Collection."""
    if SpamClassifier is None:
        raise RuntimeError("LocalMod is not importable. Run with: PYTHONPATH=src python evaluation/chi2025_benchmark.py ...")

    print("=" * 70)
    print("Benchmark - LocalMod Spam Detection (UCI SMS Spam Collection)")
    print("=" * 70)

    ds = load_dataset("ucirvine/sms_spam", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = [x["sms"] for x in ds]
    labels = [int(x["label"]) for x in ds]

    clf = SpamClassifier(device=device, threshold=0.5, use_ml_model=True)
    clf.load()

    preds: List[int] = []
    start = time.time()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    batch_i = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        heuristic_scores = []
        for t in batch:
            _, h = clf._check_heuristics(t)  # type: ignore[attr-defined]
            heuristic_scores.append(float(h))

        # If model failed to load, fall back to heuristic-only.
        if not clf.use_ml_model or clf._tokenizer is None:
            confidences = heuristic_scores
        else:
            inputs = clf._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = clf._model(**inputs)  # type: ignore[misc]
            probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
            confidences = (0.6 * probs + 0.4 * np.array(heuristic_scores)).tolist()

        preds.extend([1 if c >= clf.threshold else 0 for c in confidences])
        batch_i += 1
        _print_batch_progress(batch_i, total_batches, every=10)

    elapsed = time.time() - start
    metrics = _binary_metrics(labels, preds)

    print(f"\nSamples: {len(texts)}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"FP / FN:           {int(metrics['fp'])} / {int(metrics['fn'])}")
    print(f"Time: {elapsed:.1f}s")

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": "ucirvine/sms_spam (train split)",
        "samples": int(metrics["support"]),
        "metrics": metrics,
        "time_seconds": elapsed,
        "notes": [
            "Positive label = spam (1).",
            "Evaluates the shipped LocalMod spam classifier (ML + heuristics).",
        ],
    }


def run_prompt_injection_benchmark(
    max_samples: int = None,
    device: str = "cpu",
    batch_size: int = 64,
    threshold: float = 0.5,
) -> Dict:
    """
    Benchmark LocalMod prompt injection detector on a dedicated injection dataset.

    Uses `S-Labs/prompt-injection-dataset` (binary: safe=0, injection=1).
    """
    if PromptInjectionDetector is None:
        raise RuntimeError("LocalMod is not importable. Run with: PYTHONPATH=src python evaluation/chi2025_benchmark.py ...")

    print("=" * 70)
    print("Benchmark - LocalMod Prompt Injection Detection (S-Labs/prompt-injection-dataset)")
    print("=" * 70)

    ds = load_dataset("S-Labs/prompt-injection-dataset")
    # Tune on validation, report on held-out test (more benchmark-like).
    tune_ds = ds["validation"]
    test_ds = ds["test"]

    # Keep tuning reasonably fast on CPU.
    tune_cap = 2000 if max_samples is None else int(max_samples)
    test_cap = None if max_samples is None else int(max_samples)

    if tune_cap:
        tune_ds = tune_ds.select(range(min(tune_cap, len(tune_ds))))
    if test_cap:
        test_ds = test_ds.select(range(min(test_cap, len(test_ds))))

    detector = PromptInjectionDetector(device=device, threshold=threshold, use_ml_model=True)
    detector.load()

    def _confidences_for(texts: List[str]) -> List[float]:
        """Compute detector confidence per text (mirrors detector.predict())."""
        confs: List[float] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for bi, i in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[i : i + batch_size]

            pattern_scores = []
            for t in batch:
                matches = detector._check_patterns(t)  # type: ignore[attr-defined]
                pattern_scores.append(min(len(matches) * 0.3, 0.9) if matches else 0.0)

            if detector.use_ml_model and detector._tokenizer is not None and detector._model is not None:
                inputs = detector._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = detector._model(**inputs)  # type: ignore[misc]
                ml_probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
                # Match PromptInjectionDetector.predict() combination:
                # confidence = max(ml_score, pattern_score); +0.1 boost if pattern_score>0 and ml_score>0.3
                cs = [max(float(p), float(ps)) for p, ps in zip(ml_probs, pattern_scores)]
                cs = [
                    min(c + 0.1, 1.0) if (ps > 0 and float(p) > 0.3) else c
                    for c, p, ps in zip(cs, ml_probs, pattern_scores)
                ]
                confs.extend(cs)
            else:
                confs.extend(pattern_scores)

            _print_batch_progress(bi, total_batches, every=10)
        return confs

    def _best_threshold_by_f1(y_true: List[int], scores: List[float]) -> Tuple[float, Dict[str, float]]:
        # Reasonable grid for moderation thresholds.
        grid = [round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()]
        best_t = grid[0]
        best_m = _binary_metrics(y_true, [1 if s >= best_t else 0 for s in scores])
        for t in grid[1:]:
            m = _binary_metrics(y_true, [1 if s >= t else 0 for s in scores])
            # Primary: maximize F1. Tie-breaker: higher precision.
            if (m["f1_score"] > best_m["f1_score"]) or (
                m["f1_score"] == best_m["f1_score"] and m["precision"] > best_m["precision"]
            ):
                best_t, best_m = t, m
        return float(best_t), best_m

    tune_texts = [x["text"] for x in tune_ds]
    tune_labels = [int(x["label"]) for x in tune_ds]
    test_texts = [x["text"] for x in test_ds]
    test_labels = [int(x["label"]) for x in test_ds]

    start_all = time.time()
    tune_scores = _confidences_for(tune_texts)
    tuned_threshold, tuned_metrics = _best_threshold_by_f1(tune_labels, tune_scores)

    print("\nTuning (validation split)")
    print(f"  best_threshold:    {tuned_threshold:.2f}")
    print(f"  Balanced Accuracy: {tuned_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {tuned_metrics['precision']:.4f}")
    print(f"  Recall:            {tuned_metrics['recall']:.4f}")
    print(f"  F1 Score:          {tuned_metrics['f1_score']:.4f}")

    test_scores = _confidences_for(test_texts)
    # If threshold <= 0, auto-select (tuned). Otherwise use provided threshold.
    use_t = tuned_threshold if float(threshold) <= 0 else float(threshold)
    preds = [1 if s >= use_t else 0 for s in test_scores]
    metrics = _binary_metrics(test_labels, preds)

    elapsed_all = time.time() - start_all
    print(f"\nEvaluation (test split, threshold={use_t:.2f})")
    print(f"Samples: {len(test_texts)}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"FP / FN:           {int(metrics['fp'])} / {int(metrics['fn'])}")
    print(f"Time: {elapsed_all:.1f}s")

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": "S-Labs/prompt-injection-dataset (tune=validation[:cap], eval=test)",
        "threshold": float(use_t),
        "tuned_threshold": float(tuned_threshold),
        "tune_samples": len(tune_texts),
        "tune_metrics": tuned_metrics,
        "samples": int(metrics["support"]),
        "metrics": metrics,
        "time_seconds": elapsed_all,
        "notes": [
            "Positive label = injection (1).",
            "Evaluates LocalMod PromptInjectionDetector (pattern + ML).",
        ],
    }


def run_nsfw_text_benchmark(
    max_samples: int = 2000,
    device: str = "cpu",
    batch_size: int = 64,
    threshold: float = 0.5,
) -> Dict:
    """
    Benchmark LocalMod NSFW text classifier on a labeled NSFW vs safe dataset.

    Uses `eliasalbouzidi/NSFW-Safe-Dataset` (binary: safe vs nsfw).
    """
    if NSFWClassifier is None:
        raise RuntimeError("LocalMod is not importable. Run with: PYTHONPATH=src python evaluation/chi2025_benchmark.py ...")

    print("=" * 70)
    print("Benchmark - LocalMod NSFW Text Detection (NSFW-Safe-Dataset)")
    print("=" * 70)

    # Use streaming to avoid long dataset materialization on CPU boxes.
    from itertools import islice
    from datasets.exceptions import DatasetNotFoundError

    n = int(max_samples) if max_samples else 2000
    texts: List[str] = []
    labels: List[int] = []
    dataset_name = "eliasalbouzidi/NSFW-Safe-Dataset (train, streaming)"

    try:
        ds_stream = load_dataset("eliasalbouzidi/NSFW-Safe-Dataset", split="train", streaming=True)
        for row in islice(ds_stream, n):
            # Common conventions: text field + label field (string or int).
            text = (row.get("text") or row.get("sentence") or row.get("content") or "")[:2000]
            lab = row.get("label") or row.get("labels") or row.get("class") or row.get("category")
            if not text:
                continue

            # Map labels to {0,1}: 1=nsfw, 0=safe.
            y = None
            if isinstance(lab, str):
                y = 1 if lab.strip().lower() in ("nsfw", "adult", "unsafe", "sexual") else 0
            elif isinstance(lab, (int, np.integer)):
                y = int(lab)
            if y is None:
                continue
            texts.append(text)
            labels.append(int(y))
    except DatasetNotFoundError:
        # Some "safety" datasets are gated due to sensitive content. Fall back to an open proxy benchmark.
        dataset_name = "Proxy: Maxx0/Texting_sex (NSFW) vs ag_news (SFW), streaming"
        pos_stream = load_dataset("Maxx0/Texting_sex", split="train", streaming=True)
        neg_stream = load_dataset("ag_news", split="test", streaming=True)

        # Collect balanced positives/negatives.
        n_pos = n // 2
        n_neg = n - n_pos

        for row in islice(pos_stream, n_pos):
            text = (row.get("text") or row.get("content") or row.get("prompt") or "")[:2000]
            if text:
                texts.append(text)
                labels.append(1)
        for row in islice(neg_stream, n_neg):
            text = (row.get("text") or "")[:2000]
            if text:
                texts.append(text)
                labels.append(0)

    if not texts:
        raise RuntimeError("Failed to load any labeled NSFW text samples from dataset(s).")

    clf = NSFWClassifier(device=device, threshold=threshold)
    clf.load()

    # Compute confidences once, then (optionally) tune threshold.
    scores: List[float] = []
    start = time.time()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        batch_res = clf.predict_batch(batch)
        scores.extend([float(r.confidence) for r in batch_res])
        _print_batch_progress(bi, total_batches, every=10)

    def _best_threshold_by_f1(y_true: List[int], s: List[float]) -> Tuple[float, Dict[str, float]]:
        grid = [round(x, 2) for x in np.linspace(0.05, 0.95, 19).tolist()]
        best_t = grid[0]
        best_m = _binary_metrics(y_true, [1 if x >= best_t else 0 for x in s])
        for t in grid[1:]:
            m = _binary_metrics(y_true, [1 if x >= t else 0 for x in s])
            if (m["f1_score"] > best_m["f1_score"]) or (
                m["f1_score"] == best_m["f1_score"] and m["precision"] > best_m["precision"]
            ):
                best_t, best_m = t, m
        return float(best_t), best_m

    # Deterministic 50/50 split for threshold tuning vs evaluation.
    rng = np.random.RandomState(42)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    split = len(idx) // 2
    tune_idx = idx[:split].tolist()
    test_idx = idx[split:].tolist()

    tune_labels = [labels[i] for i in tune_idx]
    tune_scores = [scores[i] for i in tune_idx]
    tuned_threshold, tuned_metrics = _best_threshold_by_f1(tune_labels, tune_scores)

    use_t = tuned_threshold if float(threshold) <= 0 else float(threshold)
    test_labels = [labels[i] for i in test_idx]
    test_scores = [scores[i] for i in test_idx]
    preds = [1 if s >= use_t else 0 for s in test_scores]
    metrics = _binary_metrics(test_labels, preds)
    elapsed = time.time() - start

    pos = int(sum(labels))
    print(f"\nSamples: {len(texts)} (pos={pos}, neg={len(labels) - pos})")
    print("\nTuning (50% split)")
    print(f"  best_threshold:    {tuned_threshold:.2f}")
    print(f"  Balanced Accuracy: {tuned_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {tuned_metrics['precision']:.4f}")
    print(f"  Recall:            {tuned_metrics['recall']:.4f}")
    print(f"  F1 Score:          {tuned_metrics['f1_score']:.4f}")
    print(f"\nEvaluation (held-out 50%, threshold={use_t:.2f})")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1_score']:.4f}")
    print(f"  FP / FN:           {int(metrics['fp'])} / {int(metrics['fn'])}")
    print(f"Time: {elapsed:.1f}s")

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "samples": int(len(texts)),
        "threshold": float(use_t),
        "tuned_threshold": float(tuned_threshold),
        "tune_samples": len(tune_idx),
        "tune_metrics": tuned_metrics,
        "metrics": metrics,
        "time_seconds": elapsed,
        "notes": [
            "Positive label = NSFW (1).",
            "If threshold <= 0, uses an auto-tuned threshold from a deterministic 50/50 split.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run LocalMod benchmarks (toxicity + others)")
    parser.add_argument(
        "--task",
        default="toxicity",
        choices=["toxicity", "spam", "prompt_injection", "nsfw_text", "nsfw_image", "pii", "all"],
        help="Which benchmark to run (default: toxicity).",
    )
    parser.add_argument("--samples", type=int, default=None, help="Max samples per dataset/subset")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for model inference")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for applicable benchmarks (default: 0.5).",
    )
    parser.add_argument("--output", default=None, help="Output JSON file (writes combined results)")
    args = parser.parse_args()
    
    device = _device_for_arg(args.device)
    out: Dict[str, Dict] = {"timestamp": datetime.now().isoformat(), "device": device}

    if args.task in ("toxicity", "all"):
        out["toxicity"] = run_chi2025_toxicity_benchmark(
            max_samples=args.samples, device=device, batch_size=args.batch_size
        )
    if args.task in ("spam", "all"):
        out["spam"] = run_uci_sms_spam_benchmark(
            max_samples=args.samples, device=device, batch_size=args.batch_size
        )
    if args.task in ("prompt_injection", "all"):
        out["prompt_injection"] = run_prompt_injection_benchmark(
            max_samples=args.samples, device=device, batch_size=args.batch_size, threshold=args.threshold
        )
    if args.task in ("nsfw_text", "all"):
        # Default to a modest sample size to keep this benchmark quick.
        nsfw_n = args.samples if args.samples is not None else 2000
        out["nsfw_text"] = run_nsfw_text_benchmark(
            max_samples=nsfw_n, device=device, batch_size=args.batch_size, threshold=args.threshold
        )
    if args.task in ("nsfw_image", "all"):
        img_n = args.samples if args.samples is not None else 500
        out["nsfw_image"] = run_nsfw_image_benchmark(
            max_samples=img_n, device=device, batch_size=min(args.batch_size, 16), threshold=args.threshold
        )
    if args.task in ("pii", "all"):
        pii_n = args.samples if args.samples is not None else 2000
        out["pii"] = run_pii_benchmark(max_samples=pii_n)
    
    if args.output and out:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
