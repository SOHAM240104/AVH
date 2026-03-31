"""
Paper-style evaluation utilities for AVH experiments (AUC/AP + silence bias checks).

Implements the key reporting style from arXiv:2412.00175:
- Video-level AUC and AP (fake as positive class).
- Compare untrimmed vs trimmed variants.
- Leading-silence baseline classifier.

Usage examples:

1) Evaluate model predictions from CSV:
   python AVH/eval_metrics_avhalign_style.py \
     --pred_csv /path/to/preds.csv

   CSV format (required columns):
   - video_id
   - y_true           (0=real, 1=fake)
   - score_untrimmed  (higher => faker)
   Optional:
   - score_trimmed
   - dataset
   - method

2) Evaluate silence baseline directly from audio files:
   python AVH/eval_metrics_avhalign_style.py \
     --audio_csv /path/to/audio_eval.csv \
     --silence_threshold 5e-4

   audio_csv format:
   - video_id
   - y_true
   - audio_path
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import librosa


@dataclass
class MetricResult:
    auc: float
    ap: float
    n: int


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    # roc_auc_score fails when only one class is present
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def eval_scores(y_true: np.ndarray, scores: np.ndarray) -> MetricResult:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    return MetricResult(
        auc=_safe_auc(y, s),
        ap=_safe_ap(y, s),
        n=int(len(y)),
    )


def leading_silence_ms(audio: np.ndarray, sr: int, threshold: float = 5e-4) -> float:
    mag = np.abs(audio)
    idx = np.where(mag > threshold)[0]
    if len(idx) == 0:
        return 1000.0 * (len(audio) / float(sr))
    first = int(idx[0])
    return 1000.0 * (first / float(sr))


def max_amplitude_in_window(audio: np.ndarray, sr: int, window_ms: float = 30.0) -> float:
    n = int(sr * (window_ms / 1000.0))
    n = max(1, min(n, len(audio)))
    return float(np.max(np.abs(audio[:n])))


def evaluate_pred_csv(pred_csv: str) -> Dict:
    df = pd.read_csv(pred_csv)
    required = {"video_id", "y_true", "score_untrimmed"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in pred_csv: {sorted(miss)}")

    out: Dict[str, Dict] = {}
    y = df["y_true"].to_numpy()

    untrim = eval_scores(y, df["score_untrimmed"].to_numpy())
    out["untrimmed"] = {"auc": untrim.auc, "ap": untrim.ap, "n": untrim.n}

    if "score_trimmed" in df.columns:
        trim = eval_scores(y, df["score_trimmed"].to_numpy())
        out["trimmed"] = {"auc": trim.auc, "ap": trim.ap, "n": trim.n}
        out["delta_auc_trim_minus_untrim"] = trim.auc - untrim.auc
        out["delta_ap_trim_minus_untrim"] = trim.ap - untrim.ap

    # Optional grouped report, e.g., by dataset or method.
    for gcol in ["dataset", "method"]:
        if gcol in df.columns:
            grouped = {}
            for name, g in df.groupby(gcol):
                gy = g["y_true"].to_numpy()
                gu = eval_scores(gy, g["score_untrimmed"].to_numpy())
                entry = {"untrimmed": {"auc": gu.auc, "ap": gu.ap, "n": gu.n}}
                if "score_trimmed" in g.columns:
                    gt = eval_scores(gy, g["score_trimmed"].to_numpy())
                    entry["trimmed"] = {"auc": gt.auc, "ap": gt.ap, "n": gt.n}
                    entry["delta_auc_trim_minus_untrim"] = gt.auc - gu.auc
                    entry["delta_ap_trim_minus_untrim"] = gt.ap - gu.ap
                grouped[str(name)] = entry
            out[f"group_by_{gcol}"] = grouped

    return out


def evaluate_silence_audio_csv(
    audio_csv: str,
    silence_threshold: float = 5e-4,
    maxamp_window_ms: float = 30.0,
) -> Dict:
    df = pd.read_csv(audio_csv)
    required = {"video_id", "y_true", "audio_path"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in audio_csv: {sorted(miss)}")

    lead_ms = []
    maxamp = []
    y_true = df["y_true"].astype(int).to_numpy()

    for p in df["audio_path"].tolist():
        audio, sr = librosa.load(p, sr=None, mono=True)
        lead_ms.append(leading_silence_ms(audio, sr, threshold=silence_threshold))
        maxamp.append(max_amplitude_in_window(audio, sr, window_ms=maxamp_window_ms))

    lead_ms = np.asarray(lead_ms, dtype=np.float64)
    maxamp = np.asarray(maxamp, dtype=np.float64)

    # In the paper, fake tends to have longer leading silence,
    # and lower early-window max amplitude.
    lead_res = eval_scores(y_true, lead_ms)
    # For max-amplitude, invert sign so higher score => faker.
    maxamp_res = eval_scores(y_true, -maxamp)

    return {
        "silence_classifier": {
            "feature": "leading_silence_ms",
            "threshold_used_for_measurement": silence_threshold,
            "auc": lead_res.auc,
            "ap": lead_res.ap,
            "n": lead_res.n,
            "mean_leading_silence_ms_real": float(np.mean(lead_ms[y_true == 0])) if np.any(y_true == 0) else None,
            "mean_leading_silence_ms_fake": float(np.mean(lead_ms[y_true == 1])) if np.any(y_true == 1) else None,
        },
        "max_amplitude_baseline": {
            "feature": f"max_amplitude_first_{maxamp_window_ms:.1f}ms",
            "auc": maxamp_res.auc,
            "ap": maxamp_res.ap,
            "n": maxamp_res.n,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="AVH paper-style metrics (AUC/AP, trimmed, silence baseline)")
    parser.add_argument("--pred_csv", type=str, default=None, help="CSV with y_true + score_untrimmed (+ optional score_trimmed)")
    parser.add_argument("--audio_csv", type=str, default=None, help="CSV with y_true + audio_path for silence-bias baseline")
    parser.add_argument("--silence_threshold", type=float, default=5e-4)
    parser.add_argument("--maxamp_window_ms", type=float, default=30.0)
    parser.add_argument("--out_json", type=str, default="", help="Optional output json path")
    args = parser.parse_args()

    if not args.pred_csv and not args.audio_csv:
        raise ValueError("Provide at least one of --pred_csv or --audio_csv")

    report: Dict[str, Dict] = {}
    if args.pred_csv:
        report["prediction_metrics"] = evaluate_pred_csv(args.pred_csv)
    if args.audio_csv:
        report["silence_bias_metrics"] = evaluate_silence_audio_csv(
            args.audio_csv,
            silence_threshold=args.silence_threshold,
            maxamp_window_ms=args.maxamp_window_ms,
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()

