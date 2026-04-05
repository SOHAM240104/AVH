"""
Unsupervised AVH scoring (no training, no labels).

Algorithm (zero-shot):
1) Reuse AVH preprocessing + AV-HuBERT feature extraction.
2) Build lag profile between audio/video embeddings:
   for lag in [-max_lag, +max_lag], compute mean cosine similarity.
3) Compute mismatch evidence:
   - best lag distance from 0 (temporal offset evidence)
   - advantage of best lag over zero lag (misalignment evidence)
   - flat/weak peak penalty (low alignment confidence)

Higher score => more likely fake/misaligned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import numpy as np
import torch

# Reuse existing AVH pipeline utilities
sys.path.insert(0, os.path.dirname(__file__))
import test_video as avh


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def _mean_cosine_with_lag(v: np.ndarray, a: np.ndarray, lag: int) -> float:
    """
    v, a: (T, D)
    lag > 0 means audio is shifted forward relative to video.
    """
    T = min(v.shape[0], a.shape[0])
    if lag >= 0:
        v_seg = v[: T - lag]
        a_seg = a[lag:T]
    else:
        k = -lag
        v_seg = v[k:T]
        a_seg = a[: T - k]
    if len(v_seg) < 5:
        return -1.0
    c = np.sum(v_seg * a_seg, axis=1)
    return float(np.mean(c))


def unsupervised_score(visual_feats: np.ndarray, audio_feats: np.ndarray, max_lag: int = 12):
    v = _normalize_rows(np.asarray(visual_feats, dtype=np.float32))
    a = _normalize_rows(np.asarray(audio_feats, dtype=np.float32))

    lags = list(range(-max_lag, max_lag + 1))
    prof = np.array([_mean_cosine_with_lag(v, a, lag) for lag in lags], dtype=np.float32)

    idx_best = int(np.argmax(prof))
    best_lag = lags[idx_best]
    sim_best = float(prof[idx_best])
    sim_zero = float(prof[max_lag])  # lag=0
    sim_mean = float(np.mean(prof))

    # mismatch components
    lag_offset = abs(best_lag) / float(max_lag)           # 0..1
    nonzero_adv = max(0.0, sim_best - sim_zero)           # >0 when off-zero aligns better
    flat_peak = max(0.0, 1.0 - max(0.0, sim_best - sim_mean))  # higher when profile is weak/flat

    # weighted unsupervised fake evidence score
    score = 1.5 * lag_offset + 2.0 * nonzero_adv + 0.5 * flat_peak

    details = {
        "best_lag": int(best_lag),
        "sim_best": sim_best,
        "sim_zero": sim_zero,
        "sim_mean": sim_mean,
        "lag_offset": float(lag_offset),
        "nonzero_adv": float(nonzero_adv),
        "flat_peak": float(flat_peak),
    }
    return float(score), details


def main():
    p = argparse.ArgumentParser(description="Unsupervised AVH scoring (no training)")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--avhubert_ckpt", default=avh.AVHUBERT_CKPT, help="AV-HuBERT checkpoint")
    p.add_argument("--max_lag", type=int, default=12, help="Max lag in frames for sync profile")
    p.add_argument("--keep_temp", action="store_true", help="Keep temp preprocessing files")
    p.add_argument("--json_out", type=str, default=None, help="Optional path to write result JSON.")
    p.add_argument("--use_mps", action="store_true", help="Use Apple MPS if available")
    p.add_argument(
        "--smart_crop",
        type=str,
        default="auto",
        choices=["off", "auto", "reel", "face"],
        help="Spatial pre-crop before mouth ROI (reels / UI overlays).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    work_dir = tempfile.mkdtemp(prefix="avh_unsup_")
    print(f"Working directory: {work_dir}")
    print(f"Using device: {device}")
    json_payload = None
    try:
        roi_path, audio_path = avh.preprocess_video(args.video, work_dir, smart_crop=args.smart_crop)
        model, task = avh.load_avhubert(args.avhubert_ckpt, device)
        transform = avh.avhubert_utils.Compose([
            avh.avhubert_utils.Normalize(0.0, 255.0),
            avh.avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            avh.avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
        ])
        f_audio, f_video = avh.extract_avhubert_features(model, roi_path, audio_path, transform, device)
        score, d = unsupervised_score(f_video, f_audio, max_lag=args.max_lag)

        json_payload = {
            "success": True,
            "score": float(score),
            "audio_path": audio_path if args.keep_temp else None,
            "roi_path": roi_path if args.keep_temp else None,
            "details": d,
        }
        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(json_payload, f)

        print("=" * 60)
        print(f"UNSUPERVISED SCORE: {score:.4f}")
        print(f"best_lag={d['best_lag']} sim_zero={d['sim_zero']:.4f} sim_best={d['sim_best']:.4f}")
        print("Higher score = more likely lip-audio mismatch")
        print("=" * 60)
    except Exception as e:
        json_payload = {"success": False, "error": str(e)}
        if args.json_out:
            try:
                os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(json_payload, f)
            except Exception:
                pass
        raise
    finally:
        if not args.keep_temp:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

