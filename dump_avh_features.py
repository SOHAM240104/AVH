"""
Dump AV-HuBERT audio+visual features for multiple videos.

This is the foundation for the “feature-space adversarial GAN” training:
you can stop/resume without re-running slow AV-HuBERT extraction.

Inputs:
- Single video via --video_path OR a manifest CSV via --manifest_csv.

Outputs per sample:
- <out_dir>/<clip_id>.npz with arrays: audio, visual
- <out_dir>/<clip_id>.json with metadata: label, language, video_path, etc.

Example:
  conda activate avh
  python AVH/dump_avh_features.py \
    --video_path /path/to/video.mp4 \
    --out_dir /tmp/avh_features \
    --label REAL --language en --device cpu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


def _hash_clip(video_path: str) -> str:
    return hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:12]


def parse_label(label: Any) -> int:
    if isinstance(label, (int, np.integer)):
        return int(label)
    s = str(label).strip().lower()
    if s in {"real", "real_audio", "genuine", "genuine_audio", "1"}:
        return 1
    if s in {"fake", "fake_audio", "synthetic", "0"}:
        return 0
    raise ValueError(f"Unrecognized label: {label!r} (expected REAL/FAKE or 1/0)")


def load_test_video_module():
    """
    Load AVH/test_video.py as a module so we can reuse its stage-1+2 functions.

    Note: AVH/test_video.py does sys.path + sys.argv + os.chdir work at import-time.
    We isolate this by using module loading (not package imports) so it doesn't affect callers.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_video_path = os.path.join(this_dir, "test_video.py")
    spec = importlib.util.spec_from_file_location("avh_test_video", test_video_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {test_video_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_manifest_rows(manifest_csv: str) -> Iterable[Dict[str, str]]:
    """
    Expect CSV with one of these formats:
    1) Header:
       video_path,label,language (language optional)
    2) No header (3 columns):
       video_path,label,language
       or (2 columns): video_path,label
    """
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        # Peek first line to decide header vs no header
        pos = f.tell()
        first = f.readline().strip()
        f.seek(pos)
        has_header = any(k in first.lower() for k in ["video_path", "label", "language"]) and "," in first

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        else:
            reader = csv.reader(f)
            for cols in reader:
                if len(cols) < 2:
                    continue
                row: Dict[str, str] = {"video_path": cols[0], "label": cols[1]}
                if len(cols) >= 3:
                    row["language"] = cols[2]
                yield row


def ensure_device(device_str: str):
    import torch

    d = (device_str or "").strip().lower()
    if d in {"", "cpu"}:
        return torch.device("cpu")
    if d in {"mps"}:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if d in {"cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device_str!r}")


def main():
    parser = argparse.ArgumentParser(description="Dump AV-HuBERT audio+visual features to .npz")
    parser.add_argument("--video_path", type=str, default=None, help="Single video path (.mp4)")
    parser.add_argument("--manifest_csv", type=str, default=None, help="CSV manifest with video_path,label[,language]")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .npz + .json")
    parser.add_argument("--label", type=str, default="REAL", help="Label for --video_path: REAL/FAKE or 1/0")
    parser.add_argument("--language", type=str, default="", help="Language tag for --video_path (optional)")
    parser.add_argument("--avhubert_ckpt", type=str, default=None, help="Path to self_large_vox_433h.pt")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--keep_roi", action="store_true", help="Copy mouth_roi.mp4 next to the .npz for inspection")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary preprocessing directories")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature dumps")
    parser.add_argument("--max_samples", type=int, default=0, help="Debug: stop after N samples (0=all)")

    args = parser.parse_args()

    if (args.video_path is None) == (args.manifest_csv is None):
        raise ValueError("Provide exactly one of --video_path or --manifest_csv")

    os.makedirs(args.out_dir, exist_ok=True)

    mod = load_test_video_module()
    import torch

    device = ensure_device(args.device)

    avhubert_ckpt = args.avhubert_ckpt or mod.AVHUBERT_CKPT

    # Load AV-HuBERT once (stage-2 extractor)
    print(f"[AVH dump] Loading AV-HuBERT from: {avhubert_ckpt}")
    avh_model, task = mod.load_avhubert(avhubert_ckpt, device)
    transform = mod.avhubert_utils.Compose([
        mod.avhubert_utils.Normalize(0.0, 255.0),
        mod.avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        mod.avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
    ])

    # Build input iterator
    if args.video_path is not None:
        rows = [{"video_path": args.video_path, "label": args.label, "language": args.language}]
    else:
        rows = iter_manifest_rows(args.manifest_csv)

    n = 0
    for row in rows:
        video_path = row["video_path"]
        clip_id = _hash_clip(video_path)
        out_npz = os.path.join(args.out_dir, f"{clip_id}.npz")
        out_meta = os.path.join(args.out_dir, f"{clip_id}.json")

        if (not args.overwrite) and os.path.isfile(out_npz) and os.path.isfile(out_meta):
            print(f"[AVH dump] Skip existing: {clip_id}")
            continue

        label_int = parse_label(row.get("label", args.label))
        language = (row.get("language", args.language) or "").strip()

        work_dir = tempfile.mkdtemp(prefix="avh_dump_")
        try:
            print(f"[AVH dump] Preprocess: {video_path}")
            roi_path, audio_path = mod.preprocess_video(video_path, work_dir)

            print(f"[AVH dump] Extract features: {clip_id}")
            f_audio, f_video = mod.extract_avhubert_features(
                avh_model, roi_path, audio_path, transform, device
            )

            f_audio = np.asarray(f_audio)
            f_video = np.asarray(f_video)

            if (not np.isfinite(f_audio).all()) or (not np.isfinite(f_video).all()):
                raise RuntimeError("Non-finite values found in extracted features")

            np.savez_compressed(out_npz, audio=f_audio, visual=f_video)
            meta: Dict[str, Any] = {
                "clip_id": clip_id,
                "video_path": video_path,
                "label": label_int,
                "language": language,
                "device": str(device),
                "keep_roi": bool(args.keep_roi),
            }

            if args.keep_roi and roi_path and os.path.isfile(roi_path):
                roi_out = os.path.join(args.out_dir, f"{clip_id}_mouth_roi.mp4")
                shutil.copy2(roi_path, roi_out)
                meta["roi_path_out"] = roi_out

            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"[AVH dump] Wrote: {out_npz}")
        finally:
            if not args.keep_temp:
                shutil.rmtree(work_dir, ignore_errors=True)

        n += 1
        if args.max_samples and n >= args.max_samples:
            break

    # Clean up model to free memory
    del avh_model
    torch.mps.empty_cache() if device.type == "mps" else None
    print(f"[AVH dump] Done. Dumped up to {n} samples into: {args.out_dir}")


if __name__ == "__main__":
    main()

