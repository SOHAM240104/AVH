"""
Feature-space adversary training (GAN-style hard misalignment) for AVH-Align.

This consumes dumped AV-HuBERT features:
  <features_dir>/<clip_id>.npz with arrays: audio, visual
  <features_dir>/<clip_id>.json with metadata: label (REAL/FAKE or 1/0), language (optional)

Training loop (proof-of-concept-friendly):
- Freeze FusionModel parameters, but keep gradients to inputs so G can learn.
- Generator perturbs visual (mouth-ROI) features in embedding space: visual_adv = visual + delta
- Discriminator tries to distinguish natural aligned pairs vs adversarially perturbed pairs
- Generator is also pushed to create “hard” score changes:
    For REAL (label=1): increase AVH score on adversarial features
    For FAKE (label=0): decrease AVH score on adversarial features

Stop/Resume:
- Saves checkpoints periodically + always updates latest checkpoint.
- Resume loads model/optimizer states + step counters.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure we can import AVH/model.py when running from repo root.
AVH_DIR = os.path.dirname(os.path.abspath(__file__))
if AVH_DIR not in sys.path:
    sys.path.insert(0, AVH_DIR)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label_from_meta(meta: Dict) -> int:
    # meta["label"] is expected to be 1 for REAL and 0 for FAKE
    lab = meta.get("label", None)
    if lab is None:
        raise ValueError("Missing label in meta json")
    if isinstance(lab, (int, np.integer)):
        return int(lab)
    s = str(lab).strip().lower()
    if s in {"real", "genuine", "1"}:
        return 1
    if s in {"fake", "synthetic", "0"}:
        return 0
    return int(lab)


def compute_score(fusion_model, visual_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
    """
    visual_feats, audio_feats: shape (T, 1024)
    Returns scalar tensor score (higher => more likely fake).
    """
    # Match AVH/test_video.py: normalize embeddings before FusionModel.
    visual_tensor = visual_feats / torch.linalg.norm(visual_feats, ord=2, dim=-1, keepdim=True)
    audio_tensor = audio_feats / torch.linalg.norm(audio_feats, ord=2, dim=-1, keepdim=True)

    # FusionModel expects last dim to match 1024; T dimension is treated as batch-like.
    output = fusion_model(visual_tensor, audio_tensor)  # (T, 1)
    score = torch.logsumexp(-output, dim=0).squeeze()  # scalar
    return score


class VisualPerturber(nn.Module):
    """
    Generator G: learns a per-time-step delta for visual embeddings.
    """

    def __init__(self, dim: int = 1024, hidden: int = 512, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = float(epsilon)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, audio_feats: torch.Tensor, visual_feats: torch.Tensor) -> torch.Tensor:
        # Inputs: (T, 1024)
        x = torch.cat([audio_feats, visual_feats], dim=-1)  # (T, 2048)
        delta_raw = self.net(x)  # (T, 1024)
        delta = self.epsilon * torch.tanh(delta_raw / (self.epsilon + 1e-8))
        return delta


class PairCritic(nn.Module):
    """
    Discriminator D (critic): outputs a single logit for (audio, visual) pair realism.
    """

    def __init__(self, dim: int = 1024, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, audio_feats: torch.Tensor, visual_feats: torch.Tensor) -> torch.Tensor:
        # (T, 1024) -> apply per-step then mean pool
        x = torch.cat([audio_feats, visual_feats], dim=-1)  # (T, 2048)
        logits = self.net(x)  # (T, 1)
        return logits.mean(dim=0).squeeze()  # scalar


@dataclass
class TrainConfig:
    features_dir: str
    fusion_ckpt: str
    out_dir: str
    device: str = "cpu"
    epochs: int = 3
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    epsilon: float = 0.05
    lambda_hard: float = 1.0
    lambda_reg: float = 0.05
    max_steps: int = 0  # 0 = all steps
    disc_steps: int = 1
    gen_steps: int = 1
    save_every_steps: int = 50
    debug_eval_every_steps: int = 25
    batch_size: int = 1  # variable-length features: keep batch_size=1 for now
    languages: Optional[List[str]] = None
    overwrite_latest: bool = True
    seed: int = 42
    resume_checkpoint: Optional[str] = None


def load_samples(features_dir: str, languages: Optional[List[str]] = None) -> List[Dict]:
    npz_paths = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz feature dumps found in: {features_dir}")

    samples: List[Dict] = []
    for npz_path in npz_paths:
        clip_id = os.path.basename(npz_path).replace(".npz", "")
        json_path = os.path.join(features_dir, f"{clip_id}.json")
        if not os.path.isfile(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if languages:
            lang = (meta.get("language", "") or "").strip()
            if lang not in languages:
                continue

        meta["npz_path"] = npz_path
        meta["clip_id"] = clip_id
        samples.append(meta)

    if not samples:
        raise RuntimeError("No samples matched the given filters (languages?).")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train feature-space adversary on dumped AVH features")
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--fusion_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--lambda_hard", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.05)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--save_every_steps", type=int, default=50)
    parser.add_argument("--debug_eval_every_steps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--languages", type=str, default="", help="Comma-separated language tags (optional)")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to a saved checkpoint to resume")
    parser.add_argument("--disc_steps", type=int, default=1, help="How many D steps per sample iteration")
    parser.add_argument("--gen_steps", type=int, default=1, help="How many G steps per sample iteration")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = args.device.strip().lower()
    if device == "cpu":
        dev = torch.device("cpu")
    elif device == "mps":
        dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    languages = [s.strip() for s in args.languages.split(",") if s.strip()] if args.languages else None
    samples = load_samples(args.features_dir, languages=languages)

    # Load FusionModel (frozen params, but keep gradients wrt inputs).
    from model import FusionModel  # local import within AVH context
    ckpt = torch.load(args.fusion_ckpt, map_location=dev, weights_only=False)
    fusion_model = FusionModel().to(dev)
    fusion_model.load_state_dict(ckpt["state_dict"])
    fusion_model.eval()
    for p in fusion_model.parameters():
        p.requires_grad_(False)

    G = VisualPerturber(dim=1024, hidden=512, epsilon=args.epsilon).to(dev)
    D = PairCritic(dim=1024, hidden=512).to(dev)

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr_g)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d)

    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location=dev)
        G.load_state_dict(state["G_state"])
        D.load_state_dict(state["D_state"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("step", 0))
        print(f"[Adversary train] Resumed from: {args.resume_checkpoint} (epoch={start_epoch}, step={global_step})")

    # Training loop
    best_debug = None
    for epoch in range(start_epoch, args.epochs):
        random.shuffle(samples)
        for s in samples:
            if args.max_steps and global_step >= args.max_steps:
                break

            npz_path = s["npz_path"]
            meta_label = parse_label_from_meta(s)
            # load features
            data = np.load(npz_path, allow_pickle=True)
            audio = torch.from_numpy(data["audio"]).float().to(dev)  # (T, 1024)
            visual = torch.from_numpy(data["visual"]).float().to(dev)  # (T, 1024)

            T = min(audio.shape[0], visual.shape[0])
            audio_t = audio[:T]
            visual_t = visual[:T]

            # --- Discriminator step(s)
            for _ in range(args.disc_steps):
                delta = G(audio_t, visual_t)
                visual_adv = visual_t + delta

                # Critic logits
                d_real = D(audio_t, visual_t)
                d_fake = D(audio_t, visual_adv.detach())

                # Hinge GAN loss
                loss_d = F.relu(1.0 - d_real) + F.relu(1.0 + d_fake)

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            # --- Generator step(s)
            for _ in range(args.gen_steps):
                delta = G(audio_t, visual_t)
                visual_adv = visual_t + delta

                d_fake_for_g = D(audio_t, visual_adv)
                loss_g_gan = -d_fake_for_g

                # Hard misalignment loss using frozen FusionModel score
                score_real = compute_score(fusion_model, visual_t, audio_t)
                score_adv = compute_score(fusion_model, visual_adv, audio_t)

                # label: REAL=1, FAKE=0
                target_direction = 1.0 if meta_label == 1 else -1.0
                # For REAL, want score_adv > score_real. For FAKE, want score_adv < score_real.
                loss_hard = -(target_direction * (score_adv - score_real))

                # Regularize perturbation magnitude
                loss_reg = (delta.pow(2).mean())

                loss_g = loss_g_gan + args.lambda_hard * loss_hard + args.lambda_reg * loss_reg

                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()

            # Debug eval
            if (global_step % args.debug_eval_every_steps) == 0:
                with torch.no_grad():
                    delta_dbg = G(audio_t, visual_t)
                    visual_adv_dbg = visual_t + delta_dbg
                    score_real_dbg = compute_score(fusion_model, visual_t, audio_t).item()
                    score_adv_dbg = compute_score(fusion_model, visual_adv_dbg, audio_t).item()
                    delta_mag = float(delta_dbg.pow(2).mean().sqrt().item())

                debug = {
                    "epoch": epoch,
                    "step": global_step,
                    "clip_id": s.get("clip_id", ""),
                    "label": meta_label,
                    "score_real": score_real_dbg,
                    "score_adv": score_adv_dbg,
                    "score_delta": score_adv_dbg - score_real_dbg,
                    "delta_rmse": delta_mag,
                    "loss_d": float(loss_d.item()),
                    "loss_g": float(loss_g.item()),
                }
                print("[Adversary train][debug]", json.dumps(debug, indent=2))

                # Track best: maximize “hardness” direction
                val = debug["score_delta"] * (1.0 if meta_label == 1 else -1.0)
                if best_debug is None or val > best_debug:
                    best_debug = val

            # Save checkpoints
            if (global_step % args.save_every_steps) == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step:06d}.pt")
                state = {
                    "G_state": G.state_dict(),
                    "D_state": D.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_debug": best_debug,
                    "config": vars(args),
                }
                torch.save(state, ckpt_path)
                # latest symlink-like copy (plain copy for portability)
                torch.save(state, os.path.join(ckpt_dir, "latest.pt"))

            global_step += 1

        if args.max_steps and global_step >= args.max_steps:
            break

    # Final save
    final_state = {
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "best_debug": best_debug,
        "config": vars(args),
    }
    torch.save(final_state, os.path.join(ckpt_dir, "final.pt"))
    print(f"[Adversary train] Done. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()

