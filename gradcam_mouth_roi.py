"""
Visual-only Grad-CAM for the mouth-ROI CNN path in AV-HuBERT.

Evidence output:
- Saves top-K heatmap overlay frames computed from gradients of the FusionModel
  deepfake score backpropagated through the AV-HuBERT visual encoder.

This is intentionally “visual-only CAM” (no transformer token CAM, no audio CAM)
to keep it robust and panel-friendly.

Example:
  conda activate avh
  python AVH/gradcam_mouth_roi.py \
    --video_path /path/to/video.mp4 \
    --out_dir /tmp/gradcam_out \
    --top_k 5 \
    --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from typing import Optional, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_test_video_module():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    test_video_path = os.path.join(this_dir, "test_video.py")
    spec = importlib.util.spec_from_file_location("avh_test_video", test_video_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {test_video_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_video_gray_frames(video_path: str):
    """
    Returns frames as a list of uint8 grayscale images (H,W).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM mouth ROI evidence (visual-only)")
    parser.add_argument("--video_path", type=str, default=None, help="Path to MP4 video (optional if --roi_path/--audio_path provided)")
    parser.add_argument("--avhubert_ckpt", type=str, default=None, help="Path to self_large_vox_433h.pt")
    parser.add_argument("--fusion_ckpt", type=str, default=None, help="Path to AVH-Align fusion ckpt")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for overlay frames")
    parser.add_argument("--roi_path", type=str, default=None, help="Optional: mouth ROI MP4 produced by preprocess_video (real-time reuse)")
    parser.add_argument("--audio_path", type=str, default=None, help="Optional: audio WAV produced by preprocess_video (real-time reuse)")
    parser.add_argument("--top_k", type=int, default=5, help="Save top-K frames with highest CAM intensity")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--keep_temp", action="store_true", help="Keep preprocess temporary dirs")
    parser.add_argument("--overwrite", action="store_true", help="Recompute CAM even if outputs exist")
    parser.add_argument("--adv_ckpt", type=str, default=None, help="Optional: adversary checkpoint (.pt) for robustness delta")
    parser.add_argument("--adv_epsilon", type=float, default=None, help="Optional: override adversary epsilon (perturbation strength)")
    parser.add_argument("--capture_attention", action="store_true", help="Best-effort: capture transformer attention weights (adds overhead).")
    args = parser.parse_args()

    mod = load_test_video_module()
    # Defaults from test_video.py constants
    avhubert_ckpt = args.avhubert_ckpt or mod.AVHUBERT_CKPT
    fusion_ckpt = args.fusion_ckpt or mod.FUSION_CKPT

    device_str = (args.device or "").strip().lower()
    if device_str == "cpu":
        device = torch.device("cpu")
    elif device_str == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.out_dir)
    overlay_dir = os.path.join(args.out_dir, "overlays")
    ensure_dir(overlay_dir)
    index_path = os.path.join(args.out_dir, "index.json")
    cam_volume_path = os.path.join(args.out_dir, "cam_volume.npy")
    # Simple cache: if index exists and not overwrite, skip (overlays are assumed present).
    if not args.overwrite and os.path.isfile(index_path):
        # If the CAM volume is missing, fall through to recompute so
        # downstream explainability can run reliably.
        if os.path.isfile(cam_volume_path):
            if args.adv_ckpt:
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        idx = json.load(f)
                    if idx.get("adv_score") is not None:
                        print(f"[Grad-CAM] Found existing robustness delta in: {args.out_dir}. Use --overwrite to recompute.")
                        return
                except Exception:
                    pass
            else:
                print(f"[Grad-CAM] Found existing index.json in: {args.out_dir}. Use --overwrite to recompute.")
                return
        # else: cam_volume is missing -> recompute

    # Load models
    print(f"[Grad-CAM] Loading AV-HuBERT: {avhubert_ckpt}")
    avh_model, task = mod.load_avhubert(avhubert_ckpt, device)
    print(f"[Grad-CAM] Loading FusionModel: {fusion_ckpt}")
    ckpt = torch.load(fusion_ckpt, map_location=device, weights_only=False)
    fusion_model = mod.FusionModel().to(device)
    fusion_model.load_state_dict(ckpt["state_dict"])
    fusion_model.eval()
    for p in fusion_model.parameters():
        p.requires_grad_(False)

    transform = mod.avhubert_utils.Compose([
        mod.avhubert_utils.Normalize(0.0, 255.0),
        mod.avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        mod.avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
    ])

    # Preprocess video (mouth ROI + audio), unless user already provided them.
    created_work_dir = False
    work_dir = None
    roi_path = args.roi_path
    audio_path = args.audio_path
    if not roi_path or not audio_path:
        if not args.video_path:
            raise ValueError("Provide either --video_path OR both --roi_path and --audio_path.")
        work_dir = tempfile.mkdtemp(prefix="avh_gradcam_")
        created_work_dir = True
        print(f"[Grad-CAM] Working directory: {work_dir}")
        roi_path, audio_path = mod.preprocess_video(args.video_path, work_dir)
    else:
        # Ensure the provided paths exist before proceeding.
        if not os.path.isfile(roi_path):
            raise FileNotFoundError(f"[Grad-CAM] --roi_path not found: {roi_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"[Grad-CAM] --audio_path not found: {audio_path}")

    # Read ROI frames for overlay
    roi_frames_gray = read_video_gray_frames(roi_path)
    if len(roi_frames_gray) == 0:
        raise RuntimeError(f"[Grad-CAM] Failed to read mouth ROI frames from: {roi_path}")

    # Load visual input frames from the same ROI video used for AV-HuBERT
    frames = mod.avhubert_utils.load_video(roi_path)
    frames = transform(frames)
    frames_t = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,H,W)

    # Audio features (constant for CAM; no_grad to save memory)
    with torch.no_grad():
        audio_feats = mod.load_audio_features(audio_path)  # (Ta, 4*dim)
        audio = audio_feats[None, :, :].transpose(1, 2).to(device)  # (1, F, Ta)

    # Align lengths like extract_avhubert_features does
    # For visual-only CAM, we only need visual activations; audio can be sliced.
    with torch.no_grad():
        # Extract audio features (no grad)
        f_audio, _ = avh_model.extract_finetune({"video": None, "audio": audio}, None, None)
        f_audio = f_audio.squeeze(0)  # (Ta', 1024)

    # --- Grad-CAM hooks on the last visual conv block ---
    target_module = None
    # Expected: avh_model.feature_extractor_video.resnet.trunk.layer4
    if hasattr(avh_model, "feature_extractor_video"):
        sub = avh_model.feature_extractor_video
        if hasattr(sub, "resnet") and sub.resnet is not None and hasattr(sub.resnet, "trunk"):
            if hasattr(sub.resnet.trunk, "layer4"):
                target_module = sub.resnet.trunk.layer4
    if target_module is None:
        raise RuntimeError("Could not locate the target conv layer for Grad-CAM. "
                           "Expected: feature_extractor_video.resnet.trunk.layer4")

    activations = {"value": None}
    gradients = {"value": None}

    def fwd_hook(_module, _inp, out):
        activations["value"] = out

    def bwd_hook(_module, _grad_in, grad_out):
        # grad_out[0] corresponds to gradients w.r.t. outputs of the hooked layer
        gradients["value"] = grad_out[0]

    handle_fwd = target_module.register_forward_hook(fwd_hook)
    handle_bwd = target_module.register_full_backward_hook(bwd_hook)

    # Extract visual features with gradients
    attention_per_time = None
    if args.capture_attention:
        attention_weights: list[torch.Tensor] = []
        patched: list[tuple[torch.nn.Module, Any]] = []

        # Best-effort: wrap attention modules to request weights and collect them.
        import inspect

        def _is_mha(mod: torch.nn.Module) -> bool:
            cls = mod.__class__.__name__.lower()
            return "multiheadattention" in cls or "multi_head_attention" in cls

        for m in avh_model.modules():
            if not _is_mha(m):
                continue
            orig_forward = m.forward
            try:
                sig = inspect.signature(orig_forward)
                accepts_need_weights = "need_weights" in sig.parameters
            except Exception:
                accepts_need_weights = True

            def _make_wrapped_forward(mod_ref: torch.nn.Module, orig_fwd, accepts_kw):
                def wrapped_forward(*f_args, **f_kwargs):
                    if accepts_kw:
                        try:
                            f_kwargs["need_weights"] = True
                        except Exception:
                            pass
                    try:
                        out = orig_fwd(*f_args, **f_kwargs)
                    except TypeError:
                        # If the underlying forward doesn't support need_weights, try without it.
                        f_kwargs.pop("need_weights", None)
                        out = orig_fwd(*f_args, **f_kwargs)

                    weights = None
                    if isinstance(out, tuple) and len(out) >= 2:
                        weights = out[1]
                    if torch.is_tensor(weights):
                        try:
                            attention_weights.append(weights.detach())
                        except Exception:
                            pass
                    return out

                return wrapped_forward

            m.forward = _make_wrapped_forward(m, orig_forward, accepts_need_weights)  # type: ignore[assignment]
            patched.append((m, orig_forward))

        try:
            f_video, _ = avh_model.extract_finetune({"video": frames_t, "audio": None}, None, None)
        finally:
            # Restore original forwards
            for m, orig_forward in patched:
                m.forward = orig_forward  # type: ignore[assignment]

        # Summarize captured attention weights into a per-time trace.
        # We average over heads (if present) and source tokens.
        per_time = []
        for w in attention_weights:
            if not torch.is_tensor(w):
                continue
            if w.ndim == 4:
                # (B, heads, T_tgt, T_src)
                per_t = w.mean(dim=(0, 1, 3))  # (T_tgt,)
            elif w.ndim == 3:
                # (B, T_tgt, T_src)
                per_t = w.mean(dim=(0, 2))  # (T_tgt,)
            elif w.ndim == 2:
                # (T_tgt, T_src)
                per_t = w.mean(dim=1)
            else:
                continue
            per_time.append(per_t.detach().cpu().float().numpy())

        if per_time:
            # Use the shortest captured length to avoid alignment bugs.
            T_min = min(x.shape[0] for x in per_time)
            stack = np.stack([x[:T_min] for x in per_time], axis=0)
            attention_per_time = stack.mean(axis=0)
    else:
        f_video, _ = avh_model.extract_finetune({"video": frames_t, "audio": None}, None, None)
    f_video = f_video.squeeze(0)  # (Tv', 1024)

    # Align time length (min over extracted audio and visual features)
    min_len = min(f_video.shape[0], f_audio.shape[0])
    f_video = f_video[:min_len]
    f_audio = f_audio[:min_len]

    # Compute scalar score with gradients
    visual_raw = f_video
    audio_raw = f_audio.detach()  # keep audio fixed for CAM

    visual_tensor = visual_raw / torch.linalg.norm(visual_raw, ord=2, dim=-1, keepdim=True)
    audio_tensor = audio_raw / torch.linalg.norm(audio_raw, ord=2, dim=-1, keepdim=True)

    output = fusion_model(visual_tensor, audio_tensor)  # (T, 1)
    score = torch.logsumexp(-output, dim=0).squeeze()  # scalar
    baseline_score = score.detach().clone()

    print(f"[Grad-CAM] Backprop from score={score.item():.4f}")
    fusion_model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    handle_fwd.remove()
    handle_bwd.remove()

    if activations["value"] is None or gradients["value"] is None:
        raise RuntimeError("[Grad-CAM] Hooks did not capture activations/gradients.")

    acts = activations["value"]  # (B*Tnew, C, Hc, Wc)
    grads = gradients["value"]   # (B*Tnew, C, Hc, Wc)
    if acts.ndim != 4 or grads.ndim != 4:
        raise RuntimeError(f"[Grad-CAM] Unexpected activation shape: {acts.shape}")

    weights = grads.mean(dim=(2, 3), keepdim=True)  # (B*T, C, 1, 1)
    cam = (weights * acts).sum(dim=1)  # (B*T, Hc, Wc)
    cam = F.relu(cam)

    cam = cam.detach().cpu().numpy()
    # Normalize each time slice independently.
    # cam shape: (T, Hc, Wc), so min/max must broadcast as (T,1,1).
    cam_min = cam.reshape(cam.shape[0], -1).min(axis=1, keepdims=True)  # (T,1)
    cam_max = cam.reshape(cam.shape[0], -1).max(axis=1, keepdims=True)  # (T,1)
    cam_min = cam_min[:, None, None]
    cam_max = cam_max[:, None, None]
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    # Prepare full temporal CAM vector (before mapping/truncation).
    T_full = int(cam.shape[0])
    frame_intensity_full = cam.mean(axis=(1, 2)).astype(np.float64).reshape(-1)  # (T_full,)

    # Map CAM time slices to ROI frames.
    # Current assumption in this script: cam index aligns with ROI frame index,
    # and we truncate to the shortest length to keep mapping deterministic.
    T_roi = int(len(roi_frames_gray))
    T_use = min(T_roi, T_full)
    cam_to_roi_index = list(range(T_use)) + ([None] * (T_full - T_use))

    # fps helps Streamlit label time in seconds (best-effort; may be 0/unknown for some codecs).
    roi_fps = None
    try:
        cap = cv2.VideoCapture(roi_path)
        if cap is not None and cap.isOpened():
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            if fps_val is not None and float(fps_val) > 0:
                roi_fps = float(fps_val)
        if cap is not None:
            cap.release()
    except Exception:
        roi_fps = None

    cam = cam[:T_use]
    roi_frames_gray = roi_frames_gray[:T_use]
    # Persist the full CAM volume used for overlays so downstream XAI can compute
    # temporal inconsistency, region tracking, and fused heatmaps.
    try:
        np.save(cam_volume_path, cam)
    except Exception as e:
        print(f"[Grad-CAM] Failed to save cam volume to {cam_volume_path}: {e}")

    # Strictly 1D vector to avoid nested-index surprises.
    frame_intensity = frame_intensity_full[:T_use]  # (T_use,)
    top_k = max(1, min(int(args.top_k), T_use))

    order = np.argsort(frame_intensity)[::-1]
    top_idx = [int(x) for x in order[:top_k].tolist()]
    # Safety clamp: avoids edge cases where cam/ROI lengths diverge.
    cam_T = int(cam.shape[0])
    top_idx = [t for t in top_idx if 0 <= t < cam_T]
    if not top_idx:
        top_idx = [0]

    print(f"[Grad-CAM] Debug selection: T_use={T_use}, cam_T={cam_T}, roi_len={len(roi_frames_gray)}, top_idx={top_idx}")

    for i, t in enumerate(top_idx):
        heat = cam[t]
        base = roi_frames_gray[t]
        # Resize heatmap to base frame size
        heat_resized = cv2.resize(heat, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat_u8 = (heat_resized * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)

        out_path = os.path.join(overlay_dir, f"cam_frame_{t:05d}.png")
        cv2.imwrite(out_path, overlay)
        print(f"[Grad-CAM] Saved: {out_path}")

    # Optional robustness delta evidence (baseline score vs adversarial score)
    adv_score = None
    delta_score = None
    adv_epsilon = args.adv_epsilon
    if args.adv_ckpt:
        if not os.path.isfile(args.adv_ckpt):
            raise FileNotFoundError(f"[Grad-CAM] --adv_ckpt not found: {args.adv_ckpt}")
        print(f"[Grad-CAM] Computing robustness delta using adversary: {args.adv_ckpt}")
        from train_feature_adversary import VisualPerturber

        adv_state = torch.load(args.adv_ckpt, map_location=device, weights_only=False)
        if adv_epsilon is None:
            try:
                adv_epsilon = float(adv_state.get("config", {}).get("epsilon", 0.05))
            except Exception:
                adv_epsilon = 0.05

        G = VisualPerturber(dim=1024, hidden=512, epsilon=float(adv_epsilon)).to(device)
        G.load_state_dict(adv_state["G_state"])
        G.eval()
        for p in G.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            delta = G(audio_raw, visual_raw)
            visual_adv_raw = visual_raw + delta

            visual_adv_norm = visual_adv_raw / torch.linalg.norm(visual_adv_raw, ord=2, dim=-1, keepdim=True)
            audio_norm = audio_raw / torch.linalg.norm(audio_raw, ord=2, dim=-1, keepdim=True)

            output_adv = fusion_model(visual_adv_norm, audio_norm)
            score_adv = torch.logsumexp(-output_adv, dim=0).squeeze()
            adv_score = float(score_adv.item())
            delta_score = float(score_adv.item() - baseline_score.item())

    # Write a tiny index for Streamlit
    with open(index_path, "w", encoding="utf-8") as f:
        json = {
            "video_path": args.video_path,
            "roi_path": roi_path,
            "score": float(baseline_score.item()),
            "baseline_score": float(baseline_score.item()),
            "adv_score": adv_score,
            "delta_score": delta_score,
            "adv_epsilon": adv_epsilon,
            "top_idx": sorted(top_idx),
            "overlay_dir": overlay_dir,
            "cam_volume_path": cam_volume_path if os.path.isfile(cam_volume_path) else None,
            # Full temporal CAM vector for panel-friendly heatmaps.
            "cam_per_frame": frame_intensity_full.tolist(),  # (T_full,)
            "cam_to_roi_index": cam_to_roi_index,  # length (T_full); entries may be None
            "roi_fps": roi_fps,
            "T_cam_full": T_full,
            "T_roi": T_roi,
            "T_use": int(T_use),
        }
        if attention_per_time is not None:
            # Best-effort alignment: truncate/pad by CAM time usage.
            # attention_per_time is expected to be aligned with f_video time length.
            attn_vec = np.asarray(attention_per_time, dtype=np.float64).reshape(-1)
            if attn_vec.shape[0] >= T_full:
                attn_vec = attn_vec[:T_full]
            else:
                # Pad with last value if attention is shorter than CAM timeline.
                pad_len = T_full - attn_vec.shape[0]
                if pad_len > 0 and attn_vec.shape[0] > 0:
                    attn_vec = np.concatenate([attn_vec, np.full(pad_len, attn_vec[-1])])
            # Normalize to [0,1] for stable visualization.
            attn_min = float(attn_vec.min()) if attn_vec.size else 0.0
            attn_max = float(attn_vec.max()) if attn_vec.size else 1.0
            attn_norm = (attn_vec - attn_min) / (attn_max - attn_min + 1e-8) if attn_vec.size else attn_vec
            json["attention_per_frame"] = attn_norm.tolist()
            json["attention_to_roi_index"] = cam_to_roi_index
        # avoid shadowing json module
        import json as _json
        f.write(_json.dumps(json, indent=2))

    print(f"[Grad-CAM] Done. Output dir: {args.out_dir}")

    if created_work_dir and not args.keep_temp and work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

