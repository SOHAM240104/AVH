"""
End-to-end deepfake detection for a single video file.
Patched for CPU / Apple Silicon (no CUDA required).

Usage:
    conda activate avh
    python test_video.py --video /path/to/video.mp4

Outputs a deepfake score: higher = more likely fake.
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F

# ── Step 0: path setup so we can import from av_hubert/avhubert ──────────
AVHUBERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "av_hubert", "avhubert")
sys.path.insert(0, AVHUBERT_DIR)

from python_speech_features import logfbank
import librosa

# AV-HuBERT uses sys.argv length to decide import style;
# temporarily set it so absolute (non-package) imports are used.
_real_argv = sys.argv
sys.argv = [""]
os.chdir(AVHUBERT_DIR)

import hubert_pretraining, hubert, hubert_asr  # noqa: F401 (registers fairseq tasks)
import utils as avhubert_utils
from fairseq import checkpoint_utils

from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

sys.argv = _real_argv
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FusionModel

# ── Constants ────────────────────────────────────────────────────────────
FACE_PREDICTOR_PATH = os.path.join(AVHUBERT_DIR, "content", "data", "misc", "shape_predictor_68_face_landmarks.dat")
MEAN_FACE_PATH = os.path.join(AVHUBERT_DIR, "content", "data", "misc", "20words_mean_face.npy")
AVHUBERT_CKPT = os.path.join(AVHUBERT_DIR, "self_large_vox_433h.pt")
FUSION_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "AVH-Align_AV1M.pt")

STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
FPS = 25


# ═══════════════════════════════════════════════════════════════════════
#  STAGE 1 – Preprocess: crop mouth ROI + extract audio
# ═══════════════════════════════════════════════════════════════════════

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_video(video_path, work_dir):
    """Crop mouth ROI and extract audio WAV. Returns (roi_path, audio_path)."""
    import skvideo.io

    print("[Stage 1] Preprocessing video: face detection + mouth crop ...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
    mean_face_landmarks = np.load(MEAN_FACE_PATH)

    videogen = skvideo.io.vread(video_path)
    frames = np.array([frame for frame in videogen])
    print(f"  Loaded {len(frames)} frames")

    landmarks = []
    for i, frame in enumerate(frames):
        lm = detect_landmark(frame, detector, predictor)
        landmarks.append(lm)
        if (i + 1) % 100 == 0:
            print(f"  Landmark detection: {i + 1}/{len(frames)}")
    print(f"  Landmark detection: {len(frames)}/{len(frames)} done")

    preprocessed_landmarks = landmarks_interpolate(landmarks)

    json_payload = None
    try:
        rois = crop_patch(
            video_path, preprocessed_landmarks, mean_face_landmarks,
            STABLE_PNTS_IDS, STD_SIZE, window_margin=12,
            start_idx=48, stop_idx=68, crop_height=96, crop_width=96,
        )
        if rois is None:
            raise ValueError("crop_patch returned None")
    except Exception as e:
        print(f"  Warning: mouth crop failed ({e}), resizing full frames to 96x96")
        rois = np.array([cv2.resize(f, (96, 96)) for f in frames])

    roi_path = os.path.join(work_dir, "mouth_roi.mp4")
    audio_path = os.path.join(work_dir, "audio.wav")

    ffmpeg_bin = "/opt/homebrew/bin/ffmpeg"
    if not os.path.exists(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg"

    write_video_ffmpeg(rois, roi_path, ffmpeg_bin)

    subprocess.run(
        [ffmpeg_bin, "-i", video_path, "-f", "wav", "-vn", "-y", audio_path, "-loglevel", "quiet"],
        check=True,
    )
    print(f"  Saved mouth ROI → {roi_path}")
    print(f"  Saved audio     → {audio_path}")
    return roi_path, audio_path


# ═══════════════════════════════════════════════════════════════════════
#  STAGE 2 – Feature extraction with AV-HuBERT (patched for CPU)
# ═══════════════════════════════════════════════════════════════════════

def load_avhubert(ckpt_path, device):
    """Load AV-HuBERT model onto the given device (cpu / mps)."""
    print(f"[Stage 2] Loading AV-HuBERT checkpoint ({os.path.basename(ckpt_path)}) ...")
    print(f"  This may take a minute on CPU with 8GB RAM ...")

    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(model, "decoder"):
        model = model.encoder.w2v_model

    model = model.to(device).eval()
    print(f"  AV-HuBERT loaded on {device}")
    return model, task


def compute_starting_silence(audio_path, threshold=0.0005, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    for i, sample in enumerate(audio):
        if abs(sample) > threshold:
            return i / sr
    return len(audio) / sr


def load_audio_features(path, silence_duration=0, sample_rate=16000, stack_order_audio=4):
    wav_data, sr = librosa.load(path, sr=sample_rate)
    assert sr == sample_rate and len(wav_data.shape) == 1

    skiped_frames = int(silence_duration * FPS) * 640
    if silence_duration > 0:
        skiped_frames += 640
    wav_data = wav_data[skiped_frames:]

    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)

    if len(audio_feats) % stack_order_audio != 0:
        pad = stack_order_audio - len(audio_feats) % stack_order_audio
        audio_feats = np.concatenate([audio_feats, np.zeros((pad, audio_feats.shape[1]), dtype=audio_feats.dtype)])

    audio_feats = audio_feats.reshape(-1, stack_order_audio * audio_feats.shape[1])
    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    return audio_feats


def extract_avhubert_features(model, video_path, audio_path, transform, device):
    """Extract audio and visual features. Returns (audio_feats, visual_feats) as numpy arrays."""
    print("  Extracting AV-HuBERT features (this will be slow on CPU) ...")

    frames = avhubert_utils.load_video(video_path)
    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).to(device)

    audio = load_audio_features(audio_path)[None, :, :].transpose(1, 2).to(device)

    min_len = min(frames.shape[2], audio.shape[-1])
    frames, audio = frames[:, :, :min_len], audio[:, :, :min_len]

    print(f"  Frames: {min_len}, running model forward passes ...")

    with torch.no_grad():
        print("    Extracting audio features ...")
        f_audio, _ = model.extract_finetune({"video": None, "audio": audio}, None, None)
        print("    Extracting visual features ...")
        f_video, _ = model.extract_finetune({"video": frames, "audio": None}, None, None)

    f_audio = f_audio.squeeze(0).cpu().numpy()
    f_video = f_video.squeeze(0).cpu().numpy()

    print(f"  Features extracted: audio={f_audio.shape}, visual={f_video.shape}")
    return f_audio, f_video


# ═══════════════════════════════════════════════════════════════════════
#  STAGE 3 – Run AVH-Align deepfake detector
# ═══════════════════════════════════════════════════════════════════════

def run_detector(visual_feats, audio_feats, checkpoint_path, device):
    """Run the AVH-Align FusionModel and return a deepfake score."""
    print("[Stage 3] Running AVH-Align deepfake detector ...")

    sys.path.insert(0, os.path.dirname(__file__))
    from model import FusionModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FusionModel().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    visual_tensor = torch.from_numpy(visual_feats).to(device)
    audio_tensor = torch.from_numpy(audio_feats).to(device)

    visual_tensor = visual_tensor / torch.linalg.norm(visual_tensor, ord=2, dim=-1, keepdim=True)
    audio_tensor = audio_tensor / torch.linalg.norm(audio_tensor, ord=2, dim=-1, keepdim=True)

    with torch.no_grad():
        output = model(visual_tensor, audio_tensor)
        score = torch.logsumexp(-output, dim=0).squeeze().item()

    return score


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test a single video for deepfake detection")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file (.mp4)")
    parser.add_argument("--avhubert_ckpt", type=str, default=AVHUBERT_CKPT, help="Path to AV-HuBERT checkpoint")
    parser.add_argument("--fusion_ckpt", type=str, default=FUSION_CKPT, help="Path to AVH-Align checkpoint")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary preprocessed files")
    parser.add_argument("--json_out", type=str, default=None, help="Optional path to write result JSON.")
    parser.add_argument("--use_mps", action="store_true", help="Use Apple MPS GPU (experimental)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video file not found: {args.video}")
        sys.exit(1)

    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    work_dir = tempfile.mkdtemp(prefix="avh_align_")
    print(f"Working directory: {work_dir}\n")

    try:
        # Stage 1: Preprocess
        roi_path, audio_path = preprocess_video(args.video, work_dir)
        print()

        # Stage 2: Extract features
        avh_model, task = load_avhubert(args.avhubert_ckpt, device)
        transform = avhubert_utils.Compose([
            avhubert_utils.Normalize(0.0, 255.0),
            avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
        ])

        audio_feats, visual_feats = extract_avhubert_features(
            avh_model, roi_path, audio_path, transform, device
        )

        del avh_model
        torch.mps.empty_cache() if device.type == "mps" else None
        import gc; gc.collect()
        print()

        # Stage 3: Deepfake detection
        score = run_detector(visual_feats, audio_feats, args.fusion_ckpt, torch.device("cpu"))

        json_payload = {
            "success": True,
            "score": float(score),
            "audio_path": audio_path if args.keep_temp else None,
            "roi_path": roi_path if args.keep_temp else None,
        }
        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(json_payload, f)

        print()
        print("=" * 55)
        print(f"  DEEPFAKE SCORE: {score:.4f}")
        print(f"  Higher score = more likely to be a deepfake")
        print("=" * 55)

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
