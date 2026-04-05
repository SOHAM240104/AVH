"""
Spatial pre-crop for social-style vertical video (reels, Shorts, TikTok-style UI).

Removes typical caption / chrome bands before AVH mouth ROI extraction, and optionally
tightens the frame around the largest detected face across sampled frames.

Runs *before* landmark-based mouth crop in test_video.preprocess_video.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

import cv2
import numpy as np


def resolve_ffmpeg_bin() -> str:
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/bin/ffmpeg", "ffmpeg"):
        if cand == "ffmpeg":
            p = shutil.which("ffmpeg")
            if p:
                return p
        elif os.path.isfile(cand):
            return cand
    return "ffmpeg"


def _even(x: int) -> int:
    return max(2, (int(x) // 2) * 2)


def probe_video_meta(path: str) -> tuple[int, int, int]:
    """Return (width, height, frame_count_approx)."""
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if w > 0 and h > 0:
        return w, h, max(1, n)
    # Fallback: ffprobe JSON
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    if not os.path.isfile(ffprobe) and ffprobe != "ffprobe":
        ffprobe = "ffprobe"
    try:
        out = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "v:0", path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if out.returncode == 0 and out.stdout:
            data = json.loads(out.stdout)
            streams = data.get("streams") or []
            if streams:
                st = streams[0]
                w = int(st.get("width") or 0)
                h = int(st.get("height") or 0)
                nb = st.get("nb_frames")
                n = int(nb) if nb and str(nb).isdigit() else 1
                if w > 0 and h > 0:
                    return w, h, max(1, n)
    except Exception:
        pass
    return 0, 0, 1


def _vertical_band_rect(w: int, h: int, top_frac: float, bottom_frac: float) -> tuple[int, int, int, int]:
    """Return (x, y, cw, ch) full-width crop removing top/bottom UI strips."""
    tf = float(np.clip(top_frac, 0.0, 0.45))
    bf = float(np.clip(bottom_frac, 0.0, 0.45))
    if tf + bf >= 0.95:
        tf, bf = 0.08, 0.12
    y0 = int(round(h * tf))
    y1 = int(round(h * (1.0 - bf)))
    ch = max(64, y1 - y0)
    return 0, y0, w, ch


def _sample_indices(n_frames: int, max_samples: int = 36) -> list[int]:
    if n_frames <= 0:
        return []
    if n_frames <= max_samples:
        return list(range(n_frames))
    return [int(round(x)) for x in np.linspace(0, n_frames - 1, max_samples)]


def _largest_face_rect(gray: np.ndarray, detector) -> tuple[int, int, int, int] | None:
    rects = detector(gray, 1)
    if not rects:
        return None
    best = max(rects, key=lambda r: r.width() * r.height())
    return (best.left(), best.top(), best.width(), best.height())


def _aggregate_face_crop(
    video_path: str,
    n_frames: int,
    w: int,
    h: int,
    margin_frac: float = 0.38,
) -> tuple[int, int, int, int] | None:
    import dlib

    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    rects: list[tuple[int, int, int, int]] = []
    for idx in _sample_indices(n_frames, max_samples=40):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = gray
        scale = 1.0
        if w > 960:
            scale = 960.0 / float(w)
            small = cv2.resize(gray, (int(w * scale), int(h * scale)))
        r = _largest_face_rect(small, detector)
        if r is None:
            continue
        x, y, rw, rh = r
        if scale < 1.0:
            x = int(round(x / scale))
            y = int(round(y / scale))
            rw = int(round(rw / scale))
            rh = int(round(rh / scale))
        area = rw * rh
        if area < 0.015 * w * h:
            continue
        rects.append((x, y, rw, rh))
    cap.release()
    if len(rects) < 2:
        return None

    xs = np.median([a[0] for a in rects])
    ys = np.median([a[1] for a in rects])
    ws = np.median([a[2] for a in rects])
    hs = np.median([a[3] for a in rects])
    cx = xs + ws / 2.0
    cy = ys + hs / 2.0
    side = float(max(ws, hs)) * (1.0 + margin_frac)
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    cw = int(round(side))
    ch = int(round(side))
    x0 = int(np.clip(x0, 0, max(0, w - 32)))
    y0 = int(np.clip(y0, 0, max(0, h - 32)))
    cw = min(cw, w - x0)
    ch = min(ch, h - y0)
    cw = max(64, _even(cw))
    ch = max(64, _even(ch))
    if cw < 64 or ch < 64:
        return None
    return x0, y0, cw, ch


def _ffmpeg_crop(in_path: str, out_path: str, x: int, y: int, w: int, h: int, ffmpeg_bin: str) -> bool:
    x, y, w, h = _even(x), _even(y), _even(w), _even(h)
    vf = f"crop={w}:{h}:{x}:{y}"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        return r.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 100
    except Exception:
        return False


def maybe_smart_spatial_crop(
    video_path: str,
    work_dir: str,
    *,
    mode: str = "auto",
    top_frac: float = 0.10,
    bottom_frac: float = 0.14,
) -> tuple[str, str | None]:
    """
    Optionally write a spatially cropped copy of the input under work_dir.

    Returns:
        (path_to_feed_into_landmark_pipeline, log_line_or_none)

    Modes:
        off   — no crop
        reel  — vertical band only (strip top/bottom) when aspect looks like 9:16
        face  — largest-face-centered square (sampled frames); fallback to reel if vertical
        auto  — try face crop; else reel if h/w > 1.15; else off
    """
    mode = (mode or "auto").strip().lower()
    if mode in ("none", "false", "0", "off"):
        return video_path, None

    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, "spatial_cropped_input.mp4")
    ffmpeg_bin = resolve_ffmpeg_bin()

    w, h, n_frames = probe_video_meta(video_path)
    if w <= 0 or h <= 0:
        return video_path, "smart_crop: could not read dimensions; using original"

    is_vertical = h > w * 1.15
    rect: tuple[int, int, int, int] | None = None
    log: str | None = None

    if mode == "reel":
        if is_vertical:
            rect = _vertical_band_rect(w, h, top_frac, bottom_frac)
            log = f"smart_crop: reel vertical band crop ({rect[2]}x{rect[3]} @ y={rect[1]})"
        else:
            return video_path, "smart_crop: reel mode but frame not vertical; using original"

    elif mode == "face":
        try:
            rect = _aggregate_face_crop(video_path, n_frames, w, h)
        except Exception:
            rect = None
        if rect is not None:
            log = f"smart_crop: face-centered crop ({rect[2]}x{rect[3]} @ {rect[0]},{rect[1]})"
        elif is_vertical:
            rect = _vertical_band_rect(w, h, top_frac, bottom_frac)
            log = "smart_crop: face not stable; fallback reel vertical band"
        else:
            return video_path, "smart_crop: face mode found no confident face; using original"

    else:  # auto
        try:
            rect = _aggregate_face_crop(video_path, n_frames, w, h)
        except Exception:
            rect = None
        if rect is not None:
            log = f"smart_crop: auto face-centered crop ({rect[2]}x{rect[3]})"
        elif is_vertical:
            rect = _vertical_band_rect(w, h, top_frac, bottom_frac)
            log = "smart_crop: auto vertical band (reel-style UI strip removal)"
        else:
            return video_path, "smart_crop: auto skipped (landscape / no face consensus)"

    if rect is None:
        return video_path, None

    x, y, cw, ch = rect
    if _ffmpeg_crop(video_path, out_path, x, y, cw, ch, ffmpeg_bin):
        return out_path, log
    return video_path, f"smart_crop: ffmpeg failed; using original ({log or 'crop'})"


def normalize_smart_crop_mode(raw: Any) -> str:
    if raw is None:
        return "auto"
    s = str(raw).strip().lower()
    if s in ("off", "none", "false", "0", "no"):
        return "off"
    if s in ("auto", "reel", "face"):
        return s
    return "auto"
