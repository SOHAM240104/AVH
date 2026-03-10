# AVH-Align — Deepfake Detection (How I Set It Up & Run It)

[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2412.00175)

**Official PyTorch implementation of the paper:**

> **Ștefan Smeu, Dragoș-Alexandru Boldisor, Dan Oneață and Elisabeta Oneață**  
> [Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning](https://arxiv.org/abs/2412.00175)  
> *CVPR, 2025*

I got this running on **macOS (M3)** with CPU. It detects deepfakes by checking if mouth movements and speech are in sync. **Higher score = more likely fake; lower score = more likely real.**

---

## What I Did — Overview

1. Cloned the repo and set up a Python 3.10 environment (conda).
2. Installed PyTorch and all dependencies.
3. Cloned AV-HuBERT, installed fairseq, and fixed a few dependency/loading issues.
4. Downloaded the face landmark model, mean face, and the big AV-HuBERT checkpoint (~1 GB).
5. Installed ffmpeg and ran the single-video test script.

Below are the **exact steps I followed**, with details so you can run the test and everything else the same way.

---

## Detailed Step-by-Step Setup

### Step 1: Clone this repo

```bash
git clone https://github.com/SOHAM240104/AVH.git
cd AVH
```

You should see folders like `checkpoints/`, `av1m_metadata/`, and files like `test_video.py`, `model.py`, `eval.py`, etc.

---

### Step 2: Create a Python 3.10 environment

The AV-HuBERT part uses **fairseq**, which needs Python 3.10. I used conda.

```bash
conda create -n avh python=3.10 -y
conda activate avh
```

Check the version:

```bash
python --version
# Should show: Python 3.10.x
```

If you don’t have conda, use a virtualenv with Python 3.10 and activate it before the next steps.

---

### Step 3: Install PyTorch and all dependencies

Run these in order. I did this from the **AVH** repo root with `conda activate avh` already active.

**3a. PyTorch (CPU is enough for testing; use CUDA on Linux if you have a GPU):**

```bash
pip install torch torchvision torchaudio
```

**3b. Other Python packages used by the project:**

```bash
pip install scikit-learn pandas tqdm
pip install opencv-python dlib librosa python_speech_features scikit-video
pip install scikit-image sentencepiece
```

- `opencv-python`, `dlib` — face detection and mouth cropping  
- `librosa`, `python_speech_features` — audio loading and features  
- `scikit-video` — reading video frames  
- `sentencepiece` — needed when loading the AV-HuBERT checkpoint  

If something fails (e.g. `dlib` on Windows), install build tools or use pre-built wheels as per the package docs.

---

### Step 4: Clone AV-HuBERT and install fairseq

AV-HuBERT is the model that turns video + audio into feature vectors. We need it inside our repo.

**4a. Clone AV-HuBERT (from repo root):**

```bash
# You should be in the AVH folder
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert/avhubert
```

**4b. Init submodules (this pulls in fairseq):**

```bash
git submodule init
git submodule update
```

**4c. Install fairseq in editable mode:**

```bash
cd ../fairseq
pip install --editable ./
cd ../avhubert
```

**4d. Fix dependency versions (I had to do this or imports/checkpoint loading failed):**

```bash
pip install "numpy<1.24"
pip install "omegaconf>=2.1" "hydra-core>=1.1"
pip install "sentencepiece"
```

- `numpy<1.24` avoids `np.float` / `np.int` errors in fairseq.  
- Newer omegaconf/hydra work with this fairseq.  
- If you get “Unsupported global” or “sentencepiece” when loading the checkpoint, `sentencepiece` fixes it.

**4e. Optional but recommended (PyTorch 2.6+):**  
Edit `av_hubert/fairseq/fairseq/checkpoint_utils.py`, find:

```python
state = torch.load(f, map_location=torch.device("cpu"))
```

Change to:

```python
state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
```

This lets the AV-HuBERT checkpoint load. Then go back to the **AVH** repo root:

```bash
cd ../..   # back to AVH
```

---

### Step 5: Download face models and AV-HuBERT checkpoint

These files are not in the repo (too big). You need to download them once.

**5a. Go into the avhubert folder again:**

```bash
cd av_hubert/avhubert
```

**5b. Create the misc folder and download the face landmark predictor (~95 MB):**

```bash
mkdir -p content/data/misc/
curl -L -o content/data/misc/shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
```

**5c. Download the mean face (used for mouth cropping):**

```bash
curl -L -o content/data/misc/20words_mean_face.npy \
  https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy
```

**5d. Download the AV-HuBERT Large checkpoint (~1 GB). This can take a few minutes:**

```bash
curl -L -o self_large_vox_433h.pt \
  https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
```

On Linux you can use `wget` instead of `curl` if you prefer.

**5e. Copy our scripts into avhubert (from AVH repo root):**

```bash
cd ../..   # back to AVH repo root
cp deepfake_preprocess.py av_hubert/avhubert/
cp deepfake_feature_extraction.py av_hubert/avhubert/
```

---

### Step 6: Install ffmpeg

Needed for extracting audio from videos and writing the cropped mouth video.

- **macOS:** `brew install ffmpeg`  
- **Ubuntu/Debian:** `sudo apt install ffmpeg`  
- **Windows:** download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

Check:

```bash
ffmpeg -version
```

---

## How I Run the Test (Single Video)

This is the main way I use the project: one command per video.

**1. Open a terminal, go to the repo, activate the environment:**

```bash
cd /path/to/AVH
conda activate avh
```

**2. Run the test script on any video file (must have a face and audio):**

```bash
python test_video.py --video /path/to/your/video.mp4
```

Example with a file in the repo:

```bash
python test_video.py --video test_videos/deepfake_obama.mp4
```

**3. What happens:**

- **Stage 1 — Preprocess:** Detects the face, crops the mouth region, extracts audio. You’ll see “Loaded X frames” and paths to saved mouth ROI and audio.
- **Stage 2 — Features:** Loads the AV-HuBERT checkpoint (first time can take ~30 s on CPU), then runs the model on the cropped video and audio. You’ll see “Extracting audio features” and “Extracting visual features”, then “Features extracted: audio=(X, 1024), visual=(X, 1024)”.
- **Stage 3 — Detection:** Loads our small FusionModel and outputs one number.

**4. Output at the end looks like:**

```
=======================================================
  DEEPFAKE SCORE: 6.0319
  Higher score = more likely to be a deepfake
=======================================================
```

**5. How I interpret it:**

- **Negative or small positive (e.g. -2 to +1):** Looks more like a **real** video (mouth and speech in sync).
- **Larger positive (e.g. +5 to +10):** More likely a **deepfake** (mouth and speech don’t match well).

A 1–2 minute video on CPU can take several minutes; the slow part is Stage 2 (AV-HuBERT). GPU is much faster if you have it.

**Optional flags I use sometimes:**

- `--keep_temp` — Keep the temporary mouth crop and audio files (for debugging).
- `--use_mps` — Use Apple MPS on M1/M2/M3 (experimental).
- `--avhubert_ckpt path/to/self_large_vox_433h.pt` — If the checkpoint is not in `av_hubert/avhubert/`.
- `--fusion_ckpt path/to/AVH-Align_AV1M.pt` — If you want to use a different FusionModel checkpoint.

---

## Running Batch Evaluation (Pre-extracted features only)

If you already have **extracted features** (`.npz` files from the full pipeline), you can run evaluation without preprocessing again:

```bash
python eval.py \
  --checkpoint_path checkpoints/AVH-Align_AV1M.pt \
  --features_path /path/to/folder/containing/npz/files \
  --metadata av1m_metadata/test_metadata.csv \
  --dataset AV1M
```

The CSV must have columns `path` and `label`; `path` should match the `.npz` filenames (e.g. `path` like `id123/video.mp4` and files like `id123/video.npz`). This prints AUC and AP.

---

## Running Training (Optional)

I didn’t train from scratch; I only use the provided checkpoint. If you want to train on your own extracted features:

```bash
python train.py --name=my_run \
  --data_root_path=/path/to/features \
  --metadata_root_path=/path/to/av1m_metadata
```

Training uses the metadata CSVs to find train/val splits and the feature `.npz` files. Checkpoints go to `checkpoints/` (see `config.py` for `--save_path`).

---

## Full Pipeline (Preprocess → Extract features → Train/Eval)

For **training** or **batch eval**, you need to run the full pipeline once to get `.npz` features.

**1. Preprocess videos (mouth crop + audio):**  
Run from `av_hubert/avhubert` (after setup above):

```bash
cd av_hubert/avhubert
python deepfake_preprocess.py \
  --dataset AV1M \
  --split train \
  --metadata /path/to/av1m_metadata/train_metadata.csv \
  --data_path /path/to/AV1M_root \
  --save_path /path/to/preprocessed
```

**2. Extract AV-HuBERT features:**

```bash
python deepfake_feature_extraction.py \
  --dataset AV1M \
  --split train \
  --metadata /path/to/av1m_metadata/train_metadata.csv \
  --ckpt_path self_large_vox_433h.pt \
  --data_path /path/to/preprocessed \
  --save_path /path/to/features
```

Add `--trimmed` if you want silence-trimmed features.

**3. Train or run eval** as in the sections above, pointing to the same `--features_path` and metadata.

Datasets I referenced: [AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M), [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb), [AVLips/LipFD](https://github.com/AaronComo/LipFD).

---

## What’s in this repo (what I use)

| Path | What I use it for |
|------|-------------------|
| `test_video.py` | Main script: run deepfake detection on a single video (preprocess + AV-HuBERT + FusionModel). |
| `checkpoints/AVH-Align_AV1M.pt` | Pretrained detector weights; used by default by `test_video.py` and `eval.py`. |
| `model.py` | FusionModel (small MLP that scores audio–visual sync). |
| `eval.py` | Batch evaluation on pre-extracted `.npz` features. |
| `train.py` | Training the detector on extracted features. |
| `config.py` | Training/config arguments. |
| `dataset.py` | Dataset that loads `.npz` features for training. |
| `av1m_metadata/` | Example metadata CSVs (path, label) for AV-Deepfake1M. |
| `avh_sup/` | Supervised variant (PyTorch Lightning); see `avh_sup/README.md` if you need it. |

---

## Citation

```bibtex
@InProceedings{AVH-Align,
    author    = {Smeu, Stefan and Boldisor, Dragos-Alexandru and Oneata, Dan and Oneata, Elisabeta},
    title     = {Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

## License

CC BY-NC-SA 4.0. This repository uses code from [FACTOR](https://github.com/talreiss/FACTOR) and [AV-HuBERT](https://github.com/facebookresearch/av_hubert).
