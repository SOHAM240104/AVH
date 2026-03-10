# AVH-Align

[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2412.00175)

**Official PyTorch Implementation:**

> **Ștefan Smeu, Dragoș-Alexandru Boldisor, Dan Oneață and Elisabeta Oneață**  
> [Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning](https://arxiv.org/abs/2412.00175)  
> *CVPR, 2025*

Audio-visual deepfake detection: the model scores how well mouth movements and speech are synchronized. **Higher score = more likely deepfake.**

---

## Step-by-step: How to run (single video)

Follow these steps in order. Works on **macOS (M1/M2/M3)** with CPU or **Linux/Windows** with CUDA.

### Step 1: Clone this repo

```bash
git clone https://github.com/bit-ml/AVH-Align.git
cd AVH-Align
```

### Step 2: Create a Python 3.10 environment

We use Python 3.10 for compatibility with fairseq (used by AV-HuBERT).

```bash
# With conda (recommended)
conda create -n avh python=3.10 -y
conda activate avh
```

If you don’t use conda, create a venv with Python 3.10 and activate it.

### Step 3: Install PyTorch and dependencies

```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas tqdm
pip install opencv-python dlib librosa python_speech_features scikit-video
pip install scikit-image sentencepiece
```

### Step 4: Clone AV-HuBERT and install fairseq

AV-HuBERT is the backbone that extracts audio and visual features. Run from the **AVH-Align** repo root:

```bash
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert/avhubert
git submodule init
git submodule update
cd ../fairseq
pip install --editable ./
cd ../avhubert
```

Fix dependency versions (needed for this fairseq version):

```bash
pip install "numpy<1.24"
pip install "omegaconf>=2.1" "hydra-core>=1.1"
pip install "sentencepiece"
```

If you see `np.float` / `np.int` errors, ensure `numpy<1.24`. If fairseq fails to load checkpoints, install `sentencepiece`.

### Step 5: Download face models and AV-HuBERT checkpoint

Still inside `av_hubert/avhubert`:

```bash
mkdir -p content/data/misc/
# Face landmark predictor (~95 MB)
curl -L -o content/data/misc/shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2

# Mean face for mouth cropping
curl -L -o content/data/misc/20words_mean_face.npy \
  https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy

# AV-HuBERT Large checkpoint (~1 GB; required for feature extraction)
curl -L -o self_large_vox_433h.pt \
  https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
```

On Linux you can use `wget` instead of `curl` if you prefer.

### Step 6: Copy scripts and fix fairseq checkpoint loading

From the **AVH-Align** repo root (not inside `av_hubert`):

```bash
cp deepfake_preprocess.py av_hubert/avhubert/
cp deepfake_feature_extraction.py av_hubert/avhubert/
```

Optional but recommended for PyTorch ≥2.6: allow loading the AV-HuBERT checkpoint by editing fairseq:

- Open `av_hubert/fairseq/fairseq/checkpoint_utils.py`
- Find the line: `state = torch.load(f, map_location=torch.device("cpu"))`
- Change it to: `state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)`

### Step 7: Install ffmpeg

Required for audio extraction and video writing.

- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **Windows:** download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Step 8: Run detection on a single video

From the **AVH-Align** repo root, with `conda activate avh` (or your venv) active:

```bash
python test_video.py --video /path/to/your/video.mp4
```

Example:

```bash
python test_video.py --video test_videos/deepfake_obama.mp4
```

Output:

- **DEEPFAKE SCORE:** one number. **Higher = more likely fake**, lower = more likely real.
- Typical range: real videos often negative to small positive; clear deepfakes can be +5 to +10.

Options:

- `--keep_temp` — keep temporary preprocessed files (mouth crop, audio) for debugging.
- `--use_mps` — use Apple MPS GPU on M1/M2/M3 (experimental).
- `--avhubert_ckpt` / `--fusion_ckpt` — custom paths to AV-HuBERT and AVH-Align checkpoints.

**Note:** First run loads the large AV-HuBERT model (~30 s on CPU). A 1–2 minute video may take several minutes on CPU; GPU is much faster.

---

## What’s in this repo

| Path | Description |
|------|-------------|
| `test_video.py` | One-command deepfake detection for a single video (preprocess + AV-HuBERT + FusionModel). |
| `checkpoints/AVH-Align_AV1M.pt` | Pretrained AVH-Align weights (unsupervised, AV-Deepfake1M). |
| `model.py` | FusionModel: small MLP that scores audio–visual sync. |
| `eval.py` | Batch evaluation on pre-extracted `.npz` features (needs features from full pipeline below). |
| `train.py` | Train AVH-Align (unsupervised) on extracted features. |
| `config.py` | Training/config args. |
| `dataset.py` | Dataset for training (iterates over `.npz` feature files). |
| `av1m_metadata/` | Example metadata CSVs (path, label) for AV-Deepfake1M. |
| `avh_sup/` | Supervised variant (PyTorch Lightning); see `avh_sup/README.md`. |

---

## Full pipeline (preprocess → features → train/eval)

For training or batch evaluation on datasets (e.g. AV-Deepfake1M), you need pre-extracted features.

### Data

- **AV-Deepfake1M (AV1M):** [AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M)
- **FakeAVCeleb:** [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)
- **AVLips:** [LipFD](https://github.com/AaronComo/LipFD)

### Preprocess videos (mouth crop + audio)

Run from `av_hubert/avhubert` (after Step 4–5):

```bash
python deepfake_preprocess.py \
  --dataset AV1M \
  --split train \
  --metadata /path/to/av1m_metadata/train_metadata.csv \
  --data_path /path/to/AV1M_root \
  --save_path /path/to/preprocessed
```

### Extract AV-HuBERT features

```bash
python deepfake_feature_extraction.py \
  --dataset AV1M \
  --split train \
  --metadata /path/to/av1m_metadata/train_metadata.csv \
  --ckpt_path self_large_vox_433h.pt \
  --data_path /path/to/preprocessed \
  --save_path /path/to/features
```

Use `--trimmed` for silence-trimmed features if needed.

### Train (unsupervised)

From AVH-Align repo root:

```bash
python train.py --name=my_run \
  --data_root_path=/path/to/features \
  --metadata_root_path=/path/to/av1m_metadata
```

Checkpoints are saved under `checkpoints/` (see `config.py` for `--save_path`).

### Batch evaluation (pre-extracted features only)

```bash
python eval.py \
  --checkpoint_path checkpoints/AVH-Align_AV1M.pt \
  --features_path /path/to/features \
  --metadata av1m_metadata/test_metadata.csv \
  --dataset AV1M
```

Reports AUC and AP. Requires that `.npz` feature files and metadata CSV match (see `eval.py` and `av1m_metadata/`).

---

## Creating a new repo and pushing (your own fork)

If you want this as your own GitHub repo:

1. Create a **new repository** on GitHub (e.g. `YourUser/AVH-Align`). Do not add a README or .gitignore (we already have them).

2. Add it as a remote and push (from `AVH-Align` on your machine):

   ```bash
   git remote add myorigin https://github.com/YourUser/AVH-Align.git
   git add .gitignore README.md test_video.py
   git add deepfake_preprocess.py deepfake_feature_extraction.py
   git add model.py eval.py train.py config.py dataset.py utils.py
   git add checkpoints/ av1m_metadata/ avh_sup/
   git status   # ensure av_hubert/, venv/, test_videos/, *.mp4 are ignored
   git commit -m "Add step-by-step README, test_video.py, and .gitignore"
   git push -u myorigin main
   ```

3. Do **not** add the `av_hubert/` folder or `self_large_vox_433h.pt` (they are large and in `.gitignore`). Anyone who clones your repo will run Step 4–5 to get AV-HuBERT and the checkpoint.

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

CC BY-NC-SA 4.0. See [license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

This repository uses code from [FACTOR](https://github.com/talreiss/FACTOR) and [AV-HuBERT](https://github.com/facebookresearch/av_hubert).
