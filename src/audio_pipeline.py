"""Audio pipeline for voiceprint verification.

Sections (each one is explained in notebooks/audio_processing.ipynb):
    1. Loading + preprocessing   load_clip()
    2. Augmentation              augment()
    3. Feature extraction        extract_features(), build_feature_table()
    4. Voice model               cross_validate(), choose_threshold(), train_final_model()
    5. Use in the app            verify_speaker()

Run the whole thing from the repo root:
    python -m src.audio_pipeline
"""
from __future__ import annotations

import re
import warnings
import zlib
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"                     # data/audio/<member>/<member>_<phrase>.wav
UNAUTH_AUDIO_DIR = ROOT / "data" / "unauthorized" / "audio"
FEATURES_CSV = ROOT / "features" / "audio_features.csv"
MODEL_PATH = ROOT / "models" / "voice_model.joblib"

SR = 16_000          # every clip is resampled to 16 kHz: enough for speech, same "flipbook speed" for all phones
TOP_DB = 20          # trim edges quieter than 20 dB below the loudest point (= silence before/after speaking)
PEAK = 0.9           # normalise every clip so its loudest point is 0.9 (leaves room before clipping at 1.0)
N_MFCC = 13          # the classic number of MFCCs for speech
NOISE_SNR_DB = 20    # background noise augmentation: speech is 20 dB louder than the noise
RANDOM_STATE = 42

META_COLS = ["member", "phrase", "augmentation", "source_audio"]


# ---------------------------------------------------------------- 1. loading
def load_clip(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load one clip as 16 kHz mono, cut the silence at both ends, and even out the volume."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * PEAK
    return y


def phrase_of(path: Path) -> str:
    stem = path.stem.lower()
    if "yes_approve" in stem:
        return "yes_approve"
    if "confirm_transaction" in stem:
        return "confirm_transaction"
    return "unknown"


def speaker_of_outsider(path: Path) -> str:
    """outsider_towi_yes_approve.wav -> outsider_towi (one outsider per member who brought them)."""
    match = re.match(r"(outsider_[a-z]+)", path.stem.lower())
    return match.group(1) if match else path.stem.lower()
