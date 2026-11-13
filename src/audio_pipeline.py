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


# ---------------------------------------------------------------- 2. augmentation
def augment(y: np.ndarray, sr: int = SR, seed: int = 0) -> dict[str, np.ndarray]:
    """Five realistic variations of one clip, plus the original.

    Kept deliberately small so the voice still sounds like the same person:
      pitch +/- 1 semitone   a slightly higher/lower voice (tired, excited)
      speed x1.1 / x0.9      talking a bit faster or slower (pitch unchanged)
      background noise       a noisier room, 20 dB below the speech
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(y))
    signal_power = np.mean(y ** 2)
    noise *= np.sqrt(signal_power / 10 ** (NOISE_SNR_DB / 10))
    return {
        "original": y,
        "pitch_up": librosa.effects.pitch_shift(y, sr=sr, n_steps=1),
        "pitch_down": librosa.effects.pitch_shift(y, sr=sr, n_steps=-1),
        "speed_fast": librosa.effects.time_stretch(y, rate=1.1),
        "speed_slow": librosa.effects.time_stretch(y, rate=0.9),
        "noise": y + noise,
    }


# ---------------------------------------------------------------- 3. features
def extract_features(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Turn a clip of any length into the same 47 numbers.

    Each feature is computed per 32 ms frame, then summarised over the clip
    with its mean (typical value) and std (how much it moves around).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)          # (13, frames): shape of the vocal tract
    mfcc_delta = librosa.feature.delta(mfcc)                           # how fast that shape changes
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]                                  # energy (loudness) per frame
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    feats: dict[str, float] = {}
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i + 1}_std"] = float(mfcc[i].std())
        feats[f"mfcc_delta_{i + 1}_std"] = float(mfcc_delta[i].std())
    for name, values in [("spectral_rolloff", rolloff), ("spectral_centroid", centroid),
                         ("rms_energy", rms), ("zcr", zcr)]:
        feats[f"{name}_mean"] = float(values.mean())
        feats[f"{name}_std"] = float(values.std())
    return feats


def build_feature_table(root: Path = AUDIO_DIR, outsiders: bool = False, do_augment: bool = True) -> pd.DataFrame:
    """One row per (clip, augmentation). Members: data/audio/<member>/*.wav. Outsiders: flat folder."""
    files = sorted(root.glob("*.wav")) if outsiders else sorted(root.glob("*/*.wav"))
    rows = []
    for path in files:
        member = speaker_of_outsider(path) if outsiders else path.parent.name.lower()
        y = load_clip(path)
        versions = augment(y, seed=zlib.crc32(path.name.encode())) if do_augment else {"original": y}
        for aug_name, y_aug in versions.items():
            rows.append({"member": member, "phrase": phrase_of(path), "augmentation": aug_name,
                         "source_audio": f"{member}/{path.name}", **extract_features(y_aug)})
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]
