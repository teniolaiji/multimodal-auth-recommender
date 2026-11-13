"""Face part of the system: cut out a face, turn it into numbers and say who it is.

Used by the notebook and by the command line app.
Main function for the app:
    recognise_face(path) -> (member, confidence)

Run:  python src/image_features.py <photo path>
"""
import json

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import (FACE_BORDER, FACE_METRICS_PATH, FACE_MODEL_PATH, FACE_SIZE, FACE_THRESHOLD,
                    IMAGE_FEATURES_CSV, IMAGES_DIR, RANDOM_STATE, UNAUTHORIZED_DIR)

#OpenCV's ready-made face finder
FACE_FINDER = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#search settings from strict to soft(scale, neighbors)
FACE_SEARCH_SETTINGS = [(1.1, 5), (1.05, 3), (1.03, 2)]


# -------------------------------------------------------------- face
def crop_face(img):
    """Find the face and cut it out. If no face is found, keep the whole picture."""
    pic = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

    #ignore tiny shapes(a face is at least 15% of the picture)
    min_size = (int(0.15 * min(gray.shape)),) * 2

    #try the strict setting first then softer ones
    faces = ()
    for scale, neighbors in FACE_SEARCH_SETTINGS:
        faces = FACE_FINDER.detectMultiScale(gray, scale, neighbors, minSize=min_size)
        if len(faces) > 0:
            break
    if len(faces) == 0:
        return img

    #keep the biggest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    #add a small border around the face
    m = int(FACE_BORDER * w)
    x1, y1 = max(x - m, 0), max(y - m, 0)
    x2, y2 = x + w + m, y + h + m
    return Image.fromarray(pic[y1:y2, x1:x2])


# ------------------------------------------------------------ versions
def make_versions(img):
    """Make new versions of one face: original, flip, rotate, gray."""
    return {
        "original": img,
        "flip": ImageOps.mirror(img),                    # mirror
        "rotate": img.rotate(15),                        # turn 15 degrees
        "gray": ImageOps.grayscale(img).convert("RGB"),  # black and white
    }


# ------------------------------------------------------------ numbers
#ready-made model that can see pictures: it gives 1280 numbers for each face
FEATURE_MODEL = MobileNetV2(
    input_shape=(FACE_SIZE, FACE_SIZE, 3), weights="imagenet",
    include_top=False, pooling="avg",
)


def get_numbers(img):
    """Turn one face picture into 1280 numbers (an embedding)."""
    img = img.convert("RGB").resize((FACE_SIZE, FACE_SIZE))  #same size for all faces
    arr = np.array(img, dtype="float32")
    arr = preprocess_input(arr)                              #scale like the model wants
    arr = np.expand_dims(arr, axis=0)                        #the model wants a batch
    return FEATURE_MODEL.predict(arr, verbose=0)[0]


# ------------------------------------------------------------ who is it?
_face_model = None


def load_face_model():
    """Load the trained face model one time."""
    global _face_model
    if _face_model is None:
        _face_model = joblib.load(FACE_MODEL_PATH)
    return _face_model


def recognise_face(path):
    """Say who is in the photo and how sure the model is.

    Returns (member, confidence). Confidence is between 0 and 1.
    If no face is found, returns ("no_face", 0.0).
    """
    img = ImageOps.exif_transpose(Image.open(path))  #same upright fix as in training
    face = crop_face(img)

    #if the size didn' t change no face was found
    if face.size == img.size:
        return "no_face", 0.0

    numbers = get_numbers(face).reshape(1, -1)
    table = pd.DataFrame(numbers, columns=[f"f_{i}" for i in range(numbers.shape[1])])

    model = load_face_model()
    probs = model.predict_proba(table)[0]
    best = probs.argmax()
    return str(model.classes_[best]), float(probs[best])


# ------------------------------------------------------------ build + train
#everything below is only used to build image_features.csv and train the face model
EXPRESSIONS = ["neutral", "smiling", "surprised"]


def expression_of(path):
    """towi_smiling_2.jpg -> smiling"""
    for e in EXPRESSIONS:
        if e in path.stem.lower():
            return e
    return "unknown"


def build_feature_table(root=IMAGES_DIR):
    """One row per (photo, version): data/images/<member>/<member>_<expression>[_2].jpg"""
    rows = []
    for path in sorted(root.glob("*/*.jp*g")):
        img = ImageOps.exif_transpose(Image.open(path))  #phones save photos sideways, fix that first
        face = crop_face(img)
        if face.size == img.size:
            print(f"  no face found in {path.name}, using the whole photo")
        for version, pic in make_versions(face).items():
            numbers = get_numbers(pic)
            rows.append({"member": path.parent.name.lower(), "expression": expression_of(path),
                         "augmentation": version, "source_image": path.name,
                         **{f"f_{i}": v for i, v in enumerate(numbers)}})
    return pd.DataFrame(rows)


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))


def cross_validate(df):
    """Leave-one-photo-out: each round hides ONE photo and all its versions (flip, rotate, gray),
    so the model is never tested on a copy of a photo it trained on."""
    cols = [c for c in df.columns if c.startswith("f_")]
    X, y, groups = df[cols], df["member"], df["source_image"]
    classes = np.sort(y.unique())
    proba = np.zeros((len(df), len(classes)))
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        fitted = new_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        proba[test_idx] = fitted.predict_proba(X.iloc[test_idx])
    pred = classes[proba.argmax(axis=1)]

    def score(mask):
        return {"accuracy": accuracy_score(y[mask], pred[mask]),
                "f1_macro": f1_score(y[mask], pred[mask], average="macro"),
                "log_loss": log_loss(y[mask], proba[mask], labels=classes)}

    originals = (df["augmentation"] == "original").to_numpy()
    results = df[["member", "augmentation", "source_image"]].assign(
        predicted=pred, confidence=proba.max(axis=1))
    return {"all_rows": score(np.ones(len(df), bool)), "originals_only": score(originals)}, results


def main():
    df = build_feature_table()
    df.to_csv(IMAGE_FEATURES_CSV, index=False)
    print(f"Saved {df.shape} -> {IMAGE_FEATURES_CSV}")

    metrics, results = cross_validate(df)
    for k, d in metrics.items():
        print(k, {m: round(v, 3) for m, v in d.items()})

    cols = [c for c in df.columns if c.startswith("f_")]
    model = new_model().fit(df[cols], df["member"])
    joblib.dump(model, FACE_MODEL_PATH)
    print(f"Saved face model -> {FACE_MODEL_PATH}")

    #confidence check for the threshold: held-out member photos vs outsider photos
    genuine = results.query("augmentation == 'original' and predicted == member")["confidence"]
    outsiders = [recognise_face(p)[1] for p in sorted((UNAUTHORIZED_DIR / "images").glob("*.jp*g"))]
    metrics["threshold_check"] = {"genuine_heldout_min": float(genuine.min()),
                                  "genuine_heldout_median": float(genuine.median()),
                                  "outsider_confidences": [round(c, 3) for c in outsiders],
                                  "threshold_used": FACE_THRESHOLD}
    print("threshold check", metrics["threshold_check"])
    FACE_METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        main()  #no photo given: rebuild image_features.csv and retrain the face model
    else:
        print(recognise_face(sys.argv[1]))
