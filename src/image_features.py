"""Face part of the system: cut out a face, turn it into numbers and say who it is.

Used by the notebook and by the command line app.
Main function for the app:
    recognise_face(path) -> (member, confidence)

Run:  python src/image_features.py <photo path>
"""
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import FACE_BORDER, FACE_MODEL_PATH, FACE_SIZE

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
    img = Image.open(path)
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/image_features.py <photo path>")
    else:
        print(recognise_face(sys.argv[1]))
