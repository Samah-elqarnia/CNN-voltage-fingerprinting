from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

# ------------------------------- Configuration ------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "cnn_improved.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = [f"Class {i}" for i in range(9)]   # edit if your folders have other names
THRESHOLD = 0.90                                 # below this confidence -> "Unknown device"

st.set_page_config(page_title="Voltage Fingerprint CNN", page_icon="🔌")


# ----------------------- Improved CNN (same as training) --------------------
def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(0.25)(x)


def build_improved():
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = inp
    for f in (32, 64, 128, 256):
        x = conv_block(x, f)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return models.Model(inp, out, name="cnn_improved")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception:
        # Fallback if Keras versions differ: rebuild the architecture, load weights only
        model = build_improved()
        model.load_weights(str(MODEL_PATH))
        return model


def preprocess(img):
    # Same as training: RGB -> resize -> scale to [0, 1]
    arr = tf.convert_to_tensor(np.array(img.convert("RGB")), dtype=tf.float32)
    return (tf.image.resize(arr, IMG_SIZE) / 255.0).numpy()


# ----------------------------------- UI -------------------------------------
st.title("Hardware Fingerprinting")
st.caption("Upload an STFT spectrogram of a device's voltage signal "
           "and the CNN predicts which device produced it.")

if not MODEL_PATH.exists():
    st.error(f"Model not found: `{MODEL_PATH}`\n\n"
             "Put your downloaded weights at that location.")
    st.stop()

model = load_model()

file = st.file_uploader("Spectrogram image", type=["png", "jpg", "jpeg"])
if file is None:
    st.stop()

img = Image.open(file)
probs = model.predict(preprocess(img)[None], verbose=0)[0]
top = int(probs.argmax())
conf = float(probs[top])

col1, col2 = st.columns([1, 2])
col1.image(img, width=250)
with col2:
    if conf >= THRESHOLD:
        st.success(f"**{CLASS_NAMES[top]}**  (confidence {conf:.1%})")
    else:
        st.warning(f"**Unknown device**  (best guess: {CLASS_NAMES[top]}, "
                   f"confidence {conf:.1%})")
    st.bar_chart(pd.Series(probs, index=CLASS_NAMES, name="probability"))