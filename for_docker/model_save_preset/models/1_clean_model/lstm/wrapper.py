# 1. old_models/lstm_wrapper.py
from pathlib import Path
import numpy as np
import tensorflow as tf

_WEIGHTS = Path(__file__).with_name("lstm.h5")
_model   = tf.keras.models.load_model(_WEIGHTS, compile=False)

INPUT_SHAPE  = _model.input_shape[1:]
OUTPUT_SHAPE = _model.output_shape[1:]

def prepare(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float32)
    return np.expand_dims(x, axis=0) if x.ndim == len(INPUT_SHAPE) else x

def predict(batched: np.ndarray):
    return _model.predict(batched, verbose=0).squeeze().tolist()