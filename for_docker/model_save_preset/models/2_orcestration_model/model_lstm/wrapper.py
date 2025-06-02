# 3. alt_models/1/alt1_lstm_wrapper.py
from pathlib import Path
import numpy as np
import tensorflow as tf

_ROOT = Path(__file__).parent
cands = sorted(_ROOT.glob("model_lstm*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
if not cands:
    raise FileNotFoundError("Не найдено ни одного веса LSTM в saved_models/")
_WEIGHTS = cands[0]

_model = tf.keras.models.load_model(_WEIGHTS, compile=False)
INPUT_SHAPE, OUTPUT_SHAPE = _model.input_shape[1:], _model.output_shape[1:]

def prepare(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float32)
    return np.expand_dims(x, axis=0) if x.ndim == len(INPUT_SHAPE) else x

def predict(batched: np.ndarray):
    return _model.predict(batched, verbose=0).squeeze().tolist()
