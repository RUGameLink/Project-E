# 3. alt_models/2/alt2_cnn_lstm_wrapper.py
from pathlib import Path
import numpy as np
import tensorflow as tf

# ── ищем самый свежий весовой файл ─────────────────────────────────────────
_ROOT = Path(__file__).parent
cands = sorted(
    _ROOT.glob("model_cnn_lstm*.h5"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not cands:
    raise FileNotFoundError(
        "Не найдено ни одного файла «model_cnn_lstm_*.h5» в saved_models/"
    )
_WEIGHTS = cands[0]

# ── загружаем модель один раз при импорте ──────────────────────────────────
_model = tf.keras.models.load_model(_WEIGHTS, compile=False)

INPUT_SHAPE = _model.input_shape[1:]   # (T, F)
OUTPUT_SHAPE = _model.output_shape[1:] # (K,)  – обычно (1,)

# ── API, которое «понимает» ModelRegistry из main.py ───────────────────────
def prepare(raw: np.ndarray) -> np.ndarray:
    """
    • raw — двумерный массив (T×F) или серия «плоских» векторов;
    • здесь лишь преобразуем к float32 и добавляем batch-ось.
    """
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim == len(INPUT_SHAPE):
        x = x[None, ...]  # → (1, T, F)
    return x

def predict(batched: np.ndarray):
    """Возвращает list/float, пригодный для сериализации в JSON."""
    y = _model.predict(batched, verbose=0)
    return y.squeeze().tolist()
