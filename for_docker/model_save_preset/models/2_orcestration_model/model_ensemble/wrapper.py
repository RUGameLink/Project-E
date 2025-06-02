# 3. alt_models/2/alt2_ensemble_wrapper.py
"""
Эта обёртка поддерживает две схемы:

1) **Одиночный файл** «model_ensemble_*.h5»  
   → загружается как обычная Keras-модель;

2) **Несколько файлов** «model_ensemble_part*.h5»  
   → каждую модель грузим отдельно и усредняем их предсказания.
"""
from pathlib import Path
import numpy as np
import tensorflow as tf

_ROOT = Path(__file__).parent
# ── 2.1: пробуем «monolithic» веса ─────────────────────────────────────────
mono = sorted(
    _ROOT.glob("model_ensemble*.h5"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

# ── 2.2: пробуем «component»-веса ──────────────────────────────────────────
parts = sorted(
    _ROOT.glob("model_ensemble*.h5"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if mono:
    # ─Адекватный монолитный .h5 найден
    MODELS = [tf.keras.models.load_model(mono[0], compile=False)]
elif parts:
    # ─Композит: загружаем все найденные части
    MODELS = [tf.keras.models.load_model(p, compile=False) for p in parts]
else:
    raise FileNotFoundError(
        "Не найдено ни одного файла «model_ensemble_*.h5» "
        "или «model_ensemble_part*.h5» в saved_models/"
    )

INPUT_SHAPE = MODELS[0].input_shape[1:]
OUTPUT_SHAPE = MODELS[0].output_shape[1:]

def _to_batch(x: np.ndarray) -> np.ndarray:
    """Обеспечивает ось batch, dtype float32 и правильную размерность."""
    x = np.asarray(x, dtype=np.float32)
    return x[None, ...] if x.ndim == len(INPUT_SHAPE) else x

def prepare(raw: np.ndarray) -> np.ndarray:           # для совместимости
    return _to_batch(raw)

def _avg_preds(batched: np.ndarray) -> np.ndarray:
    """Усредняем предсказания всех под-моделей по оси моделей."""
    preds = [m.predict(batched, verbose=0) for m in MODELS]  # list of (B, …)
    return np.mean(preds, axis=0)

def predict(batched: np.ndarray):
    y = _avg_preds(batched)
    return y.squeeze().tolist()
