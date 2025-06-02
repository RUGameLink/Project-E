from __future__ import annotations

"""
Radon-Forecast API (v1.3.0)

• Доработка: кроме *.h5-файлов, сервер теперь умеет подхватывать
  Python-модули, содержащие функцию predict() (см. PyModelBundle ниже).
  Это даёт возможность быстро разворачивать модели без обязательного
  сохранения их весов в формате .h5.
• Поведение старых маршрутов полностью сохранено.
"""

# ════════════════════════════════════════════════════════════════════════════
# Стандартная библиотека
# ════════════════════════════════════════════════════════════════════════════
from pathlib import Path
import configparser
import importlib.util
import pickle
from typing import List, Dict
import os

# ════════════════════════════════════════════════════════════════════════════
# Сторонние зависимости
# ════════════════════════════════════════════════════════════════════════════
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ════════════════════════════════════════════════════════════════════════════
# 1. Конфигурация
# ════════════════════════════════════════════════════════════════════════════
class Config:
    """Читает путь к каталогу моделей из config.ini или переменной MODELS_DIR."""

    _SECTION = "paths"
    _KEY = "models_root"

    def __init__(self, cfg_path: Path | str | None = None):
        self.cfg_path = Path(cfg_path) if cfg_path else Path(__file__).with_name("config.ini")
        self._parser = configparser.ConfigParser()
        self._read()

    # ------------------------------------------------------------------
    def _read(self) -> None:
        if not self.cfg_path.exists():
            template = "[paths]\nmodels_root = models\n"
            self.cfg_path.write_text(template, encoding="utf-8")
            raise FileNotFoundError(
                f"Создан шаблон {self.cfg_path}. Укажите путь в 'models_root'."
            )
        self._parser.read(self.cfg_path, encoding="utf-8")

    # ------------------------------------------------------------------
    @property
    def models_root(self) -> Path:
        env_val = os.getenv("MODELS_DIR")
        if env_val:
            path = Path(env_val).expanduser().resolve()
        else:
            try:
                path = Path(self._parser[self._SECTION][self._KEY]).expanduser().resolve()
            except KeyError as exc:
                raise KeyError(f"В config.ini отсутствует [{self._SECTION}]/{self._KEY}.") from exc

        if not path.exists():
            raise FileNotFoundError(f"Каталог моделей '{path}' не найден.")
        return path


# ════════════════════════════════════════════════════════════════════════════
# 2. Обёртки моделей
# ════════════════════════════════════════════════════════════════════════════
class ModelBundle:
    """Классическое хранилище для tf.keras-модели, её скейлера и метаданных."""

    def __init__(self, h5_path: Path, root: Path):
        self.rel_path = h5_path.relative_to(root)
        self.name = self.rel_path.with_suffix("").as_posix().replace("/", "__")

        self.model = tf.keras.models.load_model(h5_path, compile=False)
        self.input_shape = tuple(self.model.input_shape)
        self.output_shape = tuple(self.model.output_shape)
        self.scaler = self._load_scaler(h5_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_scaler(h5_path: Path):
        pkl = h5_path.with_suffix(".pkl")
        if pkl.exists():
            try:
                return pickle.loads(pkl.read_bytes())
            except Exception as exc:
                print(f"[startup]   ! Не удалось загрузить scaler '{pkl}': {exc}")
        return None

    # ------------------------------------------------------------------
    def prepare(self, raw: np.ndarray) -> np.ndarray:
        expected = self.input_shape[1:]
        if raw.shape != expected:
            raise ValueError(f"Неверная форма входа: ожидалась {expected}, получена {raw.shape}.")
        x = raw.astype(np.float32)
        if self.scaler is not None and x.ndim == 2:
            t, f = x.shape
            x = self.scaler.transform(x.reshape(t, f)).reshape(t, f)
        return np.expand_dims(x, axis=0)

    # ------------------------------------------------------------------
    def predict(self, batched: np.ndarray):
        y = self.model.predict(batched, verbose=0)
        return y.squeeze().tolist()


# ---------------------------------------------------------------------------
class PyModelBundle:
    """
    Загружает произвольный *.py-файл как модуль и ищет в нём функцию predict().
    Модуль может (но не обязан) объявить:
        • prepare(raw: np.ndarray) -> np.ndarray
        • INPUT_SHAPE / OUTPUT_SHAPE (tuple)
    """

    def __init__(self, py_path: Path, root: Path):
        self.rel_path = py_path.relative_to(root)
        self.name = self.rel_path.with_suffix("").as_posix().replace("/", "__")

        spec = importlib.util.spec_from_file_location(self.name, py_path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore[arg-type]

        # обязательное API
        if not hasattr(module, "predict"):
            raise AttributeError(f"В «{py_path}» нет функции predict()")
        self._predict_fn = module.predict

        # необязательное
        self._prepare_fn = getattr(module, "prepare", None)
        self.input_shape = getattr(module, "INPUT_SHAPE", (None,))
        self.output_shape = getattr(module, "OUTPUT_SHAPE", (None,))

    # ------------------------------------------------------------------
    def prepare(self, raw: np.ndarray) -> np.ndarray:
        if self._prepare_fn is not None:
            return self._prepare_fn(raw)
        # по умолчанию – просто добавляем batch-ось
        return np.expand_dims(raw.astype(np.float32), axis=0)

    # ------------------------------------------------------------------
    def predict(self, batched: np.ndarray):
        return self._predict_fn(batched)


# ════════════════════════════════════════════════════════════════════════════
# 3. Реестр моделей
# ════════════════════════════════════════════════════════════════════════════
class ModelRegistry(dict):
    """Рекурсивно сканирует root и подключает *.h5- и *.py-модели."""

    _PY_EXCLUDE = {"__init__.py", Path(__file__).name}

    def __init__(self, root: Path):
        super().__init__()
        self._load_all(root)

    # ------------------------------------------------------------------
    def _try_add(self, path: Path, wrapper_cls, root: Path):
        try:
            bundle = wrapper_cls(path, root)
            if bundle.name in self:
                print(f"[startup]   ⚠ Дубликат ключа '{bundle.name}', файл '{path}'. Пропущен.")
                return
            self[bundle.name] = bundle
            inp = "?" if bundle.input_shape is None else bundle.input_shape
            out = "?" if bundle.output_shape is None else bundle.output_shape
            print(f"[startup] + {bundle.name} (input={inp}, output={out})")
        except Exception as exc:
            print(f"[startup]   ! Ошибка при загрузке '{path}': {exc}")

    # ------------------------------------------------------------------
    def _load_all(self, root: Path):
        # 1) Keras-модели
        for h5 in root.rglob("*.h5"):
            self._try_add(h5, ModelBundle, root)

        # 2) Python-модули
        for py in root.rglob("*.py"):
            if py.name in self._PY_EXCLUDE:
                continue
            self._try_add(py, PyModelBundle, root)

        if not self:
            raise RuntimeError(f"В каталоге {root} не найдено ни одной модели.")


# ════════════════════════════════════════════════════════════════════════════
# 4. Pydantic-схема запроса
# ════════════════════════════════════════════════════════════════════════════
class PredictionRequest(BaseModel):
    """
    Допускает:
        • прямоугольный list[list[float]]  (N×F)
        • плоский list[float]              (F,) → будет обёрнут в один timestep
    """

    sequence: List[float] | List[List[float]]

    @field_validator("sequence", mode="before")
    @classmethod
    def _normalize(cls, v):
        import numbers

        if isinstance(v, list):
            # плоский one-step вектор F  →  [[F]]
            if all(isinstance(x, numbers.Number) for x in v):
                return [v]
            # проверяем прямоугольность
            if all(isinstance(row, list) and row for row in v):
                width = len(v[0])
                if all(len(row) == width and all(isinstance(x, numbers.Number) for x in row) for row in v):
                    return v
        raise ValueError("sequence должен быть list[float] или прямоугольным list[list[float]].")


# ════════════════════════════════════════════════════════════════════════════
# 5. Приложение FastAPI
# ════════════════════════════════════════════════════════════════════════════
class RadonAPI:
    def __init__(self):
        self.config = Config()
        self.models = ModelRegistry(self.config.models_root)
        self.app = FastAPI(title="Radon-Forecast API", version="1.3.0")
        self._register_routes()

    # ------------------------------------------------------------------
    def _register_routes(self):
        @self.app.get("/")
        async def root():
            return {
                "service": self.app.title,
                "version": self.app.version,
                "available_models": list(self.models.keys()),
            }

        @self.app.post("/predict/{model_key}")
        async def predict(model_key: str, payload: PredictionRequest):
            if model_key not in self.models:
                raise HTTPException(status_code=404, detail="Модель не найдена.")
            bundle = self.models[model_key]
            try:
                arr = np.asarray(payload.sequence, dtype=np.float32)
                batched = bundle.prepare(arr)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            pred = bundle.predict(batched)
            return {
                "model": model_key,
                "input_shape": bundle.input_shape[1:] if bundle.input_shape else None,
                "prediction": pred,
            }


# ════════════════════════════════════════════════════════════════════════════
# 6. Точка входа
# ════════════════════════════════════════════════════════════════════════════
api = RadonAPI()
app = api.app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
