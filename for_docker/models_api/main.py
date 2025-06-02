from __future__ import annotations

"""Radon‑Forecast API (v1.2.1)

Исправлена синтаксическая ошибка, появившаяся при
автоматическом обновлении: лишний фрагмент `(self):`
после класса `PredictionRequest`.  Код полностью
переписан ниже без изменений логики.
"""
# ════════════════════════════════════════════════════════════════════════════
# Стандартная библиотека
# ════════════════════════════════════════════════════════════════════════════
from pathlib import Path
import configparser
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
    """Читает путь к корневому каталогу моделей."""

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
            raise FileNotFoundError(f"Создан шаблон {self.cfg_path}. Укажите путь в 'models_root'.")
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

        if not path.is_dir():
            raise NotADirectoryError(f"Каталог '{path}' не существует.")
        return path

# ════════════════════════════════════════════════════════════════════════════
# 2. Обёртка модели
# ════════════════════════════════════════════════════════════════════════════
class ModelBundle:
    """Хранит tf.keras‑модель, скейлер и метаданные."""

    def __init__(self, h5_path: Path, root: Path):
        self.rel_path = h5_path.relative_to(root)
        self.name = self.rel_path.with_suffix("").as_posix().replace("/", "__")

        self.model = tf.keras.models.load_model(h5_path, compile=False)
        self.input_shape = tuple(self.model.input_shape)
        self.output_shape = tuple(self.model.output_shape)
        self.scaler = self._load_scaler(h5_path)

    # ------------------------------------------------------------------
    def _load_scaler(self, h5_path: Path):
        pkl_path = h5_path.with_suffix(".pkl")
        if pkl_path.exists():
            with pkl_path.open("rb") as fh:
                return pickle.load(fh)
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
    def predict(self, batched: np.ndarray) -> List[float]:
        y_pred = self.model.predict(batched, verbose=0)
        if self.scaler is not None:
            y_pred = self.scaler.inverse_transform(y_pred)
        return y_pred.squeeze().tolist()

# ════════════════════════════════════════════════════════════════════════════
# 3. Реестр моделей
# ════════════════════════════════════════════════════════════════════════════
class ModelRegistry(dict):
    """Рекурсивно сканирует root и загружает *.h5‑модели."""

    def __init__(self, root: Path):
        super().__init__()
        self._load_all(root)

    def _load_all(self, root: Path):
        for h5 in root.rglob("*.h5"):
            try:
                bundle = ModelBundle(h5, root)
                if bundle.name in self:
                    print(f"[startup] Дубликат ключа '{bundle.name}', файл '{h5}'. Пропущен.")
                    continue
                self[bundle.name] = bundle
                print(f"[startup] + {bundle.name} (input={bundle.input_shape[1:]}, output={bundle.output_shape[1:]})")
            except Exception as exc:
                print(f"[startup]   ! Ошибка при загрузке '{h5}': {exc}")
        if not self:
            raise RuntimeError(f"В каталоге {root} не найдено ни одной .h5‑модели.")

# ════════════════════════════════════════════════════════════════════════════
# 4. Pydantic‑схема запроса
# ════════════════════════════════════════════════════════════════════════════
class PredictionRequest(BaseModel):
    """Допускает:
    • прямоугольный list[list[float]]  (N×F)
    • плоский list[float]             (F,)  → будет обёрнут в один timestep
    """

    sequence: List[float] | List[List[float]]

    @field_validator("sequence", mode="before")
    @classmethod
    def _normalize(cls, v):
        import numbers
        if isinstance(v, list):
            if all(isinstance(x, numbers.Number) for x in v):
                return [v]  # превращаем F,  в 1×F
            if all(isinstance(row, list) and all(isinstance(x, numbers.Number) for x in row) for row in v):
                row_len = len(v[0])
                if any(len(r) != row_len for r in v):
                    raise ValueError("Все строки должны иметь одинаковую длину.")
                return v
        raise ValueError("'sequence' должен быть списком чисел или списком списков одинаковой длины.")

# ════════════════════════════════════════════════════════════════════════════
# 5. Основной класс API
# ════════════════════════════════════════════════════════════════════════════
class RadonAPI:
    def __init__(self):
        self.config = Config()
        self.models = ModelRegistry(self.config.models_root)
        self.app = FastAPI(title="Radon‑Forecast API", version="1.2.1")
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
                "input_shape": bundle.input_shape[1:],
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
