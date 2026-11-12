"""Градиентный бустинг sklearn без sklearn: деревья в .npz и предсказание на numpy.

Есть ли scikit-learn в образе платформы, мы не знаем. Поэтому модель
HistGradientBoostingClassifier (бинарная, log_loss) выгружается при обучении в .npz —
только массивы numpy, без pickle, — и NumpyHGB предсказывает по тем же деревьям ровно
так же, как sklearn: числовой сплит «x <= порог → влево», NaN — по missing_go_to_left,
вероятность класса 1 = sigmoid(baseline + сумма листьев).

Выгрузка (export_hgb, save_npz) нужна sklearn — она идёт на машине обучения
(training/train_maggr.py, tools/export_trees.py). Чтение и предсказание (load_npz,
NumpyHGB) — только numpy.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

_FIELDS: tuple[str, ...] = (
    "feature",
    "threshold",
    "missing_left",
    "left",
    "right",
    "leaf",
    "value",
)
"""Массивы узлов одной модели (все деревья подряд; индексы детей — внутри своего дерева)."""


class NumpyHGB:
    """Бинарный HistGradientBoostingClassifier из массивов узлов; predict_proba как в sklearn."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.baseline = float(arrays["baseline"])
        self.n_features = int(arrays["n_features"])
        self.starts = np.asarray(arrays["tree_start"], dtype=np.int64)
        self.feature = np.asarray(arrays["feature"], dtype=np.int64)
        self.threshold = np.asarray(arrays["threshold"], dtype=np.float64)
        self.missing_left = np.asarray(arrays["missing_left"], dtype=bool)
        self.left = np.asarray(arrays["left"], dtype=np.int64)
        self.right = np.asarray(arrays["right"], dtype=np.int64)
        self.leaf = np.asarray(arrays["leaf"], dtype=bool)
        self.value = np.asarray(arrays["value"], dtype=np.float64)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Сырой выход: baseline + сумма значений листьев по всем деревьям."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"признаков {x.shape}, модель ждёт (n, {self.n_features})")
        raw = np.full(x.shape[0], self.baseline)
        rows = np.arange(x.shape[0])
        for start in self.starts:
            node = np.full(x.shape[0], start, dtype=np.int64)
            for _ in range(self.feature.size):  # глубина дерева ≤ числа узлов
                active = ~self.leaf[node]
                if not active.any():
                    break
                idx = node[active]
                values = x[rows[active], self.feature[idx]]
                go_left = np.where(
                    np.isnan(values), self.missing_left[idx], values <= self.threshold[idx]
                )
                node[active] = start + np.where(go_left, self.left[idx], self.right[idx])
            raw += self.value[node]
        return raw

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """(n, 2): вероятности классов 0 и 1, как HistGradientBoostingClassifier."""
        p1 = 1.0 / (1.0 + np.exp(-self.decision_function(x)))
        return np.column_stack([1.0 - p1, p1])


def export_hgb(model: Any) -> dict[str, np.ndarray]:
    """Массивы узлов обученного бинарного HistGradientBoostingClassifier (нужен sklearn)."""
    if int(getattr(model, "n_trees_per_iteration_", 0)) != 1:
        raise ValueError("поддерживается только бинарная классификация (1 дерево на итерацию)")
    if getattr(model, "is_categorical_", None) is not None and np.any(model.is_categorical_):
        raise ValueError("категориальные признаки не поддерживаются")
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"классы {list(model.classes_)}, ожидались [0, 1]")
    parts: dict[str, list[np.ndarray]] = {name: [] for name in _FIELDS}
    starts = []
    offset = 0
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        starts.append(offset)
        parts["feature"].append(nodes["feature_idx"].astype(np.int64))
        parts["threshold"].append(nodes["num_threshold"].astype(np.float64))
        parts["missing_left"].append(nodes["missing_go_to_left"].astype(bool))
        parts["left"].append(nodes["left"].astype(np.int64))
        parts["right"].append(nodes["right"].astype(np.int64))
        parts["leaf"].append(nodes["is_leaf"].astype(bool))
        parts["value"].append(nodes["value"].astype(np.float64))
        offset += len(nodes)
    arrays = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    arrays["tree_start"] = np.asarray(starts, dtype=np.int64)
    arrays["baseline"] = np.asarray(float(np.ravel(model._baseline_prediction)[0]))
    arrays["n_features"] = np.asarray(int(model.n_features_in_))
    return arrays


def save_npz(path: Path, models: dict[str, Any], meta: dict[str, Any]) -> Path:
    """Модели sklearn {имя: HGB} и метаданные (JSON) → один .npz без pickle."""
    payload: dict[str, np.ndarray] = {"meta": np.asarray(json.dumps(meta, ensure_ascii=False))}
    for name, model in models.items():
        for key, array in export_hgb(model).items():
            payload[f"{name}/{key}"] = array
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:  # np.savez к имени без .npz добавил бы расширение
        np.savez_compressed(handle, **payload)
    return path


def load_npz(path: Path) -> tuple[dict[str, NumpyHGB], dict[str, Any]]:
    """.npz из save_npz → ({имя: NumpyHGB}, метаданные); только numpy, без pickle."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        grouped: dict[str, dict[str, np.ndarray]] = {}
        for key in data.files:
            if "/" in key:
                name, field = key.split("/", 1)
                grouped.setdefault(name, {})[field] = data[key]
    return {name: NumpyHGB(arrays) for name, arrays in grouped.items()}, meta
