"""maggr_models.joblib → maggr_models.npz: те же деревья без sklearn для образа платформы.

Есть ли scikit-learn в образе, мы не знаем. Детектор магнитки сначала пробует joblib
(sklearn), а если не вышло — берёт .npz (koz3.tree_models, только numpy). Новый
train_maggr.py пишет .npz сам; этот скрипт — для уже обученного joblib. Запуск там, где
стоит sklearn той же версии, что при обучении (машина обучения):

    python tools/export_trees.py --models submission/weights/maggr_models.joblib

Пишет maggr_models.npz рядом и сразу сверяет предсказания с sklearn.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("submission", "training"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from train_maggr import export_npz  # noqa: E402

from koz3 import tree_models  # noqa: E402


def check(result: dict, npz: Path, n: int = 20000, seed: int = 0) -> float:
    """Макс. расхождение вероятностей numpy и sklearn на случайных признаках (с NaN)."""
    trees, _ = tree_models.load_npz(npz)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for name in ("points", "polys"):
        model = result[name]
        x = rng.normal(size=(n, int(model.n_features_in_))) * 3
        x[rng.random(x.shape) < 0.03] = np.nan
        diff = np.abs(model.predict_proba(x) - trees[name].predict_proba(x)).max()
        worst = max(worst, float(diff))
    return worst


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(description="maggr_models.joblib → maggr_models.npz")
    parser.add_argument("--models", type=Path, required=True, help="путь к maggr_models.joblib")
    parser.add_argument("--out", type=Path, help="куда писать .npz (по умолчанию — рядом)")
    args = parser.parse_args(argv)
    result = joblib.load(args.models)
    out = args.out or args.models.with_name("maggr_models.npz")
    export_npz(result, out)
    worst = check(result, out)
    print(f"Записано {out}; расхождение с sklearn: {worst:.2e}")
    if worst > 1e-9:
        print("ОШИБКА: предсказания не совпадают — .npz не использовать")
        out.unlink()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
