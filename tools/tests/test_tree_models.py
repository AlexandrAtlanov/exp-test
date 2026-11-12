"""koz3.tree_models: деревья HistGradientBoosting в .npz предсказывают ровно как sklearn."""

from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pytest

from koz3 import tree_models

sklearn_ensemble = pytest.importorskip("sklearn.ensemble")


def _fit(n_features: int, seed: int, **params):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3000, n_features))
    x[rng.random(x.shape) < 0.05] = np.nan  # пропуски: missing_go_to_left
    y = ((np.nan_to_num(x[:, 0]) + 0.5 * np.nan_to_num(x[:, 1]) ** 2) > 0.3).astype(int)
    model = sklearn_ensemble.HistGradientBoostingClassifier(
        max_iter=60, random_state=seed, class_weight="balanced", **params
    )
    return model.fit(x, y), rng


@pytest.mark.parametrize(("n_features", "params"), [(16, {}), (15, {"early_stopping": True})])
def test_numpy_matches_sklearn(tmp_path: Path, n_features: int, params: dict) -> None:
    model, rng = _fit(n_features, 7, **params)
    path = tree_models.save_npz(tmp_path / "m.npz", {"points": model}, {"thr": 0.65})
    models, meta = tree_models.load_npz(path)
    assert meta == {"thr": 0.65}
    x = rng.normal(size=(5000, n_features)) * 2
    x[rng.random(x.shape) < 0.05] = np.nan
    nodes = model._predictors[0][0].nodes
    split = ~nodes["is_leaf"].astype(bool)
    x[:10, nodes["feature_idx"][split][0]] = nodes["num_threshold"][split][0]  # ровно на пороге
    expected = model.predict_proba(x)
    actual = models["points"].predict_proba(x)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_npz_has_no_pickle(tmp_path: Path) -> None:
    model, _ = _fit(4, 1)
    path = tree_models.save_npz(tmp_path / "maggr_models.npz", {"polys": model}, {})
    assert path.name == "maggr_models.npz"
    with np.load(path, allow_pickle=False) as data:  # объектных массивов нет
        assert all(data[key].dtype != object for key in data.files)


def test_wrong_feature_count_rejected(tmp_path: Path) -> None:
    model, _ = _fit(4, 2)
    models, _ = tree_models.load_npz(tree_models.save_npz(tmp_path / "m.npz", {"a": model}, {}))
    with pytest.raises(ValueError):
        models["a"].predict_proba(np.zeros((3, 5)))


def test_export_trees_cli(tmp_path: Path) -> None:
    """tools/export_trees.py: joblib → .npz рядом, сверка с sklearn, код 0."""
    joblib = pytest.importorskip("joblib")
    import export_trees

    from koz3.geophys import maggr_features as mf

    rng = np.random.default_rng(3)
    result: dict = {"thr_points": 0.65, "thr_polys": 0.5, "sklearn": "test"}
    result["feature_names"] = {
        "points": list(mf.POINT_FEATURE_NAMES),
        "polys": list(mf.POLY_FEATURE_NAMES),
    }
    for key, names in (("points", mf.POINT_FEATURE_NAMES), ("polys", mf.POLY_FEATURE_NAMES)):
        x = rng.normal(size=(300, len(names)))
        result[key] = sklearn_ensemble.HistGradientBoostingClassifier(max_iter=10).fit(
            x, (x[:, 1] > 0).astype(int)
        )
    src = tmp_path / "maggr_models.joblib"
    joblib.dump(result, src)
    assert export_trees.main(["--models", str(src)]) == 0
    _, meta = tree_models.load_npz(tmp_path / "maggr_models.npz")
    assert meta["thr_points"] == 0.65 and meta["feature_names"] == result["feature_names"]
