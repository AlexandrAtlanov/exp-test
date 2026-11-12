"""tools/tune_thresholds.py: перебор порогов на готовых детекциях и метрике организаторов."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pytest
from shapely.geometry import box, mapping

import tune_thresholds
from koz3 import geometry
from koz3.datatypes import Detection, Region

UTM = "EPSG:32637"
REGION = Region(name="R", path=Path("R"), utm_crs=UTM, files={})


def _det(x: float, conf: float, cls: str = "kurgany_tselye") -> Detection:
    return Detection(
        geom=box(500000 + x, 6000000, 500000 + x + 10, 6000010),
        class_name=cls,
        confidence=conf,
        markup_type="li",
        region_name="R",
    )


def _gt() -> dict:
    """Разметка: курганы в x = 0 и x = 100 (в 3857, как у организаторов)."""
    features = []
    for x in (0, 100):
        poly = geometry.reproject(box(500000 + x, 6000000, 500010 + x, 6000010), UTM, "EPSG:3857")
        features.append(
            {
                "type": "Feature",
                "properties": {"class_name": "kurgany_tselye", "markup_type": "li"},
                "geometry": mapping(poly),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def test_sweep_picks_threshold_that_drops_low_confidence_fp(tmp_path: Path, monkeypatch) -> None:
    detections = [_det(0, 0.9), _det(100, 0.3)] + [_det(300 + 30 * i, 0.12) for i in range(6)]
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(_gt()), encoding="utf-8")
    raw = tmp_path / "raw.pkl"
    calls = []

    def fake_collect(input_dir):
        calls.append(input_dir)
        return detections, {"R": REGION}

    monkeypatch.setattr(tune_thresholds, "collect", fake_collect)
    args = ["--input", str(tmp_path), "--gt", str(gt_path), "--raw", str(raw)]
    assert tune_thresholds.main(args + ["--grid", "0.1", "0.15", "0.5"]) == 0
    assert raw.is_file() and len(calls) == 1
    assert tune_thresholds.main(args) == 0  # второй раз — из кэша, детекторы не запускаются
    assert len(calls) == 1

    from evaluate import _load_geojson, load_metric

    metric = load_metric(Path(tune_thresholds.__file__).parent / "compute_metrics.py")
    gt = _load_geojson(gt_path)
    regions = {"R": REGION}
    params, result = tune_thresholds.search(
        detections, regions, gt, metric, "thr", (0.1, 0.15, 0.5)
    )
    assert params.conf == {"kurgany_tselye": 0.15}  # 0.1 — 6 FP; 0.5 — теряется курган с 0.3
    assert result["final_score"] == pytest.approx(1.0, abs=1e-3)
    low = tune_thresholds.score(
        detections, regions, gt, metric, tune_thresholds.Params(conf={"kurgany_tselye": 0.1})
    )
    assert low["final_score"] < result["final_score"]


def test_full_search_tunes_area_and_nms(tmp_path: Path) -> None:
    """full: уверенный мелкий FP отсекает минимальная площадь; config не меняется."""
    from evaluate import _load_geojson, load_metric
    from koz3 import config

    tiny = Detection(
        geom=box(500400, 6000000, 500401.5, 6000001.5),  # 2.25 м² — больше MIN_AREA 2, меньше 4
        class_name="kurgany_tselye",
        confidence=0.9,
        markup_type="li",
        region_name="R",
    )
    detections = [_det(0, 0.9), _det(100, 0.9), tiny]
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(_gt()), encoding="utf-8")
    metric = load_metric(Path(tune_thresholds.__file__).parent / "compute_metrics.py")
    before = (dict(config.MIN_AREA_M2), config.NMS_IOU, config.NMS_COVER)
    params, result = tune_thresholds.search(
        detections, {"R": REGION}, _load_geojson(gt_path), metric, "full",
        (0.5,), (1.0, 2.0), (0.5,), (0.7,),
    )  # fmt: skip
    assert params.min_area["kurgany_tselye"] == pytest.approx(
        2 * config.MIN_AREA_M2["kurgany_tselye"]
    )
    assert result["final_score"] == pytest.approx(1.0, abs=1e-3)
    assert (dict(config.MIN_AREA_M2), config.NMS_IOU, config.NMS_COVER) == before
    block = tune_thresholds.config_block(params, "full")
    assert "MIN_AREA_M2" in block and "NMS_IOU: float = 0.5" in block


def test_apply_writes_file_only_with_gain(tmp_path: Path, monkeypatch) -> None:
    """--apply пишет weights/postprocess.json при приросте ≥ --min-gain, иначе нет."""
    from koz3 import config, tuning

    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights)
    detections = [_det(0, 0.9), _det(100, 0.3)] + [_det(300 + 30 * i, 0.12) for i in range(6)]
    monkeypatch.setattr(tune_thresholds, "collect", lambda d: (detections, {"R": REGION}))
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(_gt()), encoding="utf-8")
    args = ["--input", str(tmp_path), "--gt", str(gt_path), "--grid", "0.1", "0.15", "--apply"]

    monkeypatch.setattr(config, "DEFAULT_CONF_THRESHOLD", 0.1)  # «сейчас» хуже подбора
    assert tune_thresholds.main(args + ["--min-gain", "2.0"]) == 0  # прирост < 2 — не пишем
    assert not (weights / tuning.FILE_NAME).exists()
    assert tune_thresholds.main(args) == 0
    data = json.loads((weights / tuning.FILE_NAME).read_text(encoding="utf-8"))
    assert data["conf_threshold"] == {"kurgany_tselye|li": 0.15}
    assert data["holdout_final_score"] > data["config_final_score"]
