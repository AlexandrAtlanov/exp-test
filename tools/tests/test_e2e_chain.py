"""Сквозные тесты всей цепочки на синтетике: LAS → solution → GeoJSON → make_gt → метрика.

Синтетический LAS с холмом (write_synthetic_las из test_lidar_dem) проходит через
настоящие discovery, lidar_dem, lidar, postprocess и output. YOLO заменена
«детектором холма» по яркости отмывки «g»: проверяется не качество модели, а то,
что полигон оказывается в правильном месте карты и засчитывается метрикой
организаторов (tools/compute_metrics.py).
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pytest
import rasterio
import shapely
from pyproj import Transformer
from shapely.geometry import Point, mapping, shape

pytest.importorskip("cv2")  # lidar_dem строит отмывки через OpenCV
ndimage = pytest.importorskip("scipy.ndimage")

from test_lidar_dem import write_synthetic_las  # noqa: E402

import make_gt  # noqa: E402
import solution  # noqa: E402
import validate_output  # noqa: E402
from koz3 import config, geometry, output, yolo_runtime  # noqa: E402
from koz3.datatypes import Detection  # noqa: E402
from koz3.detectors import lidar, lidar_dem  # noqa: E402
from koz3.yolo_runtime import TileInstance  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[2]
UTM = "EPSG:32637"
"""Зона синтетического региона: x 500 000 — осевой меридиан, координаты 3857 реальные."""

HILL_X, HILL_Y = 500020.0, 6000015.0
"""Центр холма в write_synthetic_las, UTM, м."""

G_BACKGROUND = 181
"""Яркость ровной земли в отмывке «g» этого LAS (замер)."""


def _hill_mask(image: np.ndarray) -> np.ndarray:
    """Пиксели холма: заметно отличаются от фона отмывки «g» (попиксельное правило)."""
    gray = image[..., 0].astype(int)
    valid = ~(np.all(image == 0, axis=-1) | np.all(image == 255, axis=-1))
    return valid & (np.abs(gray - G_BACKGROUND) > 40)


def _fake_predict(_model, tiles):
    """«Модель»: по экземпляру на каждое связное пятно холма в тайле."""
    for image in tiles:
        labels, count = ndimage.label(_hill_mask(image))
        found = []
        for k in range(1, count + 1):
            part = labels == k
            if part.sum() >= 50:
                found.append(TileInstance("kurgany_tselye", 0.9, part))
        yield found


def _mk(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_input(root: Path) -> Path:
    """Вход «как на платформе»: регион R01 с LAS и UTM.json."""
    region = root / "input" / "R01"
    write_synthetic_las(_mk(region / "01_R01_La_точки") / "R01_La.las")
    (region / "UTM.json").write_text(json.dumps({"crs": UTM}), encoding="utf-8")
    return root / "input"


def _make_train(root: Path, dx_m: float = 0.0) -> Path:
    """Трейн с разметкой: курган — круг R = 10 м вокруг холма (склоны холма, UTM, сдвиг dx_m)."""
    data = root / f"train_{int(dx_m)}"
    circle = Point(HILL_X + dx_m, HILL_Y).buffer(10.0)
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": UTM}},
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(circle),
                "properties": {"class_name": "Курганы"},
            }
        ],
    }
    path = _mk(data / "R01" / "06_R01_разметка") / "R01_Li_курганы.geojson"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return data


def _score(pred_path: Path, gt_path: Path) -> dict:
    """Метрика организаторов (загрузка по пути, дефолты платформы)."""
    spec = importlib.util.spec_from_file_location(
        "koz3_metric", REPO_DIR / "tools" / "compute_metrics.py"
    )
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    assert metric.PLATFORM_CONFIG_LOADED is False
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    return metric.F1Calculator().calculate_f1(pred, gt)


def _to_utm(geom):
    to_utm = Transformer.from_crs("EPSG:3857", UTM, always_xy=True)
    return shapely.transform(geom, lambda xy: np.column_stack(to_utm.transform(*xy.T)))


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Подмены: фиктивные веса, детектор холма вместо YOLO, тайл 256 px (4 тайла)."""
    weights = _mk(tmp_path / "weights")
    (weights / str(config.YOLO["weights_pt"])).write_bytes(b"fake")
    monkeypatch.setattr(config, "WEIGHTS_DIR", weights)
    monkeypatch.setitem(config.LIDAR, "tile_px", 256)
    monkeypatch.setitem(config.LIDAR, "infer_hillshades", ("g",))
    stub = yolo_runtime.LoadedModel(model=object(), kind="stub", names={}, batch=8)
    monkeypatch.setattr(lidar, "_load_model_once", lambda: stub)
    monkeypatch.setattr(yolo_runtime, "predict_tiles", _fake_predict)
    return tmp_path


def test_chain_polygon_lands_where_hill_is(chain):
    """Полигоны после тайлов и postprocess совпадают с эталоном без тайлов; метрика — TP."""
    inp = _make_input(chain)
    out_dir = chain / "out"  # несуществующая папка → out/result.geojson
    assert solution.main([str(inp), str(out_dir)]) == 0
    result = out_dir / "result.geojson"
    assert validate_output.validate(result)["defects"] == []
    features = json.loads(result.read_text(encoding="utf-8"))["features"]
    assert features
    assert {f["geometry"]["type"] for f in features} == {"Polygon"}
    assert {f["properties"]["markup_type"] for f in features} == {"li"}

    # эталон: та же маска на ЦЕЛОМ растре, без тайлов
    las = next(inp.rglob("*.las"))
    paths = lidar_dem.hillshades_for_las(las, UTM, chain / "ref")
    g_path = next(p for p in paths if p.stem.endswith("_g"))
    min_area_m2 = config.MIN_AREA_M2["kurgany_tselye"]  # мелкие пятна postprocess отсеивает
    with rasterio.open(g_path) as src:
        mask = _hill_mask(yolo_runtime.to_model_image(src.read()))
        polygons = geometry.mask_to_polygons(mask, src.transform)
    ref = shapely.unary_union([p for p in polygons if p.area >= min_area_m2])
    got = shapely.unary_union([_to_utm(shape(f["geometry"])) for f in features])
    iou = got.intersection(ref).area / got.union(ref).area
    # без тайлов IoU ≈ 0.88 (упрощение контура 0.3 м в postprocess); обрезок на стыке тайлов
    # вместо целого объекта даёт ≈ 0.71, сдвиг CRS — около 0
    assert iou > 0.85, f"IoU с эталоном {iou:.3f}: тайлы, NMS или CRS портят полигоны"
    assert got.centroid.distance(ref.centroid) < 0.5

    gt_path = chain / "gt.geojson"
    args = ["--data", str(_make_train(chain)), "--regions", "R01", "--out", str(gt_path)]
    assert make_gt.main(args) == 0
    assert _score(result, gt_path)["per_class"]["kurgany_tselye"]["tp"] >= 1

    shifted = chain / "gt_shifted.geojson"
    args = ["--data", str(_make_train(chain, 40.0)), "--regions", "R01", "--out", str(shifted)]
    make_gt.main(args)
    assert _score(result, shifted)["final_score"] == 0.0


def test_chain_zero_budget_writes_valid_file(chain, monkeypatch):
    """Бюджет 0 с: детекторы пропущены, валидный файл всё равно записан."""
    monkeypatch.setattr(config, "TIME_LIMIT_S", 0)
    out = chain / "r.geojson"
    assert solution.main([str(_make_input(chain)), str(out)]) == 0
    assert validate_output.validate(out)["defects"] == []


def test_train_layout_hillshades_and_annotations_not_inputs(tmp_path):
    """Готовые отмывки Li и разметка трейна не становятся входом детекторов."""
    from koz3 import discovery

    inp = _make_input(tmp_path)
    region = inp / "R01"
    (_mk(region / "02_R01_Li_карты") / "R01_Li_c.tif").write_bytes(b"x")
    (_mk(region / "06_R01_разметка") / "R01_Li_курганы.geojson").write_text("{}")
    (_mk(region / "05_R01_Or_ортофото") / "R01_Or.tif").write_bytes(b"x")
    [found] = discovery.find_regions(inp)
    assert [p.suffix for p in found.files["li"]] == [".las"]
    assert found.utm_crs == UTM
    assert "or" in found.files


def test_finds_points_circle_matches_gt_point(tmp_path):
    """Круг finds_points из output.py засчитывается против точки GT (допуск метрики 2 м)."""
    det = Detection(geometry.point_to_circle(HILL_X, HILL_Y), "finds_points", 0.8, "maggr", "R01")
    region = type("RegionStub", (), {"utm_crs": UTM})()
    pred = tmp_path / "pred.geojson"
    output.write_result(solution._to_features([det], {"R01": region}), pred)
    to_3857 = Transformer.from_crs(UTM, "EPSG:3857", always_xy=True)
    x, y = to_3857.transform(HILL_X, HILL_Y)
    gt = tmp_path / "gt.geojson"
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": {"class_name": "finds_points", "markup_type": "maggr"},
    }
    gt.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    assert _score(pred, gt)["final_score"] == pytest.approx(1.0, abs=1e-4)


def test_cli_from_archive_root_without_weights(tmp_path):
    """Как на платформе: из корня распакованного архива python solution.py <вход> <выход>."""
    root = tmp_path / "sol"
    shutil.copytree(
        REPO_DIR / "submission", root, ignore=shutil.ignore_patterns("__pycache__", "weights")
    )
    proc = subprocess.run(
        [sys.executable, "solution.py", str(_make_input(tmp_path)), str(tmp_path / "o")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads((tmp_path / "o" / "result.geojson").read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
