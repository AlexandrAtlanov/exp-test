"""training/audit_data.py на синтетике: сдвиг разметки находится, битые подписи ловятся."""

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import Point
from test_lidar_dem import write_synthetic_las

import audit_data
from koz3.detectors import lidar_dem

HILL = Point(500020.0, 6000015.0)
"""Центр холма в write_synthetic_las."""


def _relief(tmp_path: Path):
    las = write_synthetic_las(tmp_path / "s_La.las", n=60000)
    dem, transform, _ = lidar_dem.build_dem_from_las(las, res=0.1)
    relief, step = audit_data.local_relief(dem, abs(transform.a), sigma_m=6.0)
    return relief, transform * transform.scale(step / abs(transform.a))


def test_aligned_mound_has_positive_contrast_and_no_shift(tmp_path: Path) -> None:
    relief, transform = _relief(tmp_path)
    result = audit_data.mound_alignment(relief, transform, [HILL.buffer(4.0)], max_shift_m=6.0)
    assert result is not None
    assert result["contrast_m"] > 0
    assert np.hypot(*result["best_shift_m"]) <= 1.0
    assert not result["shifted"]


def test_shifted_annotation_is_detected(tmp_path: Path) -> None:
    relief, transform = _relief(tmp_path)
    moved = translate(HILL.buffer(4.0), xoff=4.0, yoff=-3.0)  # разметка сдвинута на 5 м
    result = audit_data.mound_alignment(relief, transform, [moved], max_shift_m=6.0)
    assert result is not None
    east, north = result["best_shift_m"]
    assert abs(east + 4.0) <= 1.0 and abs(north - 3.0) <= 1.0  # обратный сдвиг найден
    assert result["shifted"]


def test_dataset_audit_finds_broken_labels(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True)
        (ds / "labels" / split).mkdir(parents=True)
    (ds / "dataset_meta.json").write_text(json.dumps({"tile_px": 64}), encoding="utf-8")
    image = np.full((64, 64, 3), 120, np.uint8)
    image[:, 32:] = 0  # правая половина — «нет данных»
    Image.fromarray(image).save(ds / "images" / "train" / "a.png")
    good = "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4"
    over_nodata = "1 0.6 0.1 0.9 0.1 0.9 0.4 0.6 0.4"
    lines = [good, over_nodata, "9 0.1 0.1 0.2 0.1 0.2 0.2", "0 0.1 0.1 1.5 0.1 0.2 0.2", "x y"]
    (ds / "labels" / "train" / "a.txt").write_text("\n".join(lines), encoding="utf-8")
    Image.fromarray(image).save(ds / "images" / "train" / "orphan.png")
    (ds / "images" / "val" / "broken.png").write_bytes(b"not a png")
    (ds / "labels" / "val" / "broken.txt").write_text("", encoding="utf-8")

    report = audit_data.audit_dataset(ds, tmp_path / "out", samples=5)
    problems = report["train"]["problems"]
    assert problems["объект над «нет данных» (> 50 %)"] == 1
    assert problems["id класса вне диапазона"] == 1
    assert problems["координаты вне [0, 1]"] == 1
    assert problems["строка не разбирается"] == 1
    assert problems["картинка без подписи"] == 1
    assert report["val"]["problems"]["битый png"] == 1
    assert (tmp_path / "out" / "samples" / "a.png").is_file()
    text = audit_data.render_markdown([], report)
    assert "битый png" in text


def _region(root: Path, shift_m: float) -> Path:
    """Регион трейна «как у организаторов»: LAS с холмом и разметка кургана (со сдвигом)."""
    region = root / "R01"
    las_dir = region / "01_R01_La_точки"
    las_dir.mkdir(parents=True)
    write_synthetic_las(las_dir / "R01_La.las", n=60000)
    (region / "UTM.json").write_text(json.dumps({"crs": "EPSG:32637"}), encoding="utf-8")
    markup = region / "06_R01_разметка"
    markup.mkdir()
    circle = translate(HILL.buffer(4.0), xoff=shift_m)
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32637"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Курганы повреждённые"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[list(xy) for xy in circle.exterior.coords]],
                },
            }
        ],
    }
    path = markup / "R01_Li_Курганы_целые.geojson"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return root


def test_cli_region_report(tmp_path: Path) -> None:
    """CLI на регионе: LAS читается, класс файла ≠ свойства пойман, картинка и отчёт есть."""
    data = _region(tmp_path / "train", shift_m=0.0)
    out = tmp_path / "audit"
    args = ["--data", str(data), "--cache", str(tmp_path / "cache"), "--out", str(out)]
    assert audit_data.main(args + ["--max-shift", "5"]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    [region] = report["regions"]
    assert region["las"][0]["points"] == 60000
    [ann] = region["annotations"]
    assert ann["loaded"] == 1 and ann["outside_las"] == 0
    assert ann["file_vs_property"]  # файл «целые», свойство name — «повреждённый»
    [align] = region["alignment"]
    assert align["contrast_m"] > 0 and not align["shifted"]
    assert list((out / "regions").glob("*.png"))
    assert "Совпадение курганов с насыпями" in (out / "report.md").read_text(encoding="utf-8")


def test_weak_relief_does_not_trigger_shift() -> None:
    """Низкие насыпи в шуме: выигрыш сдвига < 5 см — разметка не сдвигается (13.11)."""
    rng = np.random.default_rng(3)
    relief = rng.normal(0.0, 0.01, size=(200, 200))
    transform = from_origin(0.0, 100.0, 0.5, 0.5)
    polys = [Point(x, y).buffer(10.0) for x, y in ((25, 25), (75, 25), (25, 75), (75, 75))]
    result = audit_data.mound_alignment(relief, transform, polys, max_shift_m=8.0)
    assert result is not None
    assert result["gain_m"] < 0.05
    assert not result["shifted"]
