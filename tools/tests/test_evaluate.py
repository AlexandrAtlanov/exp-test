"""Тесты tools/evaluate.py: обёртка над метрикой организаторов.

Критерии приёмки C11-1/2: pred = gt → скор ≈ 1.0; пустой pred → 0.0;
итог evaluate совпадает с CLI метрики (tools/compute_metrics.py) на тех
же файлах. Синтетика — EPSG:3857, точки finds_points остаются точками.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import evaluate
import pytest

_FINAL_RE = re.compile(r"final_score:\s*([0-9.]+)")


def _write_geojson(path: Path, features: list[dict]) -> Path:
    """Синтетический GeoJSON платформенного формата (EPSG:3857)."""
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": features,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _feature(class_name: str, markup_type: str, geometry: dict) -> dict:
    """Фича с минимальными свойствами платформы."""
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "class_name": class_name,
            "markup_type": markup_type,
            "region_name": "001_Тестовый",
        },
    }


def _poly(class_name: str, markup_type: str, x: float, y: float, side: float) -> dict:
    """Квадратная фича (координаты EPSG:3857)."""
    ring = [[x, y], [x + side, y], [x + side, y + side], [x, y + side], [x, y]]
    return _feature(class_name, markup_type, {"type": "Polygon", "coordinates": [ring]})


def _point(class_name: str, markup_type: str, x: float, y: float) -> dict:
    """Точечная фича (координаты EPSG:3857)."""
    return _feature(class_name, markup_type, {"type": "Point", "coordinates": [x, y]})


def _shift(feature: dict, dx: float) -> dict:
    """Копия полигонной фичи, сдвинутая по x (частичное перекрытие)."""
    ring = [list(point) for point in feature["geometry"]["coordinates"][0]]
    for point in ring:
        point[0] += dx
    shifted = dict(feature)
    shifted["geometry"] = {"type": "Polygon", "coordinates": [ring]}
    return shifted


def _gt_features() -> list[dict]:
    """GT: 3 поверхностных класса полигонами + 2 точки finds_points."""
    return [
        _poly("kurgany_tselye", "li", 2796000.0, 6215000.0, 60.0),
        _poly("kurgany_povrezhdennye", "li", 2797000.0, 6215000.0, 60.0),
        _poly("gorodishcha", "ae", 2798000.0, 6215000.0, 400.0),
        _point("finds_points", "maggr", 2796500.0, 6216500.0),
        _point("finds_points", "terfl", 2796600.0, 6216600.0),
    ]


def _final_score(stdout: str) -> float:
    """Итоговый скор из вывода evaluate.py."""
    match = _FINAL_RE.search(stdout)
    assert match is not None, f"в выводе нет final_score:\n{stdout}"
    return float(match.group(1))


def test_pred_equals_gt_near_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Критерий 1: pred = gt → скор ≈ 1.0 (точки матчатся по расстоянию 0)."""
    gt = _write_geojson(tmp_path / "gt.geojson", _gt_features())
    code = evaluate.main(["--pred", str(gt), "--gt", str(gt)])
    out = capsys.readouterr().out
    assert code == 0
    assert 0.999 < _final_score(out) <= 1.0
    assert "kurgany_tselye" in out


def test_empty_pred_score_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Критерий 1: пустой pred → скор 0.0."""
    gt = _write_geojson(tmp_path / "gt.geojson", _gt_features())
    empty = _write_geojson(tmp_path / "empty.geojson", [])
    code = evaluate.main(["--pred", str(empty), "--gt", str(gt)])
    out = capsys.readouterr().out
    assert code == 0
    assert _final_score(out) == 0.0


def test_same_final_as_organizers_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Критерий 2: итог evaluate == итогу CLI метрики на тех же файлах."""
    gt = _write_geojson(tmp_path / "gt.geojson", _gt_features())
    features = _gt_features()
    pred_features = [
        features[0],
        features[1],
        features[3],
        _shift(features[2], 250.0),
        _poly("fortifikatsii", "or", 2795000.0, 6215000.0, 50.0),
    ]
    pred = _write_geojson(tmp_path / "pred.geojson", pred_features)
    code = evaluate.main(["--pred", str(pred), "--gt", str(gt)])
    ours = _final_score(capsys.readouterr().out)
    assert code == 0

    metric_cli = Path(evaluate.__file__).resolve().parent / "compute_metrics.py"
    proc = subprocess.run(
        [sys.executable, str(metric_cli), "-p", str(pred), "-g", str(gt)],
        capture_output=True,
        text=True,
        check=True,
    )
    theirs = float(proc.stdout.strip().splitlines()[-1])
    assert abs(ours - theirs) < 1e-12


def test_save_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--save пишет полный результат; --compare печатает было / стало / Δ."""
    gt = _write_geojson(tmp_path / "gt.geojson", _gt_features())
    empty = _write_geojson(tmp_path / "empty.geojson", [])
    run = tmp_path / "run.json"

    code = evaluate.main(["--pred", str(gt), "--gt", str(gt), "--save", str(run)])
    assert code == 0
    capsys.readouterr()
    saved = json.loads(run.read_text(encoding="utf-8"))
    assert 0.999 < saved["results"]["final_score"] <= 1.0
    assert saved["meta"]["pred"].endswith("gt.geojson")
    assert "auxiliary" in saved["results"]

    code = evaluate.main(["--pred", str(empty), "--gt", str(gt), "--compare", str(run)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Сравнение" in out
    assert "было" in out and "стало" in out
    assert "→ 0.0000000000" in out
    assert "kurgany_tselye" in out


def test_markdown_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--md: таблицы в Markdown для вставки в PR."""
    gt = _write_geojson(tmp_path / "gt.geojson", _gt_features())
    empty = _write_geojson(tmp_path / "empty.geojson", [])
    code = evaluate.main(["--pred", str(empty), "--gt", str(gt), "--md"])
    out = capsys.readouterr().out
    assert code == 0
    assert "### Классы" in out
    assert "| класс |" in out
    assert "|---" in out