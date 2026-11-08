"""Интеграционные тесты точки входа solution.py (критерии приёмки 1-3).

solution.main() вызывается в процессе с детекторами-заглушками, навешанными
в sys.modules как koz3.detectors.<имя> (репозиторий реальных детекторов
пока пуст), плюс один настоящий запуск файла через subprocess — так же,
как это делает платформа.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import _bootstrap  # noqa: F401
from shapely.geometry import box

import solution
import validate_output
from koz3 import config
from koz3.datatypes import CLASSES, Detection

ROOT = Path(__file__).resolve().parents[2]
SOLUTION_FILE = ROOT / "submission" / "solution.py"
UTM = "EPSG:32637"


def _make_input(tmp_path: Path, name: str = "region_0001") -> Path:
    """Входная папка с одним регионом (только UTM.json, как в синтетике)."""
    region_dir = tmp_path / "input" / name
    region_dir.mkdir(parents=True)
    (region_dir / "UTM.json").write_text(json.dumps({"crs": UTM}), encoding="utf-8")
    return tmp_path / "input"


def _box_detections():
    """Две непересекающиеся детекции валидных классов, площадь выше MIN_AREA_M2."""
    out = []
    x = 500000.0
    for cls, confidence in zip(sorted(CLASSES)[:2], (0.9, 0.8)):
        min_area = float(config.MIN_AREA_M2[cls])
        side = max(10.0, math.sqrt(max(min_area, 0.0)) * 2.0)
        out.append((box(x, 4700000.0, x + side, 4700000.0 + side), cls, confidence))
        x += side * 2.0
    return out


def _stub_detector(
    name,
    priority=10,
    detections=(),
    error=None,
    estimate=0.0,
    available=True,
    on_run=None,
):
    """Модуль-заглушка koz3.detectors.<name> с контрактом раздела 5."""
    module = types.ModuleType(f"koz3.detectors.{name}")
    module.PRIORITY = priority
    calls = {"is_available": 0, "estimate_seconds": 0, "run": 0}

    def is_available(region):
        calls["is_available"] += 1
        return available

    def estimate_seconds(region):
        calls["estimate_seconds"] += 1
        return estimate

    def run(region, budget, cfg):
        calls["run"] += 1
        if error is not None:
            raise error
        if on_run is not None:
            on_run(calls)
        return [
            Detection(
                geom=geom,
                class_name=class_name,
                confidence=confidence,
                markup_type="li",
                region_name=region.name,
            )
            for geom, class_name, confidence in detections
        ]

    module.is_available = is_available
    module.estimate_seconds = estimate_seconds
    module.run = run
    return module, calls


def _install_detectors(monkeypatch, *stub_pairs):
    """Навесить заглушки в sys.modules и включить их в ENABLED_DETECTORS."""
    package = types.ModuleType("koz3.detectors")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "koz3.detectors", package)
    names = []
    for module, _calls in stub_pairs:
        monkeypatch.setitem(sys.modules, module.__name__, module)
        names.append(module.__name__.rsplit(".", 1)[-1])
    monkeypatch.setattr(config, "ENABLED_DETECTORS", tuple(names))


def test_cli_requires_exactly_two_args():
    """Неверный CLI (не 2 аргумента) → код возврата 2."""
    assert solution.main([]) == 2
    assert solution.main(["only-input"]) == 2
    assert solution.main(["a", "b", "c"]) == 2


def test_empty_input_valid_empty_collection(tmp_path):
    """Критерий 1: пустая входная папка → валидный пустой FC, код 0."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    out_file = tmp_path / "out" / "result.geojson"

    assert solution.main([str(input_dir), str(out_file)]) == 0

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert data["features"] == []
    assert validate_output.validate(out_file)["defects"] == []


def test_failing_detector_traceback_others_work(tmp_path, monkeypatch, caplog):
    """Критерий 2: упавший детектор — трейсбек в логе, остальные работают, код 0."""
    input_dir = _make_input(tmp_path)
    out_file = tmp_path / "result.geojson"
    good, good_calls = _stub_detector("stub_good", priority=1, detections=_box_detections())
    boom, boom_calls = _stub_detector(
        "stub_boom", priority=2, error=RuntimeError("заглушка сломалась намеренно")
    )
    _install_detectors(monkeypatch, (good, good_calls), (boom, boom_calls))

    assert solution.main([str(input_dir), str(out_file)]) == 0

    assert good_calls["run"] == 1
    assert boom_calls["run"] == 1
    assert "Traceback (most recent call last)" in caplog.text
    assert "заглушка сломалась намеренно" in caplog.text

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 2
    assert validate_output.validate(out_file)["defects"] == []


def test_budget_skip_detectors_still_writes(tmp_path, monkeypatch, caplog):
    """Критерий 3: TIME_LIMIT_S=1 — детекторы пропущены по бюджету, файл записан."""
    input_dir = _make_input(tmp_path)
    out_file = tmp_path / "result.geojson"
    stub, calls = _stub_detector("stub_hungry", priority=1, estimate=600.0)
    _install_detectors(monkeypatch, (stub, calls))
    monkeypatch.setattr(config, "TIME_LIMIT_S", 1)

    assert solution.main([str(input_dir), str(out_file)]) == 0

    assert calls["is_available"] == 1
    assert calls["run"] == 0  # пропущен по бюджету, run не вызывался
    assert "не влезает в бюджет" in caplog.text

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert data["features"] == []


def test_output_directory_variant(tmp_path, monkeypatch):
    """Выход-каталог: результат появляется в <каталог>/result.geojson."""
    input_dir = _make_input(tmp_path)
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    stub, calls = _stub_detector("stub_dir", priority=1, detections=_box_detections())
    _install_detectors(monkeypatch, (stub, calls))

    assert solution.main([str(input_dir), str(out_dir)]) == 0

    result = out_dir / "result.geojson"
    assert result.is_file()
    assert len(json.loads(result.read_text(encoding="utf-8"))["features"]) == 2


def test_checkpoint_between_pairs_and_priority(tmp_path, monkeypatch):
    """Чекпоинт после каждой пары; PRIORITY задаёт порядок детекторов."""
    input_dir = _make_input(tmp_path)
    out_file = tmp_path / "result.geojson"
    seen = {}

    def on_run_second(_calls):
        # ко времени запуска второго детектора файл уже несёт результат
        # первого (промежуточная запись после каждой пары)
        try:
            seen["features"] = len(json.loads(out_file.read_text(encoding="utf-8"))["features"])
        except (OSError, ValueError):
            seen["features"] = None

    first, first_calls = _stub_detector("stub_a", priority=1, detections=_box_detections()[:1])
    second, second_calls = _stub_detector(
        "stub_b", priority=2, detections=_box_detections()[1:], on_run=on_run_second
    )
    _install_detectors(monkeypatch, (first, first_calls), (second, second_calls))

    assert solution.main([str(input_dir), str(out_file)]) == 0

    assert seen["features"] == 1  # чекпоинт первого детектора уже на диске
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data["features"]) == 2


def test_subprocess_real_entry_point(tmp_path):
    """Настоящий запуск файлом: python solution.py <вход> <выход> → код 0."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    out_file = tmp_path / "result.geojson"
    proc = subprocess.run(
        [sys.executable, str(SOLUTION_FILE), str(input_dir), str(out_file)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection"
    assert data["features"] == []


def test_subprocess_wrong_args_exit_code(tmp_path):
    """subprocess с одним аргументом → код возврата 2."""
    proc = subprocess.run(
        [sys.executable, str(SOLUTION_FILE), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2
