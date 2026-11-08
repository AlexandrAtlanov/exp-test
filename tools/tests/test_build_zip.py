"""Тесты tools/build_zip.py: состав архива, проверки весов и запуск из распакованного архива."""

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import pytest

import build_zip
from koz3 import config

REPO_DIR = Path(__file__).resolve().parents[2]
SUBMISSION = REPO_DIR / "submission"
PT = str(config.YOLO["weights_pt"])
ENGINE = str(config.YOLO["weights_engine"])


def _weights(tmp_path: Path, *names: str) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in names:
        (weights / name).write_bytes(b"weights:" + name.encode())
    return weights


def _build(tmp_path: Path, weights: Path) -> tuple[int, Path]:
    out = tmp_path / "submission.zip"
    code = build_zip.main(["--weights", str(weights), "--out", str(out)])
    return code, out


def test_archive_layout_with_pt_only(tmp_path: Path) -> None:
    code, out = _build(tmp_path, _weights(tmp_path, PT, f"{PT}.json"))
    assert code == 0
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        metadata = json.loads(archive.read("metadata.json"))
    assert {"metadata.json", "solution.py", "koz3/__init__.py", "koz3/config.py"} <= names
    assert {f"weights/{PT}", f"weights/{PT}.json"} <= names
    assert "koz3/detectors/lidar.py" in names
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    assert not [n for n in names if n.startswith(("submission/", "tools/", "training/"))]
    assert metadata == json.loads((SUBMISSION / "metadata.json").read_text(encoding="utf-8"))


def test_missing_pt_is_error(tmp_path: Path) -> None:
    code, out = _build(tmp_path, _weights(tmp_path, ENGINE, f"{ENGINE}.json"))
    assert code == 1
    assert not out.exists()


def test_engine_without_passport_is_error(tmp_path: Path) -> None:
    code, _ = _build(tmp_path, _weights(tmp_path, PT, ENGINE))
    assert code == 1


def test_engine_with_passport_goes_to_weights(tmp_path: Path) -> None:
    code, out = _build(tmp_path, _weights(tmp_path, PT, ENGINE, f"{ENGINE}.json"))
    assert code == 0
    with zipfile.ZipFile(out) as archive:
        assert {f"weights/{ENGINE}", f"weights/{ENGINE}.json"} <= set(archive.namelist())


def test_archive_is_deterministic(tmp_path: Path) -> None:
    weights = _weights(tmp_path, PT)
    hashes = []
    for name in ("a.zip", "b.zip"):
        out = tmp_path / name
        assert build_zip.main(["--weights", str(weights), "--out", str(out)]) == 0
        hashes.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def test_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_zip, "MAX_ZIP_BYTES", 10)
    code, out = _build(tmp_path, _weights(tmp_path, PT))
    assert code == 1
    assert not out.exists()


def test_unpacked_archive_runs_like_platform(tmp_path: Path) -> None:
    """Распаковать архив и запустить python solution.py <вход> <выход> из его корня.

    Веса фиктивные: модель не загрузится, детектор пропустится, но решение обязано
    дойти до конца и записать валидный файл (так же поведёт себя платформа).
    """
    code, out = _build(tmp_path, _weights(tmp_path, PT))
    assert code == 0
    root = tmp_path / "unpacked"
    with zipfile.ZipFile(out) as archive:
        archive.extractall(root)
    region = tmp_path / "input" / "R01"
    region.mkdir(parents=True)
    (region / "UTM.json").write_text(json.dumps({"crs": "EPSG:32637"}), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "solution.py", str(tmp_path / "input"), str(tmp_path / "out")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads((tmp_path / "out" / "result.geojson").read_text(encoding="utf-8"))
    assert result["type"] == "FeatureCollection"
