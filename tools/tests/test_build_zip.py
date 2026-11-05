"""Тесты tools/build_zip.py: детерминированная сборка submission-архива."""

import hashlib
import json
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
import build_zip

ENGINE_PAYLOAD = b"engine-bytes-0123456789"


def _scaffold(root: Path) -> None:
    """Минимальный проект: submission/solution.py, koz3/, engine и паспорт."""
    (root / "submission").mkdir(parents=True)
    (root / "submission" / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "koz3").mkdir()
    (root / "koz3" / "__init__.py").write_text("", encoding="utf-8")
    (root / "koz3" / "runtime.py").write_text("X = 1\n", encoding="utf-8")
    engine = root / "artifacts" / "best.engine"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(ENGINE_PAYLOAD)
    digest = hashlib.sha256(ENGINE_PAYLOAD).hexdigest()
    (engine.parent / "passport.json").write_text(
        json.dumps({"engine_sha256": digest, "ultralytics": "8.3.0"}), encoding="utf-8"
    )


def test_build_zip_full_run(tmp_path: Path) -> None:
    """Полная сборка: записи на месте, манифест совпадает, паспорт проверен."""
    _scaffold(tmp_path)
    out = tmp_path / "submission.zip"
    assert build_zip.main(["--root", str(tmp_path), "--out", str(out)]) == 0

    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    assert "solution.py" in names
    assert "koz3/__init__.py" in names
    assert "koz3/runtime.py" in names
    assert "best.engine" in names
    assert "passport.json" in names
    assert "manifest.json" in names
    assert not any("__pycache__" in name for name in names)

    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"]["sha256"] == hashlib.sha256(ENGINE_PAYLOAD).hexdigest()
    assert {entry["name"] for entry in manifest["entries"]} == set(names) - {"manifest.json"}


def test_build_zip_deterministic(tmp_path: Path) -> None:
    """Две сборки на одних входах — побитово одинаковый архив."""
    _scaffold(tmp_path)
    first = tmp_path / "one.zip"
    second = tmp_path / "two.zip"
    assert build_zip.main(["--root", str(tmp_path), "--out", str(first)]) == 0
    assert build_zip.main(["--root", str(tmp_path), "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_build_zip_missing_solution_fails(tmp_path: Path) -> None:
    """Нет solution.py — код 1."""
    (tmp_path / "submission").mkdir(parents=True)
    code = build_zip.main(["--root", str(tmp_path), "--out", str(tmp_path / "s.zip")])
    assert code == 1


def test_build_zip_broken_solution_fails(tmp_path: Path) -> None:
    """solution.py с синтаксической ошибкой — код 1."""
    _scaffold(tmp_path)
    (tmp_path / "submission" / "solution.py").write_text("def broken(\n", encoding="utf-8")
    code = build_zip.main(["--root", str(tmp_path), "--out", str(tmp_path / "s.zip")])
    assert code == 1


def test_build_zip_engine_not_found_fails(tmp_path: Path) -> None:
    """Нет .engine — код 1 с подсказкой."""
    (tmp_path / "submission").mkdir(parents=True)
    (tmp_path / "submission" / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "koz3").mkdir()
    (tmp_path / "koz3" / "__init__.py").write_text("", encoding="utf-8")
    code = build_zip.main(["--root", str(tmp_path), "--out", str(tmp_path / "s.zip")])
    assert code == 1


def test_build_zip_engine_ambiguous_fails(tmp_path: Path) -> None:
    """Два .engine — код 1 со списком."""
    _scaffold(tmp_path)
    (tmp_path / "artifacts" / "second.engine").write_bytes(b"another")
    code = build_zip.main(["--root", str(tmp_path), "--out", str(tmp_path / "s.zip")])
    assert code == 1


def test_build_zip_passport_mismatch_fails(tmp_path: Path) -> None:
    """Паспорт с чужим хэшем — код 1, архив не собирается."""
    _scaffold(tmp_path)
    (tmp_path / "artifacts" / "passport.json").write_text(
        json.dumps({"engine_sha256": "0" * 64}), encoding="utf-8"
    )
    out = tmp_path / "s.zip"
    assert build_zip.main(["--root", str(tmp_path), "--out", str(out)]) == 1
    assert not out.exists()


def test_build_zip_no_passport_check(tmp_path: Path) -> None:
    """--no-passport-check пропускает сверку, паспорт всё равно в архиве."""
    _scaffold(tmp_path)
    (tmp_path / "artifacts" / "passport.json").write_text(
        json.dumps({"engine_sha256": "0" * 64}), encoding="utf-8"
    )
    code = build_zip.main(
        ["--root", str(tmp_path), "--out", str(tmp_path / "s.zip"), "--no-passport-check"]
    )
    assert code == 0
    with zipfile.ZipFile(tmp_path / "s.zip") as archive:
        assert "passport.json" in archive.namelist()


def test_build_zip_check_mode(tmp_path: Path) -> None:
    """--check: честный архив — 0; с лишней записью — 1."""
    _scaffold(tmp_path)
    out = tmp_path / "s.zip"
    assert build_zip.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    assert build_zip.main(["--check", "--out", str(out)]) == 0

    with zipfile.ZipFile(out, "a") as archive:
        archive.writestr("extra.txt", "surprise")

    assert build_zip.main(["--check", "--out", str(out)]) == 1


def test_build_zip_extra_files(tmp_path: Path) -> None:
    """--extra добавляет файлы и папки в архив."""
    _scaffold(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("заметки", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "readme.md").write_text("док", encoding="utf-8")
    out = tmp_path / "s.zip"
    code = build_zip.main(
        ["--root", str(tmp_path), "--out", str(out), "--extra", str(notes), "--extra", str(docs)]
    )
    assert code == 0
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    assert "notes.txt" in names
    assert "docs/readme.md" in names
