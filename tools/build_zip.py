"""Сборка архива решения для платформы: submission/ в корне архива + веса в weights/.

    python tools/build_zip.py                          # веса из submission/weights
    python tools/build_zip.py --weights D:/models/v3 --out submission_v3.zip

Архив: metadata.json, solution.py и koz3/ в корне (как в submission/), веса — в
weights/<имя> (там их ищет koz3.config.WEIGHTS_DIR). Проверки до записи:
- metadata.json — формат платформы с ключами image и entry_point (копируется как есть);
- solution.py компилируется;
- есть веса, которых ждёт config: config.YOLO["weights_pt"] при включённом детекторе
  lidar (.pt — запасной вариант, если engine не загрузится);
- engine (config.YOLO["weights_engine"]) необязателен, но рядом с ним обязан лежать
  паспорт <имя>.engine.json (его пишет training/export_engine.py);
- итоговый размер ≤ 2 ГБ (лимит регламента).
Без __pycache__, *.pyc и скрытых файлов. Архив детерминированный: имена отсортированы,
время записей фиксировано — одинаковые входы дают одинаковый sha256. Таблица весов
(имя, размер, sha256) печатается для журнала команды, в архив не пишется.

Коды возврата: 0 — архив собран; 1 — проверка не прошла (архив не создаётся или удаляется).
"""

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger("build_zip")

REPO_DIR = Path(__file__).resolve().parents[1]
MAX_ZIP_BYTES = 2_000_000_000
"""Лимит архива по регламенту — 2 ГБ (берём десятичные гигабайты: строже)."""

FIXED_TIME = (2026, 1, 1, 0, 0, 0)
SKIP_DIRS = {"__pycache__", "weights"}
"""weights/ из submission/ не берём напрямую: веса идут из --weights."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(submission_dir: Path):
    """koz3.config из указанной папки решения (имена весов и включённые детекторы)."""
    sys.path.insert(0, str(submission_dir))
    try:
        from koz3 import config
    finally:
        sys.path.remove(str(submission_dir))
    return config


def code_files(submission_dir: Path) -> dict[str, Path]:
    """Файлы решения: {имя в архиве: путь}; без weights/, __pycache__, .pyc и скрытых."""
    entries: dict[str, Path] = {}
    for path in sorted(submission_dir.rglob("*")):
        rel = path.relative_to(submission_dir)
        if not path.is_file() or path.suffix in (".pyc", ".pyo"):
            continue
        if SKIP_DIRS.intersection(rel.parts) or any(p.startswith(".") for p in rel.parts):
            continue
        entries[rel.as_posix()] = path
    return entries


def weight_files(weights_dir: Path, config) -> tuple[dict[str, Path], list[str]]:
    """Веса для архива и список ошибок проверки."""
    errors: list[str] = []
    files = (
        {f"weights/{p.name}": p for p in sorted(weights_dir.iterdir()) if p.is_file()}
        if weights_dir.is_dir()
        else {}
    )
    files = {name: path for name, path in files.items() if not path.name.startswith(".")}
    if "lidar" in config.ENABLED_DETECTORS:
        pt_name = str(config.YOLO["weights_pt"])
        if f"weights/{pt_name}" not in files:
            errors.append(f"нет {pt_name} в {weights_dir} (нужен детектору lidar)")
        for extra in tuple(config.YOLO.get("extra_weights_pt", ())):
            if f"weights/{extra}" not in files:
                errors.append(f"нет {extra} в {weights_dir} (YOLO extra_weights_pt — ансамбль)")
    engine_name = str(config.YOLO["weights_engine"])
    if f"weights/{engine_name}" in files and f"weights/{engine_name}.json" not in files:
        errors.append(f"есть {engine_name}, но нет паспорта {engine_name}.json")
    empty = [name for name, path in files.items() if path.stat().st_size == 0]
    errors.extend(f"пустой файл весов: {name}" for name in empty)
    return files, errors


def check_metadata(path: Path) -> list[str]:
    """Ошибки metadata.json: нет файла, не JSON, нет image или entry_point."""
    if not path.is_file():
        return [f"нет {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"metadata.json не JSON: {exc}"]
    missing = [key for key in ("image", "entry_point") if not data.get(key)]
    return [f"в metadata.json нет ключа {key}" for key in missing]


def write_zip(out: Path, entries: dict[str, Path]) -> None:
    """Детерминированный zip: сортировка имён, фиксированное время и права."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with archive.open(info, "w") as target, entries[name].open("rb") as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    target.write(chunk)


def build(submission_dir: Path, weights_dir: Path, out: Path) -> int:
    """Проверить и собрать архив; 0 — успех, 1 — ошибка."""
    errors = check_metadata(submission_dir / "metadata.json")
    solution = submission_dir / "solution.py"
    try:
        compile(solution.read_text(encoding="utf-8"), str(solution), "exec")
    except (OSError, SyntaxError) as exc:
        errors.append(f"solution.py: {exc}")
    config = _load_config(submission_dir)
    weights, weight_errors = weight_files(weights_dir, config)
    errors.extend(weight_errors)
    if errors:
        for error in errors:
            logger.error("%s", error)
        return 1

    entries = {**code_files(submission_dir), **weights}
    write_zip(out, entries)
    size = out.stat().st_size
    if size > MAX_ZIP_BYTES:
        out.unlink()
        logger.error("архив %.2f ГБ > лимита 2 ГБ — удалён", size / 1e9)
        return 1

    logger.info("Веса (запишите в журнал, раздел «Модели и веса»):")
    for name, path in weights.items():
        logger.info("  %-32s %12d байт  sha256 %s", name, path.stat().st_size, _sha256(path))
    if not weights:
        logger.warning("Весов нет: решение запишет пустой валидный файл")
    logger.info("Архив: %s — %d файлов, %.1f МБ", out, len(entries), size / 1e6)
    logger.info("sha256 архива: %s", _sha256(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="submission/ + веса → архив для платформы")
    parser.add_argument(
        "--weights", type=Path, help="папка весов (по умолчанию submission/weights)"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_DIR / "submission.zip", help="путь к архиву"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=REPO_DIR / "submission",
        help="папка решения (по умолчанию submission/ репозитория)",
    )
    args = parser.parse_args(argv)
    weights_dir = args.weights if args.weights is not None else args.submission / "weights"
    return build(args.submission, weights_dir, args.out)


if __name__ == "__main__":
    sys.exit(main())
