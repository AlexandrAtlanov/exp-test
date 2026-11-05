"""Сборка детерминированного submission-архива для отправки.

Кладёт в zip всё содержимое папки submission/ (включая solution.py — в корень
архива), пакет koz3/, engine (TensorRT) и его паспорт, а также manifest.json
с размером и sha256 каждой записи. Записи сортируются по имени, время файлов
фиксировано, поэтому повторная сборка на тех же входах даёт побитово тот же
архив — удобно для сравнения и отладки.

Если рядом с engine нашёлся паспорт (passport.json или <имя>.json), сборка
сверяет sha256 engine с хэшем из паспорта и останавливается с кодом 1 при
несовпадении — так в архив не попадает чужой engine.

Коды возврата: 0 — успех; 1 — нет solution.py или engine, синтаксическая
ошибка, паспорт не сошёлся, архив не читается (сообщения — в stderr).
Режим --check проверяет готовый архив по manifest.json без пересборки.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

FIXED_TIME = (2020, 2, 2, 0, 0, 0)
ENGINE_GLOBS = (
    "artifacts/**/*.engine",
    "training/**/*.engine",
    "runs/**/*.engine",
    "weights/**/*.engine",
)
SKIP_DIRS = {"__pycache__", ".git"}
SKIP_SUFFIXES = (".pyc", ".pyo")
CHUNK = 1 << 20


def _fail(message: str) -> int:
    """Пишет сообщение в stderr и возвращает код 1."""
    print(f"build_zip: {message}", file=sys.stderr)
    return 1


def _sha256_file(path: Path) -> str:
    """sha256 от файла, читается кусками по 1 МБ."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_engines(root: Path) -> list[Path]:
    """Отсортированные .engine из стандартных мест проекта."""
    engines: set[Path] = set()
    for pattern in ENGINE_GLOBS:
        engines.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(engines)


def resolve_passport(engine: Path, explicit: Path | None) -> Path | None:
    """Паспорт engine: явный путь, passport.json или <имя>.json рядом с engine."""
    if explicit is not None:
        return explicit
    for candidate in (engine.parent / "passport.json", engine.with_suffix(".json")):
        if candidate.is_file():
            return candidate
    return None


def passport_matches_engine(passport: dict, engine_hash: str) -> bool:
    """Есть ли sha256 engine среди строковых значений паспорта (рекурсивно)."""
    stack = [passport]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str) and value.lower() == engine_hash:
            return True
    return False


def _iter_files(base: Path) -> list[Path]:
    """Файлы внутри base без __pycache__/.git и .pyc/.pyo, отсортированы."""
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and not SKIP_DIRS.intersection(path.parts)
        and path.suffix not in SKIP_SUFFIXES
    )


def _extra_name(root: Path, extra: Path, path: Path) -> str:
    """Имя записи --extra: относительно root; вне root — от имени extra."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        if path == extra:
            return extra.name
        return f"{extra.name}/{path.relative_to(extra).as_posix()}"


def _zip_info(name: str) -> zipfile.ZipInfo:
    """ZipInfo с фиксированным временем, правами и системой — для детерминизма."""
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_entry(archive: zipfile.ZipFile, name: str, path: Path) -> None:
    """Кладёт файл в архив потоково с фиксированным временем."""
    with archive.open(_zip_info(name), "w") as target, path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            target.write(chunk)


def write_zip(out: Path, entries: dict[str, Path], engine: Path, engine_hash: str) -> dict:
    """Пишет детерминированный архив с manifest.json, возвращает манифест."""
    described = []
    for name in sorted(entries):
        path = entries[name]
        digest = engine_hash if path == engine else _sha256_file(path)
        described.append({"name": name, "size": path.stat().st_size, "sha256": digest})
    manifest = {
        "tool": "build_zip",
        "deterministic": True,
        "engine": {"name": engine.name, "size": engine.stat().st_size, "sha256": engine_hash},
        "entries": described,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            _write_entry(archive, name, entries[name])
        payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        archive.writestr(_zip_info("manifest.json"), payload)
    return manifest


def check_zip(out: Path) -> int:
    """Проверяет готовый архив по manifest.json, возвращает 0 или 1."""
    if not out.is_file():
        return _fail(f"архив не найден: {out}")
    try:
        with zipfile.ZipFile(out) as archive:
            if archive.testzip() is not None:
                return _fail("архив повреждён (CRC)")
            if "manifest.json" not in archive.namelist():
                return _fail("в архиве нет manifest.json")
            manifest = json.loads(archive.read("manifest.json"))
            actual = {
                info.filename: (
                    info.file_size,
                    hashlib.sha256(archive.read(info.filename)).hexdigest(),
                )
                for info in archive.infolist()
                if info.filename != "manifest.json"
            }
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return _fail(f"архив не читается: {exc}")

    described = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(described, list):
        return _fail("в манифесте нет списка entries")
    expected: dict[str, tuple[int, str]] = {}
    for entry in described:
        if not isinstance(entry, dict) or {"name", "size", "sha256"} - entry.keys():
            return _fail(f"некорректная запись манифеста: {entry!r}")
        expected[entry["name"]] = (entry["size"], entry["sha256"])

    problems = [f"нет записи: {name}" for name in sorted(set(expected) - set(actual))]
    problems += [f"лишняя запись: {name}" for name in sorted(set(actual) - set(expected))]
    problems += [
        f"не совпал размер или sha256: {name}"
        for name in sorted(set(expected) & set(actual))
        if expected[name] != actual[name]
    ]
    if problems:
        return _fail("проверка не прошла:\n  " + "\n  ".join(problems))
    print(f"OK: {out.name} — {len(actual)} записей соответствуют манифесту")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа: собирает или проверяет архив, возвращает код 0 или 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="корень проекта (по умолчанию — родитель tools/)"
    )
    parser.add_argument(
        "--solution-dir", type=Path, help="папка решения (по умолчанию submission/)"
    )
    parser.add_argument("--package", type=Path, help="пакет рантайма (по умолчанию koz3/)")
    parser.add_argument(
        "--engine", type=Path, help="путь к .engine (по умолчанию — поиск в стандартных местах)"
    )
    parser.add_argument(
        "--passport", type=Path, help="паспорт engine (по умолчанию passport.json рядом)"
    )
    parser.add_argument(
        "--extra", action="append", type=Path, help="доп. файл или папка в архив (можно много раз)"
    )
    parser.add_argument(
        "--out", type=Path, help="путь к архиву (по умолчанию submission.zip в корне)"
    )
    parser.add_argument(
        "--no-passport-check", action="store_true", help="не сверять паспорт с engine"
    )
    parser.add_argument(
        "--check", action="store_true", help="проверить готовый архив по манифесту"
    )
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else Path(__file__).resolve().parents[1]
    out = args.out if args.out is not None else root / "submission.zip"
    if args.check:
        return check_zip(out)

    solution_dir = args.solution_dir if args.solution_dir is not None else root / "submission"
    package_dir = args.package if args.package is not None else root / "koz3"
    if not solution_dir.is_dir():
        return _fail(f"нет папки решения: {solution_dir}")
    solution = solution_dir / "solution.py"
    if not solution.is_file():
        return _fail(f"нет solution.py в {solution_dir}")
    try:
        compile(solution.read_text(encoding="utf-8"), str(solution), "exec")
    except (OSError, SyntaxError) as exc:
        return _fail(f"solution.py не компилируется: {exc}")
    if not (package_dir / "__init__.py").is_file():
        return _fail(f"нет пакета {package_dir} (ожидается __init__.py)")

    if args.engine is not None:
        engine = args.engine
        if not engine.is_file():
            return _fail(f"engine не найден: {engine}")
    else:
        engines = find_engines(root)
        if not engines:
            return _fail("не найден .engine — передайте путь через --engine")
        if len(engines) > 1:
            listing = ", ".join(str(path) for path in engines)
            return _fail(f"несколько .engine: {listing} — передайте один в --engine")
        engine = engines[0]
    if engine.stat().st_size == 0:
        return _fail(f"engine пустой: {engine}")

    engine_hash = _sha256_file(engine)
    passport = resolve_passport(engine, args.passport)
    if passport is not None:
        if not passport.is_file():
            return _fail(f"паспорт не найден: {passport}")
        if not args.no_passport_check:
            try:
                passport_data = json.loads(passport.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return _fail(f"паспорт не читается: {exc}")
            if not passport_matches_engine(passport_data, engine_hash):
                return _fail(
                    f"паспорт {passport.name} не совпадает с engine "
                    f"(sha256 {engine_hash})"
                )

    entries: dict[str, Path] = {}

    def add(name: str, path: Path) -> bool:
        """Регистрирует запись; False, если имя уже занято."""
        if name in entries:
            return False
        entries[name] = path
        return True

    for path in _iter_files(solution_dir):
        if not add(path.relative_to(solution_dir).as_posix(), path):
            return _fail(f"дубликат имени в архиве: {path.relative_to(solution_dir).as_posix()}")
    for path in _iter_files(package_dir):
        name = f"{package_dir.name}/{path.relative_to(package_dir).as_posix()}"
        if not add(name, path):
            return _fail(f"дубликат имени в архиве: {name}")
    if not add(engine.name, engine):
        return _fail(f"дубликат имени в архиве: {engine.name}")
    if passport is not None and not add(passport.name, passport):
        return _fail(f"дубликат имени в архиве: {passport.name}")
    for extra in args.extra or []:
        if extra.is_dir():
            for path in _iter_files(extra):
                if not add(_extra_name(root, extra, path), path):
                    return _fail(f"дубликат имени в архиве: {_extra_name(root, extra, path)}")
        elif extra.is_file():
            if not add(_extra_name(root, extra, extra), extra):
                return _fail(f"дубликат имени в архиве: {_extra_name(root, extra, extra)}")
        else:
            return _fail(f"--extra не найден: {extra}")

    manifest = write_zip(out, entries, engine, engine_hash)
    sidecar = out.parent / f"{out.stem}_manifest.json"
    sidecar.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    code = check_zip(out)
    if code != 0:
        return code
    print(f"Архив: {out} ({out.stat().st_size} байт, записей: {len(entries)} + manifest.json)")
    print(f"sha256 архива: {_sha256_file(out)}")
    print(f"engine: {engine.name} ({manifest['engine']['sha256'][:16]}…)")
    print(f"Манифест рядом с архивом: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
