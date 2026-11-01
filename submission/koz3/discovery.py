"""Поиск регионов входной папки и распределение их файлов по markup_type.

Регион — подпапка входной папки (если подпапок нет — сама входная папка).
Файлы региона относятся к markup_type по расширению (.las/.laz) и меткам
в пути относительно папки региона: La -> li, SpOr -> sp_or, Or -> or,
Ae -> ae, MagGr -> maggr, EtGr -> etgr, TerFl -> terfl; готовые
хиллшейды Li исключаются. CRS региона определяется каскадом
UTM.json -> заголовок LAS (VLR/EVLR 2112, OGC WKT) -> CRS первого
растра. Регион без проекционной CRS с EPSG-кодом пропускается
с предупреждением.
"""

import json
import logging
import re
import struct
from pathlib import Path
from typing import BinaryIO

import rasterio
from koz3.datatypes import Region
from rasterio.crs import CRS

logger = logging.getLogger(__name__)

_LAS_SUFFIXES: frozenset[str] = frozenset((".las", ".laz"))
"""Расширения облаков точек (markup_type li)."""

_LAS_SIGNATURE = b"LASF"
_LAS_HEADER_BYTES = 375
_LAS_MIN_HEADER_BYTES = 227
_LAS_EVLR_MIN_HEADER_BYTES = 247
_VLR_HEADER_BYTES = 54
_EVLR_HEADER_BYTES = 60
_WKT_RECORD_ID = 2112
_WKT_MAX_BYTES = 1 << 20

_LABEL_RULES: tuple[tuple[re.Pattern[str], str | None], ...] = (
    (re.compile(r"(?<![a-z])li(?![a-z])"), None),
    (re.compile(r"(?<![a-z])maggr(?![a-z])"), "maggr"),
    (re.compile(r"(?<![a-z])etgr(?![a-z])"), "etgr"),
    (re.compile(r"(?<![a-z])terfl(?![a-z])"), "terfl"),
    (re.compile(r"(?<![a-z])sp[._-]?or(?![a-z])"), "sp_or"),
    (re.compile(r"(?<![a-z])la(?![a-z])"), "li"),
    (re.compile(r"(?<![a-z])ae(?![a-z])"), "ae"),
    (re.compile(r"(?<![a-z])or(?![a-z])"), "or"),
)
"""Правила «метка -> markup_type» в порядке проверки; цель None — исключить.

Метка ищется в нижнем регистре пути файла относительно папки региона,
границы метки — не-буквы: «north» не попадает в «or», «flat» — в «la»,
«outline» — в «li». Длинные метки проверяются раньше коротких, а
расширение .las/.laz — раньше всех правил, поэтому «Li/cloud.las»
остаётся облаком точек li, а «Li/hillshade.tif» исключается.
"""

_RASTER_MARKUP_ORDER: tuple[str, ...] = ("ae", "sp_or", "or", "maggr", "etgr", "terfl")
"""Порядок markup_type при поиске CRS по растрам региона."""


def find_regions(input_dir: Path) -> list[Region]:
    """Регионы входной папки: файлы по markup_type и CRS региона (UTM).

    Регионы без определившейся CRS пропускаются с предупреждением:
    геометрия решения считается в UTM региона в метрах (инвариант 4.8).
    """
    if not input_dir.is_dir():
        logger.error("Входная папка не существует или не папка: %s", input_dir)
        return []
    regions: list[Region] = []
    for region_dir in _region_dirs(input_dir):
        files, unclassified = _collect_files(region_dir)
        utm_crs = _region_crs(region_dir, files)
        if utm_crs is None:
            logger.warning("Регион %s пропущен: CRS не определилась", region_dir.name)
            continue
        logger.info(
            "Регион %s: CRS %s; файлы по типам: %s; без типа: %d",
            region_dir.name,
            utm_crs,
            {key: len(value) for key, value in files.items()},
            unclassified,
        )
        regions.append(
            Region(
                name=region_dir.name,
                path=region_dir,
                utm_crs=utm_crs,
                files=files,
            )
        )
    logger.info("Регионов к обработке: %d (вход: %s)", len(regions), input_dir)
    return regions


def _region_dirs(input_dir: Path) -> list[Path]:
    """Подпапки входной папки без скрытых; если подпапок нет — сама папка."""
    subdirs = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    return subdirs or [input_dir]


def _collect_files(region_dir: Path) -> tuple[dict[str, list[Path]], int]:
    """Файлы региона по markup_type и число файлов без распознанного типа.

    Порядок файлов детерминирован (сортировка путей); скрытые файлы
    и скрытые подпапки (имя начинается с точки) пропускаются.
    """
    files: dict[str, list[Path]] = {}
    unclassified = 0
    for path in sorted(region_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(region_dir)
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        markup_type = _markup_type_of(rel_path)
        if markup_type is None:
            unclassified += 1
            continue
        files.setdefault(markup_type, []).append(path)
    return files, unclassified


def _markup_type_of(rel_path: Path) -> str | None:
    """markup_type по расширению и меткам пути относительно региона.

    Возвращает None для нераспознанных файлов и готовых хиллшейдов Li
    (правило-исключение). Порядок правил — см. ``_LABEL_RULES``.
    """
    if rel_path.suffix.lower() in _LAS_SUFFIXES:
        return "li"
    haystack = rel_path.as_posix().lower()
    for pattern, markup_type in _LABEL_RULES:
        if pattern.search(haystack) is not None:
            return markup_type
    return None


def _region_crs(region_dir: Path, files: dict[str, list[Path]]) -> str | None:
    """CRS региона каскадом: UTM.json -> заголовок LAS -> первый растр."""
    utm_crs = _crs_from_utm_json(region_dir / "UTM.json")
    if utm_crs is not None:
        return utm_crs
    utm_crs = _crs_from_las(files.get("li", []))
    if utm_crs is not None:
        return utm_crs
    return _crs_from_rasters(files)


def _crs_from_utm_json(path: Path) -> str | None:
    """CRS из UTM.json региона: ключи crs/CRS, затем utm/UTM (зона)."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("UTM.json не читается: %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("UTM.json: ожидался JSON-объект: %s", path)
        return None
    crs_value = data.get("crs") or data.get("CRS")
    if crs_value is not None:
        utm_crs = _sanitize_crs(_normalize_crs(str(crs_value)))
        if utm_crs is not None:
            return utm_crs
    utm_value = data.get("utm") or data.get("UTM")
    if utm_value is not None:
        utm_crs = _sanitize_crs(_utm_zone_to_epsg(str(utm_value)))
        if utm_crs is not None:
            return utm_crs
    logger.warning("UTM.json без распознаваемого crs/utm: %s", path)
    return None


def _normalize_crs(text: str) -> str:
    """Приводит «urn:ogc:def:crs:epsg::XXXX» и «EPSGXXXX» к «EPSG:XXXX».

    Остальные строки возвращает без изменений; пустая строка остаётся
    пустой и отбрасывается валидацией ``_sanitize_crs``.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith("urn:ogc:def:crs:epsg::"):
        code = stripped.split("::")[-1].strip()
        return f"EPSG:{code}" if code.isdigit() else stripped
    if lowered.startswith("epsg"):
        code = stripped[4:].lstrip(":").strip()
        if code.isdigit():
            return f"EPSG:{code}"
    return stripped


def _utm_zone_to_epsg(text: str) -> str | None:
    """Зона UTM («37», «37N», «N37») -> «EPSG:32637» либо «EPSG:32737».

    Полушарие — по букве «S» в значении; по умолчанию северное.
    """
    cleaned = text.strip().upper()
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if not digits:
        return None
    zone = int(digits)
    if not 1 <= zone <= 60:
        logger.warning("Некорректная зона UTM: %r", text)
        return None
    letter = next((ch for ch in reversed(cleaned) if ch.isalpha()), "N")
    base = 327 if letter == "S" else 326
    return f"EPSG:{base}{zone:02d}"


def _sanitize_crs(crs_text: str) -> str | None:
    """Валидирует строку CRS и возвращает «EPSG:XXXX», либо None.

    Проходит только проекционная система с EPSG-кодом: геометрия решения
    считается в метрах (инвариант 4.8), географические CRS не подходят.
    """
    if not crs_text:
        return None
    try:
        crs = CRS.from_user_input(crs_text)
    except ValueError as exc:
        # rasterio.crs.CRSError — подкласс ValueError
        logger.debug("CRS не распознана %r: %s", crs_text, exc)
        return None
    if not crs.is_projected:
        return None
    epsg = crs.to_epsg()
    if epsg is None:
        return None
    return f"EPSG:{epsg}"


def _crs_from_las(las_files: list[Path]) -> str | None:
    """CRS из заголовка первого LAS/LAZ с VLR/EVLR 2112 (OGC WKT)."""
    for path in las_files:
        utm_crs = _las_wkt_crs(path)
        if utm_crs is not None:
            return utm_crs
    return None


def _las_wkt_crs(path: Path) -> str | None:
    """CRS из заголовка LAS/LAZ: VLR, затем EVLR записи 2112 (OGC WKT).

    Ошибки чтения не поднимаются наружу — возвращается None. EVLR
    читается только для LAS 1.4: в версиях 1.0-1.3 байты 235-246
    принадлежат другим данным заголовка.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(_LAS_HEADER_BYTES)
            if len(head) < _LAS_MIN_HEADER_BYTES or head[:4] != _LAS_SIGNATURE:
                return None
            version_minor = head[25]
            header_size = struct.unpack_from("<H", head, 94)[0]
            if header_size < _LAS_MIN_HEADER_BYTES:
                return None
            vlr_count = struct.unpack_from("<I", head, 100)[0]
            handle.seek(header_size)
            utm_crs = _wkt_crs_from_vlrs(handle, vlr_count)
            if utm_crs is not None:
                return utm_crs
            if version_minor < 4 or len(head) < _LAS_EVLR_MIN_HEADER_BYTES:
                return None
            evlr_offset = struct.unpack_from("<Q", head, 235)[0]
            evlr_count = struct.unpack_from("<I", head, 243)[0]
            if not evlr_offset or not evlr_count:
                return None
            handle.seek(evlr_offset)
            return _wkt_crs_from_evlrs(handle, evlr_count)
    except OSError as exc:
        logger.warning("Не читается LAS/LAZ: %s: %s", path, exc)
        return None


def _wkt_crs_from_vlrs(handle: BinaryIO, count: int) -> str | None:
    """Ищет VLR 2112 с OGC WKT среди первых ``count`` VLR потока."""
    for _ in range(count):
        record = handle.read(_VLR_HEADER_BYTES)
        if len(record) < _VLR_HEADER_BYTES:
            break
        _, _, record_id, record_length, _ = struct.unpack_from("<H16sHH32s", record)
        if record_id == _WKT_RECORD_ID:
            payload = handle.read(record_length)
            if len(payload) < record_length:
                break
            utm_crs = _wkt_to_crs(payload)
            if utm_crs is not None:
                return utm_crs
            continue
        handle.seek(record_length, 1)
    return None


def _wkt_crs_from_evlrs(handle: BinaryIO, count: int) -> str | None:
    """Ищет EVLR 2112 с OGC WKT среди первых ``count`` EVLR потока."""
    for _ in range(count):
        record = handle.read(_EVLR_HEADER_BYTES)
        if len(record) < _EVLR_HEADER_BYTES:
            break
        _, _, record_id, record_length, _ = struct.unpack_from("<H16sHQ32s", record)
        if record_id == _WKT_RECORD_ID:
            payload = handle.read(min(record_length, _WKT_MAX_BYTES))
            utm_crs = _wkt_to_crs(payload)
            if utm_crs is not None:
                return utm_crs
            continue
        handle.seek(record_length, 1)
    return None


def _wkt_to_crs(payload: bytes) -> str | None:
    """Байты записи WKT -> «EPSG:XXXX», либо None."""
    text = payload.decode("utf-8", "ignore").strip("\x00").strip()
    if not text:
        return None
    return _sanitize_crs(text)


def _crs_from_rasters(files: dict[str, list[Path]]) -> str | None:
    """CRS первого открываемого растра региона с проекционным EPSG."""
    for markup_type in _RASTER_MARKUP_ORDER:
        for path in files.get(markup_type, []):
            utm_crs = _raster_crs(path)
            if utm_crs is not None:
                return utm_crs
    return None


def _raster_crs(path: Path) -> str | None:
    """CRS растра в виде «EPSG:XXXX», либо None, если не открылся."""
    # Широкий перехват намеренен: rasterio/GDAL бросает широкий спектр типов
    # на повреждённых файлах, а ошибка одного файла не должна ронять прогон
    try:
        with rasterio.open(path) as dataset:
            src_crs = dataset.crs
    except Exception as exc:  # noqa: BLE001
        logger.debug("Растр не открылся: %s: %s", path.name, exc)
        return None
    if src_crs is None:
        return None
    return _sanitize_crs(src_crs.to_string())