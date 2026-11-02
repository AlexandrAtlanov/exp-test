"""Тесты koz3.discovery: регионы, markup_type по меткам, каскад CRS.

Все данные синтетические (пустые заглушки и фейковые заголовки LAS);
реальные данные соревнования не используются.
"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401  # добавляет submission/ в sys.path
import numpy
import rasterio
from koz3.discovery import _markup_type_of, find_regions
from rasterio.crs import CRS
from rasterio.transform import from_origin


def _rel_names(paths: list[Path], region: Path) -> list[str]:
    """Относительные пути файлов региона, отсортированные."""
    return sorted(path.relative_to(region).as_posix() for path in paths)


def _las_with_vlr(wkt: str) -> bytes:
    """LAS 1.4 с одним VLR 2112 (OGC WKT)."""
    wkt_bytes = wkt.encode("utf-8")
    header = bytearray(375)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 4
    struct.pack_into("<H", header, 94, 375)
    struct.pack_into("<I", header, 100, 1)
    vlr = struct.pack(
        "<H16sHH32s", 0, b"LASF_Projection", 2112, len(wkt_bytes), b"OGC WKT CRS"
    )
    return bytes(header) + vlr + wkt_bytes


def _las_with_evlr(wkt: str) -> bytes:
    """LAS 1.4 без VLR; WKT лежит в первом EVLR."""
    wkt_bytes = wkt.encode("utf-8")
    header = bytearray(375)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 4
    struct.pack_into("<H", header, 94, 375)
    struct.pack_into("<I", header, 100, 0)
    struct.pack_into("<Q", header, 235, 375)
    struct.pack_into("<I", header, 243, 1)
    evlr = struct.pack(
        "<H16sHQ32s", 0, b"LASF_Projection", 2112, len(wkt_bytes), b"OGC WKT CRS"
    )
    return bytes(header) + evlr + wkt_bytes


def _las_12_with_vlr(wkt: str) -> bytes:
    """LAS 1.2 с WKT в VLR; user_id — мусор, заполняющий байты 235-244.

    В версиях до 1.4 поля EVLR (смещение 235, счётчик 243) принадлежат
    другим данным: чтение CRS не должно на них реагировать.
    """
    wkt_bytes = wkt.encode("utf-8")
    header = bytearray(227)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 2
    struct.pack_into("<H", header, 94, 227)
    struct.pack_into("<I", header, 100, 1)
    vlr = struct.pack(
        "<H16sHH32s", 0, b"GARBAGEUSERID01", 2112, len(wkt_bytes), b"OGC WKT CRS"
    )
    return bytes(header) + vlr + wkt_bytes


def _write_geotiff(path: Path) -> None:
    """Однопиксельный GeoTIFF в EPSG:32637."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="uint8",
        crs="EPSG:32637",
        transform=from_origin(500000.0, 6000000.0, 1.0, 1.0),
    ) as dataset:
        dataset.write(numpy.zeros((1, 1), dtype="uint8"), 1)


class DiscoveryTest(unittest.TestCase):
    """Интеграционные тесты find_regions на синтетических заглушках."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.input_dir = Path(self._tmp.name) / "input"
        self.input_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_files_by_markup_type(self) -> None:
        region = self.input_dir / "region_0001"
        layout = {
            "La": ["ground.las", "strip.laz"],
            "SpOr": ["scene.tif"],
            "Or": ["ortho.tif"],
            "Ae": ["aero.tif"],
            "MagGr": ["map.tif"],
            "EtGr": ["profile.csv"],
            "TerFl": ["thermal.tif"],
            "Li": ["hillshade_c.tif"],
        }
        for folder, names in layout.items():
            (region / folder).mkdir(parents=True)
            for name in names:
                (region / folder / name).write_bytes(b"")
        (region / "UTM.json").write_text(
            json.dumps({"crs": "EPSG:32637"}), encoding="utf-8"
        )

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual("region_0001", regions[0].name)
        self.assertEqual(region, regions[0].path)
        self.assertEqual("EPSG:32637", regions[0].utm_crs)
        files = regions[0].files
        expected = {
            "li": ["La/ground.las", "La/strip.laz"],
            "sp_or": ["SpOr/scene.tif"],
            "or": ["Or/ortho.tif"],
            "ae": ["Ae/aero.tif"],
            "maggr": ["MagGr/map.tif"],
            "etgr": ["EtGr/profile.csv"],
            "terfl": ["TerFl/thermal.tif"],
        }
        self.assertEqual(set(expected), set(files))
        for markup_type, names in expected.items():
            with self.subTest(markup_type=markup_type):
                self.assertEqual(names, _rel_names(files[markup_type], region))
        # Критерий 3: хиллшейды Li не попадают ни в один тип.
        listed = [path for paths in files.values() for path in paths]
        self.assertNotIn(region / "Li" / "hillshade_c.tif", listed)
        # Детерминизм: повторный вызов даёт тот же результат.
        self.assertEqual(regions, find_regions(self.input_dir))

    def test_input_without_subdirs_is_single_region(self) -> None:
        (self.input_dir / "cloud.las").write_bytes(b"")
        (self.input_dir / "UTM.json").write_text(
            json.dumps({"utm": "37N"}), encoding="utf-8"
        )

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual(self.input_dir, regions[0].path)
        self.assertEqual(self.input_dir.name, regions[0].name)
        self.assertEqual("EPSG:32637", regions[0].utm_crs)
        self.assertEqual(
            ["cloud.las"], _rel_names(regions[0].files["li"], self.input_dir)
        )

    def test_crs_from_utm_json(self) -> None:
        cases = [
            {"crs": "EPSG:32637"},
            {"crs": "urn:ogc:def:crs:epsg::32637"},
            {"crs": "EPSG32637"},
            {"crs": "EPSG: 32637"},
            {"utm": "37N"},
            {"utm": "37"},
            {"utm": 37},
        ]
        tmp_root = Path(self._tmp.name)
        for index, payload in enumerate(cases):
            with self.subTest(payload=payload):
                input_dir = tmp_root / f"input_{index:02d}"
                region = input_dir / "site"
                region.mkdir(parents=True)
                (region / "UTM.json").write_text(json.dumps(payload), encoding="utf-8")
                regions = find_regions(input_dir)
                self.assertEqual(1, len(regions))
                self.assertEqual("EPSG:32637", regions[0].utm_crs)

    def test_crs_from_las_vlr(self) -> None:
        wkt = CRS.from_epsg(32637).to_wkt()
        region = self.input_dir / "site"
        (region / "La").mkdir(parents=True)
        (region / "La" / "ground.las").write_bytes(_las_with_vlr(wkt))

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual("EPSG:32637", regions[0].utm_crs)

    def test_crs_from_las_evlr(self) -> None:
        wkt = CRS.from_epsg(32637).to_wkt()
        region = self.input_dir / "site"
        (region / "La").mkdir(parents=True)
        (region / "La" / "cloud.laz").write_bytes(_las_with_evlr(wkt))

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual("EPSG:32637", regions[0].utm_crs)

    def test_crs_from_las_12_ignores_evlr_fields(self) -> None:
        wkt = CRS.from_epsg(32637).to_wkt()
        region = self.input_dir / "site"
        (region / "La").mkdir(parents=True)
        (region / "La" / "ground.las").write_bytes(_las_12_with_vlr(wkt))

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual("EPSG:32637", regions[0].utm_crs)

    def test_crs_from_raster(self) -> None:
        region = self.input_dir / "site"
        (region / "Ae").mkdir(parents=True)
        _write_geotiff(region / "Ae" / "aero.tif")

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual("EPSG:32637", regions[0].utm_crs)

    def test_geographic_utm_json_falls_back_to_raster(self) -> None:
        region = self.input_dir / "site"
        (region / "Ae").mkdir(parents=True)
        (region / "UTM.json").write_text(
            json.dumps({"crs": "EPSG:4326"}), encoding="utf-8"
        )
        _write_geotiff(region / "Ae" / "aero.tif")

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual("EPSG:32637", regions[0].utm_crs)

    def test_region_without_crs_is_skipped(self) -> None:
        region = self.input_dir / "site"
        (region / "Or").mkdir(parents=True)
        (region / "Or" / "ortho.tif").write_bytes(b"")  # не растр

        with self.assertLogs("koz3.discovery", level="WARNING") as logs:
            regions = find_regions(self.input_dir)

        self.assertEqual([], regions)
        self.assertIn("пропущен", "\n".join(logs.output))

    def test_hidden_entries_are_ignored(self) -> None:
        region = self.input_dir / "site"
        (region / ".hidden").mkdir(parents=True)
        (region / ".hidden" / "cloud.las").write_bytes(b"")
        (region / ".DS_Store").write_bytes(b"")
        (region / "UTM.json").write_text(
            json.dumps({"crs": "EPSG:32637"}), encoding="utf-8"
        )

        regions = find_regions(self.input_dir)

        self.assertEqual(1, len(regions))
        self.assertEqual({}, regions[0].files)

    def test_missing_input_dir_returns_empty(self) -> None:
        with self.assertLogs("koz3.discovery", level="ERROR"):
            regions = find_regions(self.input_dir / "no_such_dir")

        self.assertEqual([], regions)


class MarkupTypeTest(unittest.TestCase):
    """Граничные случаи классификации меток в относительном пути."""

    def test_labels_with_separators(self) -> None:
        cases = {
            "Or/photo_2023.tif": "or",
            "Or_2/scene.tif": "or",
            "sp_or/block1.tif": "sp_or",
            "SpOr/scene.tif": "sp_or",
            "Ae/aerofoto.tif": "ae",
            "La/dsm.tif": "li",
            "MagGr/map.tif": "maggr",
            "EtGr/profile.csv": "etgr",
            "TerFl/thermal.tif": "terfl",
            "Li/cloud.las": "li",
        }
        for rel_path, expected in cases.items():
            with self.subTest(rel_path=rel_path):
                self.assertEqual(expected, _markup_type_of(Path(rel_path)))

    def test_words_do_not_match_labels(self) -> None:
        for rel_path in (
            "Terrain/flat_rock.tif",
            "North/area.tif",
            "raster_north.tif",
            "Hillshades/outline.tif",
            "polyline.zip",
            "Aerial/photo.tif",
            "Las/dsm.tif",
            "readme.txt",
            "UTM.json",
        ):
            with self.subTest(rel_path=rel_path):
                self.assertIsNone(_markup_type_of(Path(rel_path)))

    def test_li_hillshade_is_excluded(self) -> None:
        self.assertIsNone(_markup_type_of(Path("Li/hillshade_c.tif")))


if __name__ == "__main__":
    unittest.main()