"""Шаг 2. Совпадают ли готовые Li-хиллшейды трейна с пересчётом из LAS?

Отвечает на вопрос, можно ли строить датасет быстро из готовых Li
(--source li) или нужно пересчитывать из LAS (--source las, по умолчанию).
Для одного региона: пересчитывает хиллшейды из LAS кодом бейзлайна и
сравнивает с Li того же варианта: размер, шаг, границы, средняя разница.

Запуск: python training/check_hillshades.py --data /трейн --region 200_Судьбищи_2
"""

import argparse
import tempfile
from pathlib import Path

import hillshades
import numpy as np
import rasterio

from koz3.discovery import find_regions


def describe(path: Path) -> tuple[np.ndarray, dict]:
    """Пиксели и паспорт растра."""
    with rasterio.open(path) as src:
        info = {
            "shape": (src.height, src.width),
            "res": src.res,
            "bounds": tuple(round(b, 2) for b in src.bounds),
            "crs": str(src.crs),
        }
        return src.read(), info


def main() -> None:
    """Сравнить пересчитанные и готовые хиллшейды одного региона."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    (region,) = [r for r in find_regions(args.data) if r.name == args.region]
    provided = {hillshades.tag_of(p): p for p in hillshades.provided_li(region.path)}
    with tempfile.TemporaryDirectory() as tmp:
        for las_path in region.files.get("li", [])[:1]:
            for path in hillshades.from_las(las_path, region.utm_crs, Path(tmp)):
                tag = hillshades.tag_of(path)
                ours, ours_info = describe(path)
                print(f"\n[{tag}] пересчёт: {ours_info}")
                if tag not in provided:
                    print("  готового Li этого варианта нет")
                    continue
                theirs, theirs_info = describe(provided[tag])
                print(f"[{tag}] готовый Li: {theirs_info}")
                if ours.shape == theirs.shape:
                    diff = np.abs(ours.astype(np.int16) - theirs.astype(np.int16))
                    share = (diff > 2).mean()
                    mean = diff.mean()
                    print(f"  средняя |разница| {mean:.2f}; доля пикселей > 2: {share:.3%}")
                else:
                    print("  размеры не совпадают — использовать --source las")


if __name__ == "__main__":
    main()
