"""Шаг 4. Обучение YOLO-seg на датасете из build_dataset.py.

Аугментации выбраны под хиллшейды:
- hsv_h = 0: в цветных отмывках (c, ch) оттенок кодирует высоту, сдвиг оттенка
  ломает смысл;
- отражения и повороты по умолчанию выключены: в серых отмывках (g, i) тень
  задаёт направление освещения (азимут 315°), а на инференсе оно всегда одно.
  Включение (--flips) — отдельный эксперимент со сравнением на val.
Веса по окончании копируются в --export-dir под именем из koz3.config.YOLO.

Запуск:
  python training/train.py --data /датасет/dataset.yaml --model yolo11s-seg.pt \
      --epochs 100 --batch 8 --name lidar_v1 [--flips]
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import settings
from koz3 import config as runtime_config


def sha256(path: Path) -> str:
    """Хеш файла весов — для журнала команды («Модели и веса»)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Обучить модель и сохранить лучшие веса с паспортом."""
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--model", default="yolo11s-seg.pt", help="стартовые веса (s — быстрее, m — точнее)"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="lidar_v1")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--export-dir", type=Path, default=Path("weights_out"))
    parser.add_argument("--flips", action="store_true", help="включить отражения (эксперимент)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=settings.TILE_PX,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        seed=settings.SEED,
        deterministic=True,
        patience=25,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.1,
        scale=0.25,
        fliplr=0.5 if args.flips else 0.0,
        flipud=0.5 if args.flips else 0.0,
        mosaic=1.0,
        overlap_mask=False,  # маски объектов не сливаются при наложении
        plots=True,
    )
    best = Path(model.trainer.best)
    names = YOLO(str(best)).names
    expected = dict(enumerate(settings.CLASSES))
    if {int(k): v for k, v in names.items()} != expected:
        raise RuntimeError(
            f"классы модели {names} не совпадают с config.SURFACE_CLASSES {expected}"
        )

    args.export_dir.mkdir(parents=True, exist_ok=True)
    target = args.export_dir / str(runtime_config.YOLO["weights_pt"])
    shutil.copy2(best, target)
    passport = {
        "file": target.name,
        "sha256": sha256(target),
        "ultralytics": ultralytics_version,
        "start_weights": args.model,
        "epochs": args.epochs,
        "run": str(best.parent.parent),
        "classes": settings.CLASSES,
        "flips": args.flips,
    }
    target.with_suffix(".pt.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(passport, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
