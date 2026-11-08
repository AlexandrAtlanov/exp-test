"""Режим «вся отмывка целиком», как в бейзлайне (LIDAR["infer_mode"] = "whole").

Модель бейзлайна (model.pt) обучена на отмывках, сжатых ЦЕЛИКОМ до 1024 px по
длинной стороне и выровненных поворотом (solution.py бейзлайна: prepare_image →
downscale_image → align_and_crop_image → один кадр в модель). Эти функции —
дословный перенос, чтобы модель видела те же пиксели. Тайлы 0.1 м (режим
"tiles") показывают ей объекты в 3–7 раз крупнее, чем при обучении, и она почти
ничего не находит (первая отправка 08.11: 0.0138).

Отличие от бейзлайна — обратное преобразование. Бейзлайн возвращает полигоны
поворотом на -angle, а cv2.getRotationMatrix2D в осях изображения (y вниз)
поворачивает в другую сторону: при ненулевом угле его детекции смещены.
Здесь пиксель кадра переводится в пиксель исходной отмывки точным обращением
матрицы поворота (проверено тестом на синтетике), затем — привязкой растра.
"""

import cv2
import numpy as np
from PIL import Image
from rasterio.transform import Affine


def downscale_image(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float]:
    """Сжать (H, W, 3) так, чтобы длинная сторона стала target_size (LANCZOS, как бейзлайн).

    Возвращает (кадр, scale); scale = target_size / max(H, W), 1.0 — без сжатия.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= target_size:
        return image.copy(), 1.0
    scale = target_size / max_dim
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = Image.fromarray(image).resize(new_size, Image.LANCZOS)
    return np.asarray(resized), scale


def align_and_crop_image(
    image: np.ndarray,
) -> tuple[np.ndarray, float, tuple[int, int], tuple[float, float]]:
    """Повернуть кадр по minAreaRect области данных и обрезать фон (дословно бейзлайн).

    Возвращает (кадр RGB, angle в градусах для cv2.getRotationMatrix2D,
    смещение обрезки (x, y), центр поворота (x, y)) — всё в пикселях сжатого кадра.
    """
    img_rgb = image if image.shape[-1] == 3 else np.repeat(image[:, :, :1], 3, axis=-1)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edge = max(1, min(20, h // 10, w // 10))
    edge_regions = [gray[:edge, :], gray[-edge:, :], gray[:, :edge], gray[:, -edge:]]
    edge_means = [float(region.mean()) for region in edge_regions if region.size]
    background_sample = sum(edge_means) / len(edge_means) if edge_means else float(gray.mean())
    dark_background = background_sample < 128
    thresh_type = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, binary = cv2.threshold(gray, 30 if dark_background else 0, 255, thresh_type)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb, 0.0, (0, 0), (w / 2.0, h / 2.0)
    main = max(contours, key=cv2.contourArea)
    center, size, angle = cv2.minAreaRect(main)
    if size[0] < size[1]:
        angle += 90.0
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    border = (0, 0, 0) if dark_background else (255, 255, 255)
    rotated = cv2.warpAffine(
        img_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thresh_rot = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray_rotated, 30 if dark_background else 250, 255, thresh_rot)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB), angle, (0, 0), center
    x, y, w_box, h_box = cv2.boundingRect(coords)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w_box = min(rotated.shape[1] - x, w_box + 2 * margin)
    h_box = min(rotated.shape[0] - y, h_box + 2 * margin)
    crop = rotated[y : y + h_box, x : x + w_box]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), angle, (x, y), center


def frame_to_raster(
    raster_transform: Affine,
    scale: float,
    angle: float,
    crop_offset: tuple[int, int],
    center: tuple[float, float],
) -> Affine:
    """Аффинное преобразование «пиксель кадра модели → координаты CRS отмывки».

    Цепочка обратная подготовке кадра: сдвиг обрезки → обратный поворот
    (обращение cv2.getRotationMatrix2D(center, angle, 1)) → растяжение 1/scale
    → привязка растра. Координаты пикселей — по краям (как rasterio.features).
    """
    rot = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), 1.0)
    forward = Affine(*rot[0], *rot[1])  # сжатый кадр → повёрнутый кадр
    back = ~forward  # повёрнутый кадр → сжатый кадр
    crop = Affine.translation(float(crop_offset[0]), float(crop_offset[1]))
    return raster_transform * Affine.scale(1.0 / scale) * back * crop
