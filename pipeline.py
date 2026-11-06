"""Pipeline: anh -> gray -> (thu nho) -> Gauss -> Otsu -> dao anh -> Closing -> CCL -> box."""
import numpy as np
from core.color import rgb_to_gray, downscale_mean
from core.filters import gaussian_blur
from core.threshold import otsu_threshold
from core.morphology import rect_se, closing
from core.segmentation import (connected_components_labeling, get_bounding_boxes,
                               filter_boxes, sort_reading_order, suppress_nested,
                               detect_diagonal_gutters, detect_horizontal_gutters)

DEFAULT = dict(max_side=700, sigma=1.5, close_px=20, use_blur=True, use_closing=True,
               use_area_filter=True, use_nested=True, min_frac=0.03, max_frac=0.95,
               use_diagonal_gutter_fallback=True, use_horizontal_gutter_fallback=True)


def _has_strong_overlap(boxes, threshold=0.15):
    """Panel thật hiếm khi chồng lấn; CCL chồng lấn thường là box nội dung giả."""
    area = lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            iw = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
            ih = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
            if iw * ih > threshold * min(area(a), area(b)):
                return True
    return False


def _horizontal_boxes(gutters, scale, height, width):
    """Đổi các dải gutter trên ảnh nhỏ thành box panel trên ảnh gốc."""
    boxes, top = [], 0
    for y0, y1 in gutters:
        bottom = min(y0 * scale, height) - 1
        if bottom - top + 1 >= 0.06 * height:
            boxes.append((0, top, width - 1, bottom))
        top = min((y1 + 1) * scale, height)
    if height - top >= 0.06 * height:
        boxes.append((0, top, width - 1, height - 1))
    return boxes


def _diagonal_boxes(gutters, height, width):
    lines = [a * np.arange(width) + b for a, b in gutters]
    boxes = []
    for i in range(len(lines) + 1):
        y0 = 0 if i == 0 else max(0, int(np.floor(lines[i - 1].min())))
        y1 = height - 1 if i == len(lines) else min(height - 1, int(np.ceil(lines[i].max())))
        boxes.append((0, y0, width - 1, y1))
    return boxes


def segment_page(img_bgr, **kw):
    """Tra ve (boxes tren anh goc, dict anh trung gian)."""
    cfg = {**DEFAULT, **kw}
    H, W = img_bgr.shape[:2]
    s = int(np.ceil(max(H, W) / cfg["max_side"]))
    gray = rgb_to_gray(img_bgr)
    small = downscale_mean(gray, s)
    blur = gaussian_blur(small, cfg["sigma"]) if cfg["use_blur"] else small
    t = otsu_threshold(blur)
    ink = blur < t                       # DAO ANH: net ve = foreground (True)
    if cfg["use_closing"]:
        size = max(3, int(round(cfg["close_px"] / s)) | 1)
        ink_c = closing(ink, rect_se(size))
    else:
        ink_c = ink
    labels = connected_components_labeling(ink_c)
    boxes = get_bounding_boxes(labels)
    h, w = ink_c.shape
    if cfg["use_area_filter"]:
        boxes = filter_boxes(boxes, h, w, cfg["min_frac"], cfg["max_frac"])
    else:
        boxes = [b["box"] for b in boxes]
    if cfg["use_nested"]:
        boxes = suppress_nested(boxes)
    boxes = sort_reading_order([(x0 * s, y0 * s, min((x1 + 1) * s, W) - 1, min((y1 + 1) * s, H) - 1)
                                for x0, y0, x1, y1 in boxes])
    ccl_has_overlap = _has_strong_overlap(boxes)
    horizontal_gutters = []
    if cfg["use_horizontal_gutter_fallback"]:
        horizontal_gutters = detect_horizontal_gutters(small)
        horizontal_boxes = _horizontal_boxes(horizontal_gutters, s, H, W)
        if len(horizontal_boxes) >= 2 and (len(horizontal_boxes) > len(boxes) or ccl_has_overlap):
            boxes = horizontal_boxes

    diagonal_gutters = []
    if cfg["use_diagonal_gutter_fallback"] and 1 <= len(boxes) <= 4:
        gutters_small = detect_diagonal_gutters(blur)
        # Một gutter chỉ được dùng khi CCL tạo các box chồng lấn rõ rệt;
        # từ hai gutter trở lên là tín hiệu đủ mạnh để dùng trực tiếp.
        if len(gutters_small) >= 2 or (len(gutters_small) == 1 and ccl_has_overlap):
            diagonal_gutters = [(a, b * s) for a, b in gutters_small]
            diagonal_boxes = _diagonal_boxes(diagonal_gutters, H, W)
            if len(diagonal_boxes) > len(boxes) or ccl_has_overlap:
                boxes = diagonal_boxes
            else:
                diagonal_gutters = []
    inter = dict(gray=small, blur=blur, binary=ink, closed=ink_c, threshold=t, scale=s,
                 diagonal_gutters=diagonal_gutters, horizontal_gutters=horizontal_gutters)
    return boxes, inter
