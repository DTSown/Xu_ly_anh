"""Pipeline: anh -> gray -> (thu nho) -> Gauss -> Otsu -> dao anh -> Closing -> CCL -> box."""
import numpy as np
from core.color import rgb_to_gray, downscale_mean
from core.filters import gaussian_blur
from core.threshold import otsu_threshold
from core.morphology import rect_se, closing
from core.segmentation import (connected_components_labeling, get_bounding_boxes,
                               filter_boxes, sort_reading_order, suppress_nested,
                               detect_diagonal_gutters)

DEFAULT = dict(max_side=700, sigma=1.5, close_px=20, use_blur=True, use_closing=True,
               use_area_filter=True, use_nested=True, min_frac=0.03, max_frac=0.95,
               use_diagonal_gutter_fallback=True)


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
    diagonal_gutters = []
    if cfg["use_diagonal_gutter_fallback"] and 2 <= len(boxes) <= 4:
        gutters_small = detect_diagonal_gutters(blur)
        if len(gutters_small) >= 2:
            diagonal_gutters = [(a, b * s) for a, b in gutters_small]
            lines = [a * np.arange(W) + b for a, b in diagonal_gutters]
            boxes = []
            for i in range(len(lines) + 1):
                y0 = 0 if i == 0 else max(0, int(np.floor(lines[i - 1].min())))
                y1 = H - 1 if i == len(lines) else min(H - 1, int(np.ceil(lines[i].max())))
                boxes.append((0, y0, W - 1, y1))
    inter = dict(gray=small, blur=blur, binary=ink, closed=ink_c, threshold=t, scale=s,
                 diagonal_gutters=diagonal_gutters)
    return boxes, inter
