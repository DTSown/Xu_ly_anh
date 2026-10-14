"""Chay: python main.py --input test_cases/case1_01.png --out output
       python main.py --all"""
import argparse, glob, os, time
import cv2                      # chi dung imread / imwrite
import numpy as np
from pipeline import segment_page


def draw_boxes(img, boxes, thick=4, color=(0, 0, 255)):
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        out[y0:y0 + thick, x0:x1 + 1] = color
        out[max(y1 - thick + 1, 0):y1 + 1, x0:x1 + 1] = color
        out[y0:y1 + 1, x0:x0 + thick] = color
        out[y0:y1 + 1, max(x1 - thick + 1, 0):x1 + 1] = color
    return out


def draw_gutters(img, gutters, thick=4, color=(0, 0, 255)):
    """Ve cac duong gutter cheo bang gan mang NumPy."""
    out = img.copy()
    h, w = out.shape[:2]
    x = np.arange(w)
    for a, b in gutters:
        y = np.rint(a * x + b).astype(np.int32)
        for offset in range(-(thick // 2), thick - thick // 2):
            yy = y + offset
            valid = (0 <= yy) & (yy < h)
            out[yy[valid], x[valid]] = color
    return out


def crop_panel(img, box, panel_index=0, gutters=()):
    """Cat bbox; voi gutter cheo thi to trang phan ngoai polygon cua panel."""
    x0, y0, x1, y1 = box
    crop = img[y0:y1 + 1, x0:x1 + 1].copy()
    if not gutters:
        return crop
    yy, xx = np.indices(crop.shape[:2])
    yy, xx = yy + y0, xx + x0
    keep = np.ones(crop.shape[:2], dtype=bool)
    if panel_index > 0:
        a, b = gutters[panel_index - 1]
        keep &= yy >= a * xx + b
    if panel_index < len(gutters):
        a, b = gutters[panel_index]
        keep &= yy <= a * xx + b
    crop[~keep] = 255
    return crop


def process(path, out_dir, **kw):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(out_dir, name)
    panels_dir = os.path.join(d, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    t0 = time.time()
    boxes, inter = segment_page(img, **kw)
    dt = time.time() - t0
    for old in glob.glob(os.path.join(panels_dir, "panel_*.png")):
        os.remove(old)
    u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{d}/1_gray.png", u8(inter["gray"]))
    cv2.imwrite(f"{d}/2_blur.png", u8(inter["blur"]))
    cv2.imwrite(f"{d}/3_binary_otsu_inverted.png", inter["binary"].astype(np.uint8) * 255)
    cv2.imwrite(f"{d}/4_closing.png", inter["closed"].astype(np.uint8) * 255)
    gutters = inter.get("diagonal_gutters", [])
    result = draw_gutters(img, gutters) if gutters else draw_boxes(img, boxes)
    cv2.imwrite(f"{d}/5_result.png", result)
    for i, box in enumerate(boxes, 1):
        panel = crop_panel(img, box, i - 1, gutters)
        cv2.imwrite(os.path.join(panels_dir, f"panel_{i:02d}.png"), panel)
    print(f"{name}: {len(boxes)} panel, otsu={inter['threshold']}, {dt:.2f}s")
    return boxes, dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input"); ap.add_argument("--out", default="output")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    paths = sorted(glob.glob("test_cases/*.png")) if a.all else [a.input]
    for p in paths:
        process(p, a.out)
