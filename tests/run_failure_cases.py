"""Generate and run synthetic robustness cases outside test_cases/."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import process


ROOT = Path("tests/tmp_cases/step7_20261014")
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"


def border(image, box, color=(15, 15, 15), thick=4):
    x0, y0, x1, y1 = box
    image[y0:y0 + thick, x0:x1 + 1] = color
    image[y1 - thick + 1:y1 + 1, x0:x1 + 1] = color
    image[y0:y1 + 1, x0:x0 + thick] = color
    image[y0:y1 + 1, x1 - thick + 1:x1 + 1] = color


def line(image, p0, p1, color=(35, 35, 35), thick=3):
    x0, y0 = p0
    x1, y1 = p1
    count = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.rint(np.linspace(x0, x1, count)).astype(int)
    ys = np.rint(np.linspace(y0, y1, count)).astype(int)
    radius = thick // 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx = np.clip(xs + dx, 0, image.shape[1] - 1)
            yy = np.clip(ys + dy, 0, image.shape[0] - 1)
            image[yy, xx] = color


def content(image, box, color=(45, 45, 45)):
    x0, y0, x1, y1 = box
    line(image, (x0 + 15, y0 + 20), (x1 - 20, y1 - 25), color, 2)
    line(image, (x0 + 20, y1 - 25), (x1 - 30, y0 + 35), color, 2)


def regular_four(background):
    image = background.copy()
    boxes = [(20, 20, 140, 190), (160, 20, 280, 190),
             (20, 210, 140, 380), (160, 210, 280, 380)]
    for box in boxes:
        border(image, box)
        content(image, box)
    return image


def build_cases():
    cases = {}

    no_gutter = np.full((400, 300, 3), 255, np.uint8)
    border(no_gutter, (5, 5, 294, 394), thick=5)
    line(no_gutter, (150, 5), (150, 394), thick=5)
    line(no_gutter, (5, 200), (294, 200), thick=5)
    cases["no_gutter_expected4"] = (no_gutter, 4)

    tilted = np.full((400, 300, 3), 248, np.uint8)
    quads = [((20, 30), (135, 15), (145, 180), (30, 195)),
             ((165, 15), (280, 35), (270, 195), (155, 180)),
             ((30, 215), (145, 205), (135, 380), (15, 365)),
             ((160, 210), (285, 225), (275, 380), (150, 365))]
    for q in quads:
        for a, b in zip(q, q[1:] + q[:1]):
            line(tilted, a, b, thick=4)
    cases["tilted_frames_expected4"] = (tilted, 4)

    rng = np.random.default_rng(20261014)
    texture = np.clip(rng.normal(190, 18, (400, 300, 1)), 0, 255).astype(np.uint8)
    texture = np.repeat(texture, 3, axis=2)
    cases["gray_texture_expected4"] = (regular_four(texture), 4)

    color_bg = np.empty((400, 300, 3), dtype=np.uint8)
    color_bg[:] = (80, 190, 230)
    cases["colored_background_expected4"] = (regular_four(color_bg), 4)

    large = np.full((400, 300, 3), 255, np.uint8)
    border(large, (12, 12, 287, 387), thick=5)
    content(large, (12, 12, 287, 387))
    cases["single_large_expected1"] = (large, 1)

    cases["no_panel_expected0"] = (np.full((400, 300, 3), 255, np.uint8), 0)

    tiny = np.full((24, 24, 3), 255, np.uint8)
    border(tiny, (3, 3, 20, 20), thick=1)
    cases["very_small_expected1"] = (tiny, 1)

    gradient = np.linspace(65, 250, 300, dtype=np.uint8)[None, :, None]
    backlit = np.repeat(np.repeat(gradient, 400, axis=0), 3, axis=2)
    cases["backlit_expected4"] = (regular_four(backlit), 4)
    return cases


def main():
    INPUTS.mkdir(parents=True, exist_ok=False)
    RESULTS.mkdir(parents=True, exist_ok=False)
    for name, (image, expected) in build_cases().items():
        path = INPUTS / f"{name}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")
        try:
            boxes, elapsed = process(str(path), str(RESULTS))
            print(f"ROBUSTNESS {name}: crash=False expected={expected} predicted={len(boxes)} "
                  f"count_ok={len(boxes) == expected} time={elapsed:.6f}s boxes={boxes}")
        except Exception as exc:
            print(f"ROBUSTNESS {name}: crash=True expected={expected} "
                  f"error={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
