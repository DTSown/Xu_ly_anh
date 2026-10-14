"""Sinh trang truyen gia lap (3 muc do) + ground truth, de chay duoc ngay.
Khi co anh that: bo vao test_cases/ va tao file <ten>.json = [[x0,y0,x1,y1], ...]."""
import json, os
import cv2
import numpy as np

H, W = 1100, 760
rng = np.random.default_rng(7)


def ellipse(img, cy, cx, ry, rx, val):
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    img[((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1] = val


def line(img, y0, x0, y1, x1, thick, val):
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    for t in np.linspace(0, 1, n):
        y, x = int(y0 + t * (y1 - y0)), int(x0 + t * (x1 - x0))
        img[max(y - thick, 0):y + thick + 1, max(x - thick, 0):x + thick + 1] = val


def fill_panel(img, b, tone=False):
    x0, y0, x1, y1 = b
    for _ in range(4):  # net ve ngau nhien ben trong
        line(img, rng.integers(y0 + 15, y1 - 15), rng.integers(x0 + 15, x1 - 15),
             rng.integers(y0 + 15, y1 - 15), rng.integers(x0 + 15, x1 - 15), 2, 30)
    ellipse(img, (y0 + y1) // 2, (x0 + x1) // 2, (y1 - y0) // 5, (x1 - x0) // 6, 40)
    if tone:  # hat tram
        yy, xx = np.mgrid[y0 + 6:y1 - 6, x0 + 6:x1 - 6]
        m = ((yy % 6) < 2) & ((xx % 6) < 2)
        img[y0 + 6:y1 - 6, x0 + 6:x1 - 6][m] = 90


def border(img, b, t=4):
    x0, y0, x1, y1 = b
    img[y0:y0 + t, x0:x1 + 1] = 0; img[y1 - t + 1:y1 + 1, x0:x1 + 1] = 0
    img[y0:y1 + 1, x0:x0 + t] = 0; img[y0:y1 + 1, x1 - t + 1:x1 + 1] = 0


def layout(case):
    m, g = 40, 32
    if case == 1 or case == 2:   # luoi 2 cot x 3 hang
        cw, ch = (W - 2 * m - g) // 2, (H - 2 * m - 2 * g) // 3
        return [(m + c * (cw + g), m + r * (ch + g), m + c * (cw + g) + cw, m + r * (ch + g) + ch)
                for r in range(3) for c in range(2)]
    # case 3: so le
    rows = [(m, 300, [0.45, 0.55]), (m + 300 + g, 360, [0.6, 0.4]), (m + 300 + g + 360 + g, 276, [0.5, 0.5])]
    boxes = []
    for y, hh, ratios in rows:
        avail, x = W - 2 * m - g, m
        for r in ratios:
            w = int(avail * r)
            boxes.append((x, y, x + w, y + hh)); x += w + g
    return boxes


def make(case, idx):
    img = np.full((H, W), 255.0)
    boxes = layout(case)
    if case == 3:
        img[:] = 235
    for i, b in enumerate(boxes):
        fill_panel(img, b, tone=(case == 3 and i % 2 == 0))
        border(img, b)
    if case == 2:  # net vien bi dut + bong thoai de len vien
        for b in boxes:
            x0, y0, x1, y1 = b
            for _ in range(2):
                gx = rng.integers(x0 + 30, x1 - 60)
                img[y0:y0 + 6, gx:gx + 9] = 255
                gy = rng.integers(y0 + 30, y1 - 60)
                img[gy:gy + 9, x1 - 6:x1 + 1] = 255
            for cy, cx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1)):  # dut o ca 4 goc
                img[max(cy - 5, 0):cy + 6, max(cx - 5, 0):cx + 6] = 255
            ellipse(img, y0 + 25, x0 + 90, 28, 60, 0); ellipse(img, y0 + 25, x0 + 90, 24, 56, 255)
    if case == 3:
        img += rng.normal(0, 8, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([img] * 3), boxes


if __name__ == "__main__":
    os.makedirs("test_cases", exist_ok=True)
    for case in (1, 2, 3):
        for k in (1, 2):
            img, boxes = make(case, k)
            name = f"case{case}_{k:02d}"
            cv2.imwrite(f"test_cases/{name}.png", img)
            json.dump([list(map(int, b)) for b in boxes], open(f"test_cases/{name}.json", "w"))
    print("done")
