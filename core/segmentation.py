"""Gan nhan thanh phan lien thong (two-pass + union-find) va bounding box."""
import numpy as np


def connected_components_labeling(binary):
    """Lien thong-8. Tra ve mang nhan int32 (0 = nen)."""
    h, w = binary.shape
    b = binary.tolist()
    lab = [[0] * w for _ in range(h)]
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for y in range(h):
        row, cur = b[y], lab[y]
        up = lab[y - 1] if y > 0 else None
        for x in range(w):
            if not row[x]:
                continue
            nb = []
            if x > 0 and cur[x - 1]:
                nb.append(cur[x - 1])
            if up is not None:
                if x > 0 and up[x - 1]:
                    nb.append(up[x - 1])
                if up[x]:
                    nb.append(up[x])
                if x < w - 1 and up[x + 1]:
                    nb.append(up[x + 1])
            if not nb:
                parent.append(len(parent))
                cur[x] = len(parent) - 1
            else:
                roots = [find(n) for n in nb]
                r = min(roots)
                cur[x] = r
                for q in roots:
                    if q != r:
                        parent[q] = r
    lut = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    return lut[np.array(lab, dtype=np.int32)]


def get_bounding_boxes(labels):
    """Tra ve list dict {box:(x0,y0,x1,y1), area} cho moi nhan."""
    ys, xs = np.nonzero(labels)
    if len(ys) == 0:
        return []
    ids, inv = np.unique(labels[ys, xs], return_inverse=True)
    n = len(ids)
    x0 = np.full(n, 10 ** 9); y0 = np.full(n, 10 ** 9)
    x1 = np.full(n, -1); y1 = np.full(n, -1)
    np.minimum.at(x0, inv, xs); np.minimum.at(y0, inv, ys)
    np.maximum.at(x1, inv, xs); np.maximum.at(y1, inv, ys)
    area = np.bincount(inv)
    return [dict(box=(int(x0[i]), int(y0[i]), int(x1[i]), int(y1[i])), area=int(area[i]))
            for i in range(n)]


def detect_diagonal_gutters(gray, min_white=0.75, min_edge=0.45):
    """Tim gutter trang cheo dai bang bien doi duong thang ``y = a*x + b``.

    Moi duong ung vien duoc cham theo ti le pixel gan trang tren toan chieu rong.
    Mot gutter chi duoc chap nhan khi co them mot canh toi lien tuc o mot trong
    hai phia. Tra ve cac cap ``(a, b)`` theo thu tu tu tren xuong duoi.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 16:
        return []
    h, w = g.shape
    yy, xx = np.indices((h, w))
    yf, xf = yy.ravel(), xx.ravel()
    white = (g >= 245).ravel().astype(np.float64)
    slopes = np.concatenate((np.linspace(-0.6, -0.08, 53),
                             np.linspace(0.08, 0.6, 53)))
    candidates = []
    for a in slopes:
        bins = yf - np.rint(a * xf).astype(np.int32)
        offset = -int(bins.min())
        idx = bins + offset
        sums = np.bincount(idx, weights=white)
        counts = np.bincount(idx)
        scores = sums / np.maximum(counts, 1)
        for k in np.flatnonzero(scores >= min_white):
            b = k - offset
            y0, y1 = b, a * (w - 1) + b
            center = (y0 + y1) / 2
            if 0 <= y0 < h and 0 <= y1 < h and 0.08 * h < center < 0.92 * h:
                candidates.append((float(scores[k]), float(a), float(b), float(center)))

    x = np.arange(w)
    selected = []
    for score, a, b, center in sorted(candidates, reverse=True):
        y = np.rint(a * x + b).astype(np.int32)
        edge_up = np.mean(g[np.clip(y - 5, 0, h - 1), x] < 180)
        edge_down = np.mean(g[np.clip(y + 5, 0, h - 1), x] < 180)
        if max(edge_up, edge_down) < min_edge:
            continue
        if all(abs(center - old[3]) > 0.10 * h for old in selected):
            selected.append((score, a, b, center))
        if len(selected) >= 4:
            break
    return [(a, b) for _, a, b, _ in sorted(selected, key=lambda item: item[3])]


def filter_boxes(boxes, page_h, page_w, min_frac=0.03, max_frac=0.95):
    """Giu box co dien tich bbox trong [min_frac, max_frac] dien tich trang
    (loai chu thoai/bui ran va vien ngoai trang)."""
    page_area = page_h * page_w
    keep = []
    for b in boxes:
        x0, y0, x1, y1 = b["box"]
        frac = (x1 - x0 + 1) * (y1 - y0 + 1) / page_area
        if min_frac <= frac <= max_frac:
            keep.append(b["box"])
    return keep


def sort_reading_order(boxes, row_tol=0.5):
    """Sap xep theo thu tu doc (tren->duoi, trai->phai)."""
    boxes = sorted(boxes, key=lambda b: b[1])
    rows, out = [], []
    for b in boxes:
        same_row = False
        if rows:
            for other in rows[-1]:
                overlap = min(b[3], other[3]) - max(b[1], other[1]) + 1
                min_h = min(b[3] - b[1] + 1, other[3] - other[1] + 1)
                if overlap >= (1 - row_tol) * min_h:
                    same_row = True
                    break
        if same_row:
            rows[-1].append(b)
        else:
            rows.append([b])
    for r in rows:
        out.extend(sorted(r, key=lambda b: b[0]))
    return out


def suppress_nested(boxes, tol=0.9):
    """Bo cac box nam long (>= tol dien tich) ben trong box lon hon:
    noi dung ben trong khung (nhan vat, net ve) khong phai khung tranh."""
    area = lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    keep = []
    for i, b in enumerate(boxes):
        nested = False
        for j, o in enumerate(boxes):
            if i != j and area(o) > area(b):
                ix = max(0, min(b[2], o[2]) - max(b[0], o[0]) + 1)
                iy = max(0, min(b[3], o[3]) - max(b[1], o[1]) + 1)
                if ix * iy >= tol * area(b):
                    nested = True
                    break
        if not nested:
            keep.append(b)
    return keep
