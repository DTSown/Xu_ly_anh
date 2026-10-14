"""Chi so danh gia: IoU, Precision / Recall / F1 (khop tham lam)."""
def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / (area(a) + area(b) - inter)


def prf(pred, gt, thr=0.5):
    used, tp = set(), 0
    for p in pred:
        best, bj = 0, None
        for j, g in enumerate(gt):
            if j not in used and iou(p, g) > best:
                best, bj = iou(p, g), j
        if bj is not None and best >= thr:
            used.add(bj); tp += 1
    P = tp / len(pred) if pred else 0.0
    R = tp / len(gt) if gt else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F
