"""Hinh thai hoc nhi phan (tu cai dat). Foreground = True."""
import numpy as np


def rect_se(size):
    return np.ones((size, size), dtype=bool)


def _shift_reduce(binary, se, reduce_fn, pad_value):
    h, w = binary.shape
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    padded = np.pad(binary, ((ph, ph), (pw, pw)), constant_values=pad_value)
    out = None
    for i, j in zip(*np.nonzero(se)):
        window = padded[i:i + h, j:j + w]
        out = window.copy() if out is None else reduce_fn(out, window)
    return out


def dilate(binary, se):
    """Hit: chi can 1 diem cua SE cham foreground."""
    return _shift_reduce(binary, se, np.logical_or, False)


def erode(binary, se):
    """Fit: toan bo SE phai nam trong foreground."""
    return _shift_reduce(binary, se, np.logical_and, True)


def closing(binary, se):
    """Dilate roi erode: noi lien cac net dut / khe hep (tren anh foreground = net ve)."""
    return erode(dilate(binary, se), se)


def opening(binary, se):
    return dilate(erode(binary, se), se)
