"""Chuyen doi mau (tu cai dat)."""
import numpy as np


def rgb_to_gray(img_bgr):
    """Anh BGR (cv2.imread) -> anh xam float64.
    I = 0.299 R + 0.587 G + 0.114 B  (nhan ma tran tren kenh thu 3)."""
    img = img_bgr.astype(np.float64)
    if img.ndim == 2:
        return img
    weights = np.array([0.114, 0.587, 0.299])  # thu tu B, G, R
    return img @ weights


def downscale_mean(gray, s):
    """Thu nho anh theo he so nguyen s bang trung binh khoi s x s."""
    if s <= 1:
        return gray
    h, w = gray.shape
    h2, w2 = h // s * s, w // s * s
    return gray[:h2, :w2].reshape(h2 // s, s, w2 // s, s).mean(axis=(1, 3))
