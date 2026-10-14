"""Loc Gauss va tich chap 2D (vector hoa bang dich mang)."""
import numpy as np


def gaussian_kernel_1d(sigma):
    k = int(np.ceil(3 * sigma))
    x = np.arange(-k, k + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel(sigma):
    """Kernel Gauss 2D (2k+1)x(2k+1): G(s,t) = 1/(2*pi*sigma^2) * exp(-(s^2+t^2)/(2 sigma^2)),
    chuan hoa tong = 1."""
    g1 = gaussian_kernel_1d(sigma)
    g2 = np.outer(g1, g1)
    return g2 / g2.sum()


def convolve2d(img, kernel, pad_mode="edge"):
    """Tich chap 2D 'same'. Padding: 'edge' (nhan ban bien) hoac 'zero'.
    Moi phan tu kernel -> 1 phep dich mang (khong vong lap theo pixel)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    mode = "constant" if pad_mode == "zero" else "edge"
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=mode)
    flipped = kernel[::-1, ::-1]  # tich chap that su (lat kernel)
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += flipped[i, j] * padded[i:i + h, j:j + w]
    return out


def gaussian_blur(img, sigma):
    """Loc Gauss tach duoc: 2 lan tich chap 1D (nhanh hon 2D day du)."""
    g = gaussian_kernel_1d(sigma)
    k = len(g) // 2
    h, w = img.shape
    p = np.pad(img, ((0, 0), (k, k)), mode="edge")
    tmp = sum(g[i] * p[:, i:i + w] for i in range(len(g)))
    p = np.pad(tmp, ((k, k), (0, 0)), mode="edge")
    return sum(g[i] * p[i:i + h, :] for i in range(len(g)))
