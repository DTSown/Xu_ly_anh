"""Phan nguong Otsu (tu cai dat)."""
import numpy as np


def histogram(gray_u8):
    return np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)


def otsu_threshold(gray):
    """Tra ve nguong k toi uu: cuc dai phuong sai giua hai nhom
    sigma_B^2(k) = (m_G*P1(k) - m(k))^2 / (P1(k)*(1-P1(k)))."""
    g = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    p = histogram(g)
    if np.count_nonzero(p) <= 1:
        return 0
    p /= p.sum()
    i = np.arange(256)
    P1 = np.cumsum(p)          # tong tich luy
    m = np.cumsum(p * i)       # trung binh tich luy
    mG = m[-1]
    denom = P1 * (1 - P1)
    valid = denom > 0
    if not np.any(valid):
        return 0
    sigma_b2 = np.full(256, -np.inf)
    sigma_b2[valid] = (mG * P1[valid] - m[valid]) ** 2 / denom[valid]
    return int(np.argmax(sigma_b2))
