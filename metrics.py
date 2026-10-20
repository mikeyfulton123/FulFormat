"""
Image quality metrics for FUL Lab.

PSNR compares exact pixel values, so it punishes changes the eye can't see
(such as removing noise). SSIM compares local structure: brightness, contrast
and texture patterns in small Gaussian windows, which tracks human judgement
much better. 1.0 means identical; above ~0.98 is usually hard to tell apart.

MS-SSIM (multi-scale SSIM) repeats the comparison at 5 scales, halving the
image each time, and combines them with standard weights. It is less tied to
exact pixel structure at the finest scale, so it is a useful second opinion
on textures. Computed on brightness (Y).

SSIM here is computed in YCbCr and weighted 6:1:1 (brightness counts six
times as much as each colour channel), a common convention in codec testing.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def _ycbcr(rgb: np.ndarray) -> np.ndarray:
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
    return rgb.astype(np.float64) @ m.T


def _ssim_plane(x: np.ndarray, y: np.ndarray) -> float:
    blur = lambda img: gaussian_filter(img, 1.5, truncate=3.5)
    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    s = ((2 * mx * my + C1) * (2 * cov + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2))
    return float(s.mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Colour SSIM of two (H, W, 3) uint8 images."""
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape")
    if np.array_equal(a, b):
        return 1.0
    ya, yb = _ycbcr(a), _ycbcr(b)
    s = [_ssim_plane(ya[..., i], yb[..., i]) for i in range(3)]
    return (6 * s[0] + s[1] + s[2]) / 8


MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _ssim_parts(x, y):
    blur = lambda img: gaussian_filter(img, 1.5, truncate=3.5)
    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    lum = (2 * mx * my + C1) / (mx * mx + my * my + C1)
    cs = (2 * cov + C2) / (vx + vy + C2)
    return float(lum.mean()), float(cs.mean())


def ms_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Multi-scale SSIM on brightness, 5 scales (fewer for small images)."""
    if np.array_equal(a, b):
        return 1.0
    x, y = _ycbcr(a)[..., 0], _ycbcr(b)[..., 0]
    scales = min(len(MS_WEIGHTS), max(1, int(np.log2(min(x.shape) / 16)) + 1))
    w = np.array(MS_WEIGHTS[:scales]) / sum(MS_WEIGHTS[:scales])
    value = 1.0
    for i in range(scales):
        lum, cs = _ssim_parts(x, y)
        part = cs if i < scales - 1 else lum * cs
        value *= max(part, 1e-6) ** w[i]
        h, ww = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
        x = x[:h, :ww].reshape(h // 2, 2, ww // 2, 2).mean(axis=(1, 3))
        y = y[:h, :ww].reshape(h // 2, 2, ww // 2, 2).mean(axis=(1, 3))
    return float(value)


def texture_energy_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Fine-detail energy of b relative to a (1.0 = same amount of texture)."""
    from scipy.ndimage import uniform_filter
    ya, yb = _ycbcr(a)[..., 0], _ycbcr(b)[..., 0]
    ea = (ya - uniform_filter(ya, 3)).std()
    eb = (yb - uniform_filter(yb, 3)).std()
    return float(eb / ea) if ea > 0 else 1.0
