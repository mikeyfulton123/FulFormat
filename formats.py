"""
Other image formats to compare FUL against.

  JPEG       (1992)  8x8 DCT blocks. The baseline everyone knows.
  JPEG 2000  (2000)  Wavelets, like FUL. Used in cinema and medical imaging.
  WebP       (2010)  Google; the image mode of the VP8 video codec.
  AVIF       (2019)  The image mode of the AV1 video codec. Very efficient,
                     slower to encode. (HEIC, Apple's format, is the same idea
                     built on HEVC video.)
  JPEG XL    (2021)  Designed as JPEG's successor; very efficient, fast.

Every format here is lossy at the settings used. Encoders come from Pillow
(JPEG, WebP) and imagecodecs (AVIF, JPEG XL, JPEG 2000); formats whose
library isn't installed are skipped.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

try:
    import imagecodecs
except ImportError:  # optional
    imagecodecs = None


@dataclass
class Format:
    name: str
    encode: Callable[[np.ndarray, float], bytes]
    decode: Callable[[bytes], np.ndarray]
    settings: list          # ordered from smallest file to largest
    label: Callable[[float], str]


def _pil_encode(fmt: str, **extra):
    def enc(rgb: np.ndarray, q: float) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, fmt, quality=int(q), **extra)
        return buf.getvalue()
    return enc


def _pil_decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _rgb(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return np.ascontiguousarray(a[..., :3]) if a.ndim == 3 else np.stack([a] * 3, -1)


def available_formats() -> list[Format]:
    out = [Format("JPEG", _pil_encode("JPEG", optimize=True), _pil_decode,
                  list(range(1, 101)), lambda q: f"quality {int(q)}")]
    try:
        _pil_encode("WEBP", method=6)(np.zeros((8, 8, 3), np.uint8), 50)
        out.append(Format("WebP", _pil_encode("WEBP", method=6), _pil_decode,
                          list(range(0, 101)), lambda q: f"quality {int(q)}"))
    except Exception:
        pass
    if imagecodecs is not None:
        if getattr(imagecodecs.JPEG2K, "available", False):
            out.append(Format(
                "JPEG 2000",
                lambda rgb, q: imagecodecs.jpeg2k_encode(rgb, level=float(q), reversible=False),
                lambda d: _rgb(imagecodecs.jpeg2k_decode(d)),
                [x / 2 for x in range(30, 141)], lambda q: f"target {q:g} dB"))
        if getattr(imagecodecs.AVIF, "available", False):
            out.append(Format(
                "AVIF",
                lambda rgb, q: imagecodecs.avif_encode(rgb, level=int(q), speed=6),
                lambda d: _rgb(imagecodecs.avif_decode(d)),
                list(range(0, 101)), lambda q: f"quality {int(q)}"))
        if getattr(imagecodecs.JPEGXL, "available", False):
            dists = sorted({round(x, 2) for x in np.geomspace(0.1, 15, 70)}, reverse=True)
            out.append(Format(
                "JPEG XL",
                lambda rgb, d: imagecodecs.jpegxl_encode(rgb, distance=float(d), effort=7),
                lambda d: _rgb(imagecodecs.jpegxl_decode(d)),
                dists, lambda d: f"distance {d:g}"))
    return out


def same_size(fmt: Format, rgb: np.ndarray, target: int):
    """Best setting whose file is no bigger than `target` bytes (binary
    search; sizes rise with quality). Returns (setting, data, pixels) or None
    if even the smallest setting is too big."""
    rgb = np.ascontiguousarray(rgb[..., :3])
    lo, hi = 0, len(fmt.settings) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = fmt.encode(rgb, fmt.settings[mid])
        if len(data) <= target:
            best = (fmt.settings[mid], data)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None
    return best[0], best[1], fmt.decode(best[1])
