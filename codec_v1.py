"""
FUL v1: a small near-lossless image codec for learning.

Pipeline (encoder):
    RGB -> YCoCg-R -> optional 2x2 chroma subsampling
        -> MED prediction -> residual quantization -> adaptive Rice coding

The decoder runs the same steps in reverse. With q_luma = q_chroma = 1 and
no subsampling, the round trip is exactly lossless.

Run this file directly for a self-test:  python codec_v1.py
"""
from __future__ import annotations

import struct

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # Falls back to plain Python: correct, but much slower.
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


MAGIC = b"FUL\x00"   # 4 bytes: "FUL" plus a zero pad byte
VERSION = 1
# magic, version, width, height, q_luma, q_chroma, flags
HEADER = struct.Struct("<4sBIIBBB")
FLAG_SUBSAMPLE = 1

ESCAPE = 24      # unary run length at which we switch to a raw value
RAW_BITS = 16    # width of the raw escape value
MAX_K = 15

Y_RANGE = (0, 255)
C_RANGE = (-255, 255)


# --------------------------------------------------------------------------
# Stage 2: colour transform (YCoCg-R: integer only, perfectly reversible)
# --------------------------------------------------------------------------
def rgb_to_ycocg(rgb: np.ndarray):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    return y, co, cg


def ycocg_to_rgb(y, co, cg) -> np.ndarray:
    t = y - (cg >> 1)
    g = cg + t
    b = t - (co >> 1)
    r = b + co
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def subsample_2x2(p: np.ndarray) -> np.ndarray:
    """Average each 2x2 block (edges padded so odd sizes work)."""
    h, w = p.shape
    p = np.pad(p, ((0, h % 2), (0, w % 2)), mode="edge")
    s = p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]
    return ((s + 2) >> 2).astype(np.int32)


def upsample_2x2(p: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour upsampling. Swapping in bilinear is a good exercise."""
    return np.repeat(np.repeat(p, 2, axis=0), 2, axis=1)[:h, :w]


# --------------------------------------------------------------------------
# Stages 3 + 4: prediction and quantization
# --------------------------------------------------------------------------
@njit(cache=True)
def _med(a, b, c):
    """JPEG-LS median edge detector. a = left, b = above, c = above-left."""
    if c >= max(a, b):
        return min(a, b)
    if c <= min(a, b):
        return max(a, b)
    return a + b - c


@njit(cache=True)
def _context(img, y, x):
    """Neighbours for prediction, with simple rules at the image edges."""
    if y == 0:
        a = int(img[0, x - 1]) if x > 0 else 0
        return a, a, a
    b = int(img[y - 1, x])
    if x == 0:
        return b, b, b
    return int(img[y, x - 1]), b, int(img[y - 1, x - 1])


@njit(cache=True)
def _encode_plane(plane, q, lo, hi):
    h, w = plane.shape
    recon = np.zeros((h, w), np.int32)   # what the decoder will see
    out = np.empty(h * w, np.int32)
    half = q // 2
    i = 0
    for y in range(h):
        for x in range(w):
            a, b, c = _context(recon, y, x)   # predict from RECONSTRUCTED pixels
            pred = _med(a, b, c)
            d = int(plane[y, x]) - pred
            if d >= 0:
                qd = (d + half) // q
            else:
                qd = -((-d + half) // q)
            v = pred + qd * q
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            recon[y, x] = v
            out[i] = qd
            i += 1
    return out


@njit(cache=True)
def _decode_plane(res, h, w, q, lo, hi):
    recon = np.zeros((h, w), np.int32)
    i = 0
    for y in range(h):
        for x in range(w):
            a, b, c = _context(recon, y, x)
            v = _med(a, b, c) + int(res[i]) * q
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            recon[y, x] = v
            i += 1
    return recon


# --------------------------------------------------------------------------
# Stage 5: adaptive Rice coding
# --------------------------------------------------------------------------
@njit(cache=True)
def _pick_k(A, N):
    k = 0
    while (N << k) < A and k < MAX_K:
        k += 1
    return k


@njit(cache=True)
def _rice_encode(vals):
    n = vals.size
    buf = np.zeros(n * ((ESCAPE + RAW_BITS + 7) // 8) + 16, np.uint8)
    pos = 0          # bit position
    A = 4            # running sum of recent values
    N = 1            # running count
    for i in range(n):
        v = int(vals[i])
        z = (v << 1) if v >= 0 else ((-v) << 1) - 1      # zigzag
        k = _pick_k(A, N)
        hi = z >> k
        if hi < ESCAPE:
            for _ in range(hi):                           # quotient in unary
                buf[pos >> 3] = buf[pos >> 3] | (1 << (7 - (pos & 7)))
                pos += 1
            pos += 1                                      # terminating 0
            for bit in range(k - 1, -1, -1):              # remainder in k bits
                if (z >> bit) & 1:
                    buf[pos >> 3] = buf[pos >> 3] | (1 << (7 - (pos & 7)))
                pos += 1
        else:                                             # escape: raw value
            for _ in range(ESCAPE):
                buf[pos >> 3] = buf[pos >> 3] | (1 << (7 - (pos & 7)))
                pos += 1
            for bit in range(RAW_BITS - 1, -1, -1):
                if (z >> bit) & 1:
                    buf[pos >> 3] = buf[pos >> 3] | (1 << (7 - (pos & 7)))
                pos += 1
        A += z
        N += 1
        if N >= 64:                                       # forget old stats
            A = (A + 1) >> 1
            N >>= 1
    return buf[: (pos + 7) >> 3]


@njit(cache=True)
def _rice_decode(buf, n):
    out = np.empty(n, np.int32)
    pos = 0
    A = 4
    N = 1
    for i in range(n):
        k = _pick_k(A, N)
        count = 0
        while count < ESCAPE and (buf[pos >> 3] >> (7 - (pos & 7))) & 1:
            count += 1
            pos += 1
        z = 0
        if count < ESCAPE:
            pos += 1                                      # skip the 0
            low = 0
            for _ in range(k):
                low = (low << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                pos += 1
            z = (count << k) | low
        else:
            for _ in range(RAW_BITS):
                z = (z << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                pos += 1
        out[i] = (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)
        A += z
        N += 1
        if N >= 64:
            A = (A + 1) >> 1
            N >>= 1
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def _chroma_shape(h, w, subsampled):
    return ((h + 1) // 2, (w + 1) // 2) if subsampled else (h, w)


def encode(rgb: np.ndarray, q_luma: int = 1, q_chroma: int = 1,
           subsample_chroma: bool = False) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array to FUL bytes."""
    if not (1 <= q_luma <= 255 and 1 <= q_chroma <= 255):
        raise ValueError("q values must be between 1 and 255")
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]

    y, co, cg = rgb_to_ycocg(rgb)
    if subsample_chroma:
        co, cg = subsample_2x2(co), subsample_2x2(cg)

    flags = FLAG_SUBSAMPLE if subsample_chroma else 0
    parts = [HEADER.pack(MAGIC, VERSION, w, h, q_luma, q_chroma, flags)]
    for plane, q, (lo, hi) in ((y, q_luma, Y_RANGE),
                               (co, q_chroma, C_RANGE),
                               (cg, q_chroma, C_RANGE)):
        residuals = _encode_plane(np.ascontiguousarray(plane, np.int32), q, lo, hi)
        payload = _rice_encode(residuals).tobytes()
        parts.append(struct.pack("<I", len(payload)))
        parts.append(payload)
    return b"".join(parts)


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL image")
    magic, version, w, h, q_luma, q_chroma, flags = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Unsupported FUL version {version}")
    return {"width": w, "height": h, "q_luma": q_luma, "q_chroma": q_chroma,
            "subsample_chroma": bool(flags & FLAG_SUBSAMPLE)}


def decode(data: bytes) -> np.ndarray:
    """Decode FUL bytes to an (H, W, 3) uint8 RGB array."""
    info = read_header(data)
    w, h = info["width"], info["height"]
    sub = info["subsample_chroma"]
    ch, cw = _chroma_shape(h, w, sub)

    pos = HEADER.size
    planes = []
    for (ph, pw), q, (lo, hi) in (((h, w), info["q_luma"], Y_RANGE),
                                  ((ch, cw), info["q_chroma"], C_RANGE),
                                  ((ch, cw), info["q_chroma"], C_RANGE)):
        (n,) = struct.unpack_from("<I", data, pos)
        pos += 4
        buf = np.frombuffer(data, np.uint8, n, pos)
        pos += n
        residuals = _rice_decode(buf, ph * pw)
        planes.append(_decode_plane(residuals, ph, pw, q, lo, hi))

    y, co, cg = planes
    if sub:
        co, cg = upsample_2x2(co, h, w), upsample_2x2(cg, h, w)
    return ycocg_to_rgb(y, co, cg)


# --------------------------------------------------------------------------
# Helpers for testing
# --------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def make_test_image(w: int = 768, h: int = 512, seed: int = 0) -> np.ndarray:
    """Synthetic image with smooth areas, hard edges, fine pattern and noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([255 * xx / (w - 1),
                    255 * yy / (h - 1),
                    128 + 100 * np.sin(xx / 40) * np.cos(yy / 30)], axis=-1)
    disc = (xx - w * 0.7) ** 2 + (yy - h * 0.4) ** 2 < (h * 0.22) ** 2
    img[disc] = (230, 40, 60)
    region = (xx < w * 0.3) & (yy > h * 0.55)
    img[region] = 20
    img[region & ((xx // 8 + yy // 8) % 2 == 0)] = 240
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import time

    print(f"numba: {'yes' if HAVE_NUMBA else 'not installed (pure Python, slow)'}")
    img = make_test_image()
    decode(encode(img[:8, :8]))  # trigger JIT compilation once
    raw = img.size
    print(f"Test image {img.shape[1]}x{img.shape[0]}, raw {raw:,} bytes\n")
    print(f"{'settings':<26}{'bytes':>10}{'ratio':>8}{'PSNR':>10}{'enc ms':>9}{'dec ms':>9}")
    for ql, qc, sub in [(1, 1, False), (2, 2, False), (2, 4, True), (4, 8, True)]:
        t0 = time.perf_counter()
        data = encode(img, ql, qc, sub)
        t1 = time.perf_counter()
        out = decode(data)
        t2 = time.perf_counter()
        p = psnr(img, out)
        label = f"qY={ql} qC={qc} sub={'on' if sub else 'off'}"
        ptxt = "lossless" if p == float("inf") else f"{p:.2f}"
        print(f"{label:<26}{len(data):>10,}{raw / len(data):>7.2f}x{ptxt:>10}"
              f"{(t1 - t0) * 1e3:>9.1f}{(t2 - t1) * 1e3:>9.1f}")
        if ql == 1 and qc == 1 and not sub:
            assert np.array_equal(img, out), "Lossless round trip failed!"
    print("\nLossless round trip verified.")
