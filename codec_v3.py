"""
FUL v3: v2's wavelets + quadtree, with perceptual quantisation.

The encoder decides how coarsely to store every wavelet coefficient using
models of human vision, and writes its decisions into the file. The decoder
just reads the step sizes back and multiplies.

Perceptual models (all chosen by the encoder):
  * Contrast sensitivity: each wavelet band gets a visibility weight. Fine
    detail is less visible than broad shapes, fine colour detail far less
    visible than fine brightness detail, and diagonal detail (HH) less visible
    than horizontal/vertical detail (the "oblique effect").
  * Masking + Weber's law (optional, off by default): every 16x16 block gets
    a step multiplier. Busy texture hides errors, so it gets coarser steps;
    smooth areas get finer ones; very dark and very bright areas slightly
    coarser. SSIM measurements say this costs quality per byte, so it stays
    an experiment to judge by eye.
  * Rate-distortion decisions: a 4x4 group holding only one or two barely
    significant coefficients costs more bits than its visual contribution is
    worth, so the encoder drops it (minimising distortion + lambda * bits).

The band weights are stored in the header, so future encoders can retune them
without any change to the decoder.

Quality 100 is exactly lossless. Run for a self-test:  python codec_v3.py
"""
from __future__ import annotations

import struct

import numpy as np

from codec_v2 import (  # reuse v2's wavelet, quadtree and Rice machinery
    HAVE_NUMBA, TILE_SIZES, _fwd2d, _inv2d, _level_dims, _ll_decode,
    _ll_encode, _pool, _qt_decode, _qt_encode, _rice_get, _rice_put,
    _tile_grid, levels_for_tile, njit,
)

MAGIC = b"FUL\x00"
VERSION = 3
# magic, version, width, height, quality, flags, levels, tile size, base step
HEADER = struct.Struct("<4sBIIBBBHf")
FLAG_PERCEPTUAL = 1

BLOCK_SHIFT = 4                       # masking blocks are 16x16 pixels
MULTS = np.array([2.0 ** ((i - 2) / 4) for i in range(8)])   # 0.71 ... 2.38
DELTA = 0.4                           # reconstruction offset inside a step
RDO_LAMBDA = 0.1155                   # distortion per bit, in step^2 units
RDO_LEAF_BITS = 20.0                  # typical cost of a sparse 4x4 leaf


# --------------------------------------------------------------------------
# Encoder-side perceptual model
# --------------------------------------------------------------------------
def quality_to_base(quality: int) -> float:
    return 1.0 if quality >= 100 else float(2.0 ** ((100 - quality) / 10))


def perceptual_weights(levels: int) -> np.ndarray:
    """Visibility weights W[plane, level, orientation].
    level 1..levels are detail bands (1 = finest), levels+1 is the LL band.
    orientation 0 = horizontal/vertical detail, 1 = diagonal (HH).

    Tuned by measuring file size at equal SSIM on test photos: brightness
    bands stay flat (the level scaling already balances them), colour detail
    is stored far more coarsely, and diagonal detail gets 1.5x coarser steps."""
    w = np.ones((3, levels + 2, 2))
    chroma = [16.0, 5.0, 3.0, 2.0, 1.5]
    for j in range(1, levels + 1):
        w[1:, j, :] = chroma[j - 1]
        w[:, j, 1] *= 1.5                 # oblique effect
    w[1:, levels + 1, :] = 1.5
    return w


@njit(cache=True, nogil=True)
def _masking_map(y_plane):
    """One multiplier index per 16x16 block from local brightness and
    activity (mean absolute difference between neighbouring pixels)."""
    th, tw = y_plane.shape
    bh = (th + 15) >> BLOCK_SHIFT
    bw = (tw + 15) >> BLOCK_SHIFT
    out = np.empty((bh, bw), np.int32)
    for by in range(bh):
        for bx in range(bw):
            y0, x0 = by << BLOCK_SHIFT, bx << BLOCK_SHIFT
            y1, x1 = min(th, y0 + 16), min(tw, x0 + 16)
            total = 0.0
            act = 0.0
            n = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    v = y_plane[y, x]
                    total += v
                    if x + 1 < x1:
                        act += abs(v - y_plane[y, x + 1])
                    if y + 1 < y1:
                        act += abs(v - y_plane[y + 1, x])
                    n += 1
            mean = total / n
            a = act / (2 * n)
            idx = 2.0 + 2.0 * np.log2(max(a, 0.5) / 4.0)   # masking
            if mean < 20:                                   # Weber: darks
                idx += 2
            elif mean < 45:
                idx += 1
            elif mean > 235:                                # and highlights
                idx += 1
            idx = int(np.floor(idx + 0.5))
            out[by, bx] = min(7, max(0, idx))
    return out


# --------------------------------------------------------------------------
# Quantisation with per-coefficient steps
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _coef_step(s_base, mults, mp, py, px):
    by = min(mp.shape[0] - 1, py >> BLOCK_SHIFT)
    bx = min(mp.shape[1] - 1, px >> BLOCK_SHIFT)
    s = s_base * mults[mp[by, bx]]
    return s if s > 1.0 else 1.0


@njit(cache=True, nogil=True)
def _quant_band(a, y0, y1, x0, x1, shift, s_base, mults, mp, rdo):
    """Quantise one band in 4x4 groups (the quadtree's leaves), dropping
    groups whose few small coefficients aren't worth their bits."""
    for ly in range(y0, y1, 4):
        for lx in range(x0, x1, 4):
            ey, ex = min(y1, ly + 4), min(x1, lx + 4)
            score = 0.0
            nz = 0
            big = False
            for y in range(ly, ey):
                for x in range(lx, ex):
                    c = a[y, x]
                    s = _coef_step(s_base, mults, mp, (y - y0) << shift, (x - x0) << shift)
                    u = abs(c) / s
                    qv = int(u)
                    if qv >= 2:
                        big = True
                    elif qv == 1:
                        nz += 1
                        score += 3.0 * u - 2.25     # distortion added by zeroing
                    a[y, x] = qv if c >= 0 else -qv
            if rdo and not big and 0 < nz <= 2 and score < RDO_LAMBDA * RDO_LEAF_BITS:
                for y in range(ly, ey):
                    for x in range(lx, ex):
                        a[y, x] = 0


@njit(cache=True, nogil=True)
def _dequant_band(a, y0, y1, x0, x1, shift, s_base, mults, mp):
    for y in range(y0, y1):
        for x in range(x0, x1):
            qv = a[y, x]
            if qv != 0:
                s = _coef_step(s_base, mults, mp, (y - y0) << shift, (x - x0) << shift)
                v = int((abs(qv) + DELTA) * s + 0.5)
                a[y, x] = v if qv > 0 else -v


@njit(cache=True, nogil=True)
def _process(a, levels, base, wts, p, mults, mp, last_level, quantize, rdo):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels, last_level - 1, -1):
        h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
        s0 = base * wts[p, j, 0] / (1 << (j - 1))
        s1 = base * wts[p, j, 1] / (1 << (j - 1))
        if quantize:
            _quant_band(a, 0, hl, wl, w, j, s0, mults, mp, rdo)     # HL
            _quant_band(a, hl, h, 0, wl, j, s0, mults, mp, rdo)     # LH
            _quant_band(a, hl, h, wl, w, j, s1, mults, mp, rdo)     # HH
        else:
            _dequant_band(a, 0, hl, wl, w, j, s0, mults, mp)
            _dequant_band(a, hl, h, 0, wl, j, s0, mults, mp)
            _dequant_band(a, hl, h, wl, w, j, s1, mults, mp)
    s_ll = base * wts[p, levels + 1, 0] / (1 << levels)
    if quantize:
        _quant_band(a, 0, hs[levels], 0, ws[levels], levels, s_ll, mults, mp, False)
    else:
        _dequant_band(a, 0, hs[levels], 0, ws[levels], levels, s_ll, mults, mp)


# --------------------------------------------------------------------------
# Tile encode / decode
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _encode_tile(planes, levels, base, wts, mults, perceptual, rdo, masking):
    _, th, tw = planes.shape
    n = th * tw
    buf = np.zeros(3 * (n * 5 + n // 4 + 64) + 4096, np.uint8)
    spans = np.zeros((3, levels + 1, 2), np.int64)
    stack = np.empty((64, 3), np.int64)
    if perceptual and masking:
        mp = _masking_map(planes[0])
    elif perceptual:
        mp = np.full(((th + 15) >> BLOCK_SHIFT, (tw + 15) >> BLOCK_SHIFT), 2, np.int32)
    else:
        mp = np.full((1, 1), 2, np.int32)                 # multiplier 1.0
    hs, ws = _level_dims(th, tw, levels)
    pos = 0
    for p in range(3):
        a = planes[p].copy()
        _fwd2d(a, levels)
        _process(a, levels, base, wts, p, mults, mp, 1, True, rdo)
        for layer in range(levels + 1):
            start = pos >> 3
            A, N = 4, 1
            if layer == 0:
                if p == 0 and perceptual:                 # the masking map
                    prev = 2
                    for by in range(mp.shape[0]):
                        for bx in range(mp.shape[1]):
                            pos, A, N = _rice_put(buf, pos, mp[by, bx] - prev, A, N)
                            prev = mp[by, bx]
                    A, N = 4, 1
                pos, A, N = _ll_encode(a, hs[levels], ws[levels], buf, pos, A, N)
            else:
                j = levels + 1 - layer
                h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
                pos, A, N = _qt_encode(a, 0, wl, hl, w - wl, buf, pos, A, N, stack)
                pos, A, N = _qt_encode(a, hl, 0, h - hl, wl, buf, pos, A, N, stack)
                pos, A, N = _qt_encode(a, hl, wl, h - hl, w - wl, buf, pos, A, N, stack)
            pos = (pos + 7) & ~7
            spans[p, layer, 0] = start
            spans[p, layer, 1] = pos >> 3
    return buf, spans


@njit(cache=True, nogil=True)
def _decode_tile(data, offs, th, tw, levels, nlayers, base, wts, mults,
                 perceptual, out, oy, ox):
    stop = levels - (nlayers - 1)
    planes = np.zeros((3, th, tw), np.int32)
    stack = np.empty((64, 3), np.int64)
    hs, ws = _level_dims(th, tw, levels)
    if perceptual:
        mp = np.zeros(((th + 15) >> BLOCK_SHIFT, (tw + 15) >> BLOCK_SHIFT), np.int32)
    else:
        mp = np.full((1, 1), 2, np.int32)
    for p in range(3):
        a = planes[p]
        for layer in range(nlayers):
            start = offs[p, layer, 0]
            length = offs[p, layer, 1]
            if length == 0:
                continue
            buf = data[start:start + length]
            pos = 0
            A, N = 4, 1
            if layer == 0:
                if p == 0 and perceptual:
                    prev = 2
                    for by in range(mp.shape[0]):
                        for bx in range(mp.shape[1]):
                            v, pos, A, N = _rice_get(buf, pos, A, N)
                            prev = min(7, max(0, prev + v))
                            mp[by, bx] = prev
                    A, N = 4, 1
                _ll_decode(a, hs[levels], ws[levels], buf, pos, A, N)
            else:
                j = levels + 1 - layer
                h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
                pos, A, N = _qt_decode(a, 0, wl, hl, w - wl, buf, pos, A, N, stack)
                pos, A, N = _qt_decode(a, hl, 0, h - hl, wl, buf, pos, A, N, stack)
                pos, A, N = _qt_decode(a, hl, wl, h - hl, w - wl, buf, pos, A, N, stack)
        _process(a, levels, base, wts, p, mults, mp, stop + 1, False, False)
        _inv2d(a, levels, stop)
    h, w = hs[stop], ws[stop]
    for y in range(h):
        for x in range(w):
            Y = planes[0, y, x]
            co = planes[1, y, x]
            cg = planes[2, y, x]
            t = Y - (cg >> 1)
            g = cg + t
            b = t - (co >> 1)
            r = b + co
            out[oy + y, ox + x, 0] = min(255, max(0, r))
            out[oy + y, ox + x, 1] = min(255, max(0, g))
            out[oy + y, ox + x, 2] = min(255, max(0, b))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def encode(rgb: np.ndarray, quality: int = 75, perceptual: bool = True,
           tile_size: int = 256, rdo: bool = True, masking: bool = False,
           weights: np.ndarray | None = None) -> bytes:
    """Quality 1-100. 100 is exactly lossless (perceptual tuning is off)."""
    if tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {TILE_SIZES}")
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    lossless = quality == 100
    perceptual = perceptual and not lossless
    rdo = rdo and not lossless

    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    planes = np.stack([y, co, cg])

    levels = levels_for_tile(tile_size)
    base = quality_to_base(quality)
    if weights is not None and perceptual:
        wts = np.asarray(weights, dtype=np.float64)
    else:
        wts = perceptual_weights(levels) if perceptual else np.ones((3, levels + 2, 2))
    wts = wts.astype(np.float32).astype(np.float64)   # exactly what the file stores
    tiles = _tile_grid(w, h, tile_size)

    def work(tile):
        ty, tx = tile
        sub = np.ascontiguousarray(planes[:, ty * tile_size:(ty + 1) * tile_size,
                                          tx * tile_size:(tx + 1) * tile_size])
        return _encode_tile(sub, levels, base, wts, MULTS, perceptual, rdo, masking)

    results = list(_pool().map(work, tiles))
    lengths, payload = [], []
    for layer in range(levels + 1):
        for buf, spans in results:
            for p in range(3):
                s, e = spans[p, layer]
                lengths.append(e - s)
                payload.append(buf[s:e].tobytes())
    flags = FLAG_PERCEPTUAL if perceptual else 0
    header = HEADER.pack(MAGIC, VERSION, w, h, quality, flags, levels, tile_size, base)
    return (header + wts.astype("<f4").tobytes()
            + np.asarray(lengths, dtype="<u4").tobytes() + b"".join(payload))


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL v3 image")
    magic, version, w, h, quality, flags, levels, tile, base = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Not a FUL v3 file (version {version})")
    if tile not in TILE_SIZES or levels != levels_for_tile(tile):
        raise ValueError("Corrupt FUL v3 header")
    return {"version": 3, "width": w, "height": h, "quality": quality,
            "perceptual": bool(flags & FLAG_PERCEPTUAL), "levels": levels,
            "tile_size": tile, "base": base}


def _layout(data: bytes, info: dict):
    levels = info["levels"]
    nw = 3 * (levels + 2) * 2
    wts = np.frombuffer(data, "<f4", nw, HEADER.size).astype(np.float64)
    wts = wts.reshape(3, levels + 2, 2)
    tiles = _tile_grid(info["width"], info["height"], info["tile_size"])
    nchunks = (levels + 1) * len(tiles) * 3
    table_at = HEADER.size + 4 * nw
    base = table_at + 4 * nchunks
    if len(data) < base:
        raise ValueError("FUL v3 file is truncated")
    lengths = np.frombuffer(data, "<u4", nchunks, table_at).astype(np.int64)
    starts = base + np.concatenate([[0], np.cumsum(lengths)[:-1]])
    shape = (levels + 1, len(tiles), 3)
    return wts, tiles, starts.reshape(shape), lengths.reshape(shape), base


def layer_count(data: bytes) -> int:
    return read_header(data)["levels"] + 1


def bytes_for_layers(data: bytes, layers: int) -> int:
    info = read_header(data)
    _, _, _, lengths, base = _layout(data, info)
    return int(base + lengths[:layers].sum())


def decode(data: bytes, layers: int | None = None) -> np.ndarray:
    info = read_header(data)
    levels, tile = info["levels"], info["tile_size"]
    w, h = info["width"], info["height"]
    k = levels + 1 if layers is None else max(1, min(levels + 1, int(layers)))
    wts, tiles, starts, lengths, _ = _layout(data, info)
    if (starts[:k] + lengths[:k]).max() > len(data):
        raise ValueError("FUL v3 file is truncated")
    m = levels - (k - 1)
    scale = 1 << m
    out = np.empty(((h + scale - 1) // scale, (w + scale - 1) // scale, 3), np.uint8)
    buf = np.frombuffer(data, np.uint8)
    tile_s = tile >> m
    base = float(info["base"])

    def work(i):
        ty, tx = tiles[i]
        th = min(tile, h - ty * tile)
        tw = min(tile, w - tx * tile)
        offs = np.empty((3, k, 2), np.int64)
        offs[:, :, 0] = starts[:k, i, :].T
        offs[:, :, 1] = lengths[:k, i, :].T
        _decode_tile(buf, offs, th, tw, levels, k, base, wts, MULTS,
                     info["perceptual"], out, ty * tile_s, tx * tile_s)

    list(_pool().map(work, range(len(tiles))))
    return out


if __name__ == "__main__":
    import codec_v1

    rng = np.random.default_rng(5)
    for shape in [(1, 1, 3), (5, 70, 3), (37, 53, 3), (300, 260, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        for tile in TILE_SIZES:
            assert np.array_equal(decode(encode(img, 100, tile_size=tile)), img), (shape, tile)
            data = encode(img, 60, True, tile)
            for k in range(1, layer_count(data) + 1):
                decode(data, k)
    print("Lossless round trips and progressive decodes verified.")
    img = codec_v1.make_test_image()
    for q in (95, 85, 70, 50):
        for perc in (False, True):
            data = encode(img, q, perc)
            out = decode(data)
            print(f"quality {q:3d} perceptual {'on ' if perc else 'off'}: "
                  f"{len(data):>8,} bytes, PSNR {codec_v1.psnr(img, out):.2f} dB")
