"""
FUL v2: integer wavelets + quadtree zero coding.

Encoder pipeline, per tile and per colour plane:
    RGB -> YCoCg-R -> LeGall 5/3 integer wavelet (several levels)
        -> dead-zone quantization -> coding:
             * coarsest LL band: MED prediction + adaptive Rice codes
             * detail bands:     quadtree marks all-zero regions with one bit,
                                 nonzero 4x4 leaves get adaptive Rice codes

Why this is faster, smaller and lighter than v1:
    * Tiles are independent, so they encode and decode on all CPU cores.
    * The decoder writes RGB straight into the output; working memory is one
      tile per thread instead of three full-size integer planes.
    * The file is ordered by resolution layer. Reading only the first part of
      the file gives a complete smaller image, so previews and thumbnails are
      nearly free and a viewer can show something almost instantly.
    * Flat regions turn into all-zero detail coefficients, which the quadtree
      skips in large blocks.

With both steps = 1 the round trip is exactly lossless.
Run this file directly for a self-test:  python codec_v2.py
"""
from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # Plain Python fallback: correct but slow and single-core.
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

MAGIC = b"FUL\x00"
VERSION = 2
# magic, version, width, height, q_luma, q_chroma, flags, levels, tile size
HEADER = struct.Struct("<4sBIIBBBBH")
FLAG_SUBSAMPLE = 1

TILE_SIZES = (64, 128, 256, 512, 1024)
MAX_LEVELS = 5
LEAF = 4          # quadtree stops at 4x4 blocks
ESCAPE = 24       # unary run length at which a raw value follows
RAW_BITS = 20
MAX_K = 15

_executor: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
    return _executor


def levels_for_tile(tile: int) -> int:
    return min(MAX_LEVELS, tile.bit_length() - 1 - 2)


# --------------------------------------------------------------------------
# Bit I/O and adaptive Rice codes
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _put_bit(buf, pos, bit):
    if bit:
        buf[pos >> 3] = buf[pos >> 3] | (0x80 >> (pos & 7))
    return pos + 1


@njit(cache=True, nogil=True)
def _get_bit(buf, pos):
    i = pos >> 3
    if i >= buf.size:          # reading past the end yields zeros: a corrupt
        return 0               # file decodes to garbage instead of crashing
    return (buf[i] >> (7 - (pos & 7))) & 1


@njit(cache=True, nogil=True)
def _pick_k(A, N):
    k = 0
    while (N << k) < A and k < MAX_K:
        k += 1
    return k


@njit(cache=True, nogil=True)
def _rice_put(buf, pos, v, A, N):
    z = (v << 1) if v >= 0 else ((-v) << 1) - 1
    k = _pick_k(A, N)
    hi = z >> k
    if hi < ESCAPE:
        for _ in range(hi):
            pos = _put_bit(buf, pos, 1)
        pos += 1
        for b in range(k - 1, -1, -1):
            pos = _put_bit(buf, pos, (z >> b) & 1)
    else:
        for _ in range(ESCAPE):
            pos = _put_bit(buf, pos, 1)
        for b in range(RAW_BITS - 1, -1, -1):
            pos = _put_bit(buf, pos, (z >> b) & 1)
    A += z
    N += 1
    if N >= 64:
        A = (A + 1) >> 1
        N >>= 1
    return pos, A, N


@njit(cache=True, nogil=True)
def _rice_get(buf, pos, A, N):
    k = _pick_k(A, N)
    count = 0
    while count < ESCAPE and _get_bit(buf, pos):
        count += 1
        pos += 1
    z = 0
    if count < ESCAPE:
        pos += 1
        low = 0
        for _ in range(k):
            low = (low << 1) | _get_bit(buf, pos)
            pos += 1
        z = (count << k) | low
    else:
        for _ in range(RAW_BITS):
            z = (z << 1) | _get_bit(buf, pos)
            pos += 1
    v = (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)
    A += z
    N += 1
    if N >= 64:
        A = (A + 1) >> 1
        N >>= 1
    return v, pos, A, N


# --------------------------------------------------------------------------
# LeGall 5/3 integer wavelet (the reversible JPEG 2000 filter)
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _fwd53(x, n, out):
    """Lifting: predict odd samples from even ones, then update the evens.
    Output holds the low band first, then the high band."""
    if n <= 1:
        out[0] = x[0]
        return
    nh = n // 2
    nl = n - nh
    for i in range(nh):
        right = x[2 * i + 2] if 2 * i + 2 < n else x[2 * i]
        out[nl + i] = x[2 * i + 1] - ((x[2 * i] + right) >> 1)
    for i in range(nl):
        dl = out[nl + i - 1] if i > 0 else out[nl]
        dr = out[nl + i] if i < nh else out[nl + nh - 1]
        out[i] = x[2 * i] + ((dl + dr + 2) >> 2)


@njit(cache=True, nogil=True)
def _inv53(y, n, out):
    if n <= 1:
        out[0] = y[0]
        return
    nh = n // 2
    nl = n - nh
    for i in range(nl):
        dl = y[nl + i - 1] if i > 0 else y[nl]
        dr = y[nl + i] if i < nh else y[nl + nh - 1]
        out[2 * i] = y[i] - ((dl + dr + 2) >> 2)
    for i in range(nh):
        right = out[2 * i + 2] if 2 * i + 2 < n else out[2 * i]
        out[2 * i + 1] = y[nl + i] + ((out[2 * i] + right) >> 1)


@njit(cache=True, nogil=True)
def _level_dims(h, w, levels):
    hs = np.empty(levels + 1, np.int64)
    ws = np.empty(levels + 1, np.int64)
    hs[0] = h
    ws[0] = w
    for j in range(levels):
        hs[j + 1] = (hs[j] + 1) // 2
        ws[j + 1] = (ws[j] + 1) // 2
    return hs, ws


@njit(cache=True, nogil=True)
def _fwd2d(a, levels):
    h, w = a.shape
    m = max(h, w)
    tin = np.empty(m, np.int32)
    tout = np.empty(m, np.int32)
    for _ in range(levels):
        for y in range(h):
            for x in range(w):
                tin[x] = a[y, x]
            _fwd53(tin, w, tout)
            for x in range(w):
                a[y, x] = tout[x]
        for x in range(w):
            for y in range(h):
                tin[y] = a[y, x]
            _fwd53(tin, h, tout)
            for y in range(h):
                a[y, x] = tout[y]
        h = (h + 1) // 2
        w = (w + 1) // 2


@njit(cache=True, nogil=True)
def _inv2d(a, levels, stop):
    """Undo levels down to `stop`. stop = 0 gives full size; stop = m leaves a
    complete image at 1 / 2^m scale in the top-left corner."""
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    m = max(a.shape[0], a.shape[1])
    tin = np.empty(m, np.int32)
    tout = np.empty(m, np.int32)
    for j in range(levels - 1, stop - 1, -1):
        h = hs[j]
        w = ws[j]
        for x in range(w):
            for y in range(h):
                tin[y] = a[y, x]
            _inv53(tin, h, tout)
            for y in range(h):
                a[y, x] = tout[y]
        for y in range(h):
            for x in range(w):
                tin[x] = a[y, x]
            _inv53(tin, w, tout)
            for x in range(w):
                a[y, x] = tout[x]


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _step(q, level):
    """Coarser levels affect bigger areas, so they get finer steps."""
    s = q >> (level - 1)
    return s if s > 0 else 1


@njit(cache=True, nogil=True)
def _quant(a, y0, y1, x0, x1, step):
    if step == 1:
        return
    for y in range(y0, y1):
        for x in range(x0, x1):
            c = a[y, x]
            a[y, x] = c // step if c >= 0 else -((-c) // step)


@njit(cache=True, nogil=True)
def _dequant(a, y0, y1, x0, x1, step):
    if step == 1:
        return
    half = step // 2
    for y in range(y0, y1):
        for x in range(x0, x1):
            c = a[y, x]
            if c > 0:
                a[y, x] = c * step + half
            elif c < 0:
                a[y, x] = c * step - half


@njit(cache=True, nogil=True)
def _process_bands(a, levels, q, last_level, quantize):
    """Quantize or dequantize every band up to `last_level` (counted from the
    finest = 1). The coarsest LL band uses the step for level levels + 1."""
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels, last_level - 1, -1):
        h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
        s = _step(q, j)
        if quantize:
            _quant(a, 0, hl, wl, w, s)
            _quant(a, hl, h, 0, w, s)
        else:
            _dequant(a, 0, hl, wl, w, s)
            _dequant(a, hl, h, 0, w, s)
    s = _step(q, levels + 1)
    if quantize:
        _quant(a, 0, hs[levels], 0, ws[levels], s)
    else:
        _dequant(a, 0, hs[levels], 0, ws[levels], s)


# --------------------------------------------------------------------------
# Band coding
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _med(a, b, c):
    if c >= max(a, b):
        return min(a, b)
    if c <= min(a, b):
        return max(a, b)
    return a + b - c


@njit(cache=True, nogil=True)
def _ll_pred(a, y, x):
    if y == 0:
        return int(a[0, x - 1]) if x > 0 else 0
    if x == 0:
        return int(a[y - 1, 0])
    return _med(int(a[y, x - 1]), int(a[y - 1, x]), int(a[y - 1, x - 1]))


@njit(cache=True, nogil=True)
def _ll_encode(a, h, w, buf, pos, A, N):
    for y in range(h):
        for x in range(w):
            pos, A, N = _rice_put(buf, pos, int(a[y, x]) - _ll_pred(a, y, x), A, N)
    return pos, A, N


@njit(cache=True, nogil=True)
def _ll_decode(a, h, w, buf, pos, A, N):
    for y in range(h):
        for x in range(w):
            v, pos, A, N = _rice_get(buf, pos, A, N)
            a[y, x] = _ll_pred(a, y, x) + v
    return pos, A, N


@njit(cache=True, nogil=True)
def _qt_root(bh, bw, stack):
    size = LEAF
    while size < bh or size < bw:
        size *= 2
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack[0, 2] = size
    return 1


@njit(cache=True, nogil=True)
def _qt_push_children(stack, sp, y, x, s):
    half = s // 2                    # push in reverse: top-left comes out first
    for cy, cx in ((half, half), (half, 0), (0, half), (0, 0)):
        stack[sp, 0] = y + cy
        stack[sp, 1] = x + cx
        stack[sp, 2] = half
        sp += 1
    return sp


@njit(cache=True, nogil=True)
def _qt_encode(a, y0, x0, bh, bw, buf, pos, A, N, stack):
    """Quadtree over one band. Each node sends one bit: is anything inside
    nonzero? Zero nodes end there, so empty regions cost a single bit."""
    if bh <= 0 or bw <= 0:
        return pos, A, N
    sp = _qt_root(bh, bw, stack)
    while sp > 0:
        sp -= 1
        y, x, s = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        h = min(s, bh - y)
        w = min(s, bw - x)
        if h <= 0 or w <= 0:
            continue                 # outside the band: both sides know, no bit
        nz = 0
        for yy in range(y0 + y, y0 + y + h):
            for xx in range(x0 + x, x0 + x + w):
                if a[yy, xx] != 0:
                    nz = 1
                    break
            if nz:
                break
        pos = _put_bit(buf, pos, nz)
        if nz == 0:
            continue
        if s <= LEAF:
            for yy in range(y0 + y, y0 + y + h):
                for xx in range(x0 + x, x0 + x + w):
                    pos, A, N = _rice_put(buf, pos, int(a[yy, xx]), A, N)
        else:
            sp = _qt_push_children(stack, sp, y, x, s)
    return pos, A, N


@njit(cache=True, nogil=True)
def _qt_decode(a, y0, x0, bh, bw, buf, pos, A, N, stack):
    if bh <= 0 or bw <= 0:
        return pos, A, N
    sp = _qt_root(bh, bw, stack)
    while sp > 0:
        sp -= 1
        y, x, s = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        h = min(s, bh - y)
        w = min(s, bw - x)
        if h <= 0 or w <= 0:
            continue
        nz = _get_bit(buf, pos)
        pos += 1
        if nz == 0:
            continue                 # whole block is zero: nothing to do
        if s <= LEAF:
            for yy in range(y0 + y, y0 + y + h):
                for xx in range(x0 + x, x0 + x + w):
                    v, pos, A, N = _rice_get(buf, pos, A, N)
                    a[yy, xx] = v
        else:
            sp = _qt_push_children(stack, sp, y, x, s)
    return pos, A, N


@njit(cache=True, nogil=True)
def _layer_encode(a, levels, layer, buf, pos, stack):
    """Layer 0 is the coarsest LL band; layer k adds detail level levels+1-k."""
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    A, N = 4, 1
    if layer == 0:
        pos, A, N = _ll_encode(a, hs[levels], ws[levels], buf, pos, A, N)
        return pos
    j = levels + 1 - layer
    h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
    pos, A, N = _qt_encode(a, 0, wl, hl, w - wl, buf, pos, A, N, stack)       # HL
    pos, A, N = _qt_encode(a, hl, 0, h - hl, wl, buf, pos, A, N, stack)       # LH
    pos, A, N = _qt_encode(a, hl, wl, h - hl, w - wl, buf, pos, A, N, stack)  # HH
    return pos


@njit(cache=True, nogil=True)
def _layer_decode(a, levels, layer, buf, stack):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    A, N = 4, 1
    pos = 0
    if layer == 0:
        _ll_decode(a, hs[levels], ws[levels], buf, pos, A, N)
        return
    j = levels + 1 - layer
    h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
    pos, A, N = _qt_decode(a, 0, wl, hl, w - wl, buf, pos, A, N, stack)
    pos, A, N = _qt_decode(a, hl, 0, h - hl, wl, buf, pos, A, N, stack)
    pos, A, N = _qt_decode(a, hl, wl, h - hl, w - wl, buf, pos, A, N, stack)


# --------------------------------------------------------------------------
# Tile encode / decode (each call is independent and releases the GIL)
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _encode_tile(planes, levels, q_luma, q_chroma, subsample):
    _, th, tw = planes.shape
    n = th * tw
    buf = np.zeros(3 * (n * 5 + n // 4 + 64), np.uint8)
    spans = np.zeros((3, levels + 1, 2), np.int64)
    stack = np.empty((64, 3), np.int64)
    pos = 0
    for p in range(3):
        a = planes[p].copy()
        _fwd2d(a, levels)
        q = q_luma if p == 0 else q_chroma
        _process_bands(a, levels, q, 1, True)
        for layer in range(levels + 1):
            start = pos >> 3
            skip = subsample and p > 0 and layer == levels  # finest chroma
            if not skip:
                pos = _layer_encode(a, levels, layer, buf, pos, stack)
                pos = (pos + 7) & ~7                         # byte-align
            spans[p, layer, 0] = start
            spans[p, layer, 1] = pos >> 3
    return buf, spans


@njit(cache=True, nogil=True)
def _decode_tile(data, offs, th, tw, levels, nlayers, q_luma, q_chroma,
                 out, oy, ox):
    stop = levels - (nlayers - 1)           # how many levels stay undone
    last_level = stop + 1
    planes = np.zeros((3, th, tw), np.int32)
    stack = np.empty((64, 3), np.int64)
    for p in range(3):
        a = planes[p]
        for layer in range(nlayers):
            start = offs[p, layer, 0]
            length = offs[p, layer, 1]
            if length > 0:
                _layer_decode(a, levels, layer, data[start:start + length], stack)
        q = q_luma if p == 0 else q_chroma
        _process_bands(a, levels, q, last_level, False)
        _inv2d(a, levels, stop)
    hs, ws = _level_dims(th, tw, levels)
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
def _tile_grid(w, h, tile):
    return [(ty, tx) for ty in range((h + tile - 1) // tile)
            for tx in range((w + tile - 1) // tile)]


def encode(rgb: np.ndarray, q_luma: int = 1, q_chroma: int = 1,
           subsample_chroma: bool = False, tile_size: int = 256) -> bytes:
    if tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {TILE_SIZES}")
    if not (1 <= q_luma <= 255 and 1 <= q_chroma <= 255):
        raise ValueError("q values must be between 1 and 255")
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
    tiles = _tile_grid(w, h, tile_size)

    def work(tile):
        ty, tx = tile
        sub = np.ascontiguousarray(planes[:, ty * tile_size:(ty + 1) * tile_size,
                                          tx * tile_size:(tx + 1) * tile_size])
        return _encode_tile(sub, levels, q_luma, q_chroma, subsample_chroma)

    results = list(_pool().map(work, tiles))

    # Lay chunks out layer by layer so a file prefix is a smaller image.
    lengths, payload = [], []
    for layer in range(levels + 1):
        for buf, spans in results:
            for p in range(3):
                s, e = spans[p, layer]
                lengths.append(e - s)
                payload.append(buf[s:e].tobytes())
    flags = FLAG_SUBSAMPLE if subsample_chroma else 0
    header = HEADER.pack(MAGIC, VERSION, w, h, q_luma, q_chroma, flags,
                         levels, tile_size)
    table = np.asarray(lengths, dtype="<u4").tobytes()
    return header + table + b"".join(payload)


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL v2 image")
    magic, version, w, h, ql, qc, flags, levels, tile = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Not a FUL v2 file (version {version})")
    if tile not in TILE_SIZES or levels != levels_for_tile(tile):
        raise ValueError("Corrupt FUL v2 header")
    return {"version": 2, "width": w, "height": h, "q_luma": ql,
            "q_chroma": qc, "subsample_chroma": bool(flags & FLAG_SUBSAMPLE),
            "levels": levels, "tile_size": tile}


def _layout(data: bytes, info: dict):
    tiles = _tile_grid(info["width"], info["height"], info["tile_size"])
    nchunks = (info["levels"] + 1) * len(tiles) * 3
    base = HEADER.size + 4 * nchunks
    if len(data) < base:
        raise ValueError("FUL v2 file is truncated")
    lengths = np.frombuffer(data, "<u4", nchunks, HEADER.size).astype(np.int64)
    starts = base + np.concatenate([[0], np.cumsum(lengths)[:-1]])
    shape = (info["levels"] + 1, len(tiles), 3)
    return tiles, starts.reshape(shape), lengths.reshape(shape), base


def layer_count(data: bytes) -> int:
    return read_header(data)["levels"] + 1


def bytes_for_layers(data: bytes, layers: int) -> int:
    """How much of the file a viewer must read to show `layers` layers."""
    info = read_header(data)
    _, _, lengths, base = _layout(data, info)
    return int(base + lengths[:layers].sum())


def decode(data: bytes, layers: int | None = None) -> np.ndarray:
    """Decode to (H, W, 3) uint8. With fewer layers than the file holds, you
    get the whole picture at 1/2, 1/4, ... size, reading only a file prefix."""
    info = read_header(data)
    levels, tile = info["levels"], info["tile_size"]
    w, h = info["width"], info["height"]
    k = levels + 1 if layers is None else max(1, min(levels + 1, int(layers)))
    tiles, starts, lengths, _ = _layout(data, info)
    if (starts[:k] + lengths[:k]).max() > len(data):
        raise ValueError("FUL v2 file is truncated")

    m = levels - (k - 1)
    scale = 1 << m
    out = np.empty(((h + scale - 1) // scale, (w + scale - 1) // scale, 3), np.uint8)
    buf = np.frombuffer(data, np.uint8)
    tile_s = tile >> m

    def work(i):
        ty, tx = tiles[i]
        th = min(tile, h - ty * tile)
        tw = min(tile, w - tx * tile)
        offs = np.empty((3, k, 2), np.int64)
        offs[:, :, 0] = starts[:k, i, :].T
        offs[:, :, 1] = lengths[:k, i, :].T
        _decode_tile(buf, offs, th, tw, levels, k, info["q_luma"],
                     info["q_chroma"], out, ty * tile_s, tx * tile_s)

    list(_pool().map(work, range(len(tiles))))
    return out


if __name__ == "__main__":
    import time

    import codec_v1

    print(f"numba: {'yes' if HAVE_NUMBA else 'not installed (slow)'}, "
          f"threads: {os.cpu_count()}")
    rng = np.random.default_rng(3)
    # Correctness: lossless on awkward sizes and every tile size
    for shape in [(1, 1, 3), (1, 9, 3), (7, 1, 3), (37, 53, 3), (300, 260, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        for tile in TILE_SIZES:
            assert np.array_equal(decode(encode(img, tile_size=tile)), img), (shape, tile)
            data = encode(img, 5, 9, True, tile)
            for k in range(1, layer_count(data) + 1):
                decode(data, k)
    print("Lossless round trips and progressive decodes verified.\n")

    img = codec_v1.make_test_image()
    raw = img.size
    print(f"{'codec / settings':<34}{'bytes':>10}{'ratio':>8}{'PSNR':>10}"
          f"{'enc ms':>9}{'dec ms':>9}")
    for ql, qc, sub in [(1, 1, False), (2, 4, True), (8, 16, True)]:
        for name, enc, dec in (("v1", lambda: codec_v1.encode(img, ql, qc, sub), codec_v1.decode),
                               ("v2", lambda: encode(img, ql, qc, sub), decode)):
            t0 = time.perf_counter()
            data = enc()
            t1 = time.perf_counter()
            out = dec(data)
            t2 = time.perf_counter()
            p = codec_v1.psnr(img, out)
            ptxt = "lossless" if p == float("inf") else f"{p:.2f}"
            label = f"{name} qY={ql} qC={qc} sub={'on' if sub else 'off'}"
            print(f"{label:<34}{len(data):>10,}{raw / len(data):>7.2f}x{ptxt:>10}"
                  f"{(t1 - t0) * 1e3:>9.1f}{(t2 - t1) * 1e3:>9.1f}")
    data = encode(img, 2, 4, True)
    print("\nProgressive layers (v2, qY=2 qC=4 sub=on):")
    for k in range(1, layer_count(data) + 1):
        t0 = time.perf_counter()
        prev = decode(data, k)
        ms = (time.perf_counter() - t0) * 1e3
        need = bytes_for_layers(data, k)
        print(f"  layer {k}: {prev.shape[1]:>4}x{prev.shape[0]:<4} "
              f"needs {need / len(data) * 100:5.1f}% of file, {ms:6.1f} ms")
