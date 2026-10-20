"""
FUL v5: whole-image integer wavelets with independent code-blocks.

What changed from v4, and why (measured on test photos):
  * The wavelet runs over the WHOLE image. v4's 256-pixel tiles broke the
    transform at every tile edge and restarted the probability models in
    every tile.
  * Lossy files use the CDF 9/7 wavelet (JPEG 2000's lossy filter) instead of
    the 5/3. Its longer filters pack smooth detail into fewer coefficients.
    It runs as integer lifting (multiply by a 12-bit constant, shift), so CPU
    and GPU decoders produce identical results. The filter's scaling factors
    are folded into the step sizes stored in the header, and the image is
    pre-scaled by 2^prescale for extra precision.
  * Lossless files (quality 100) keep the 5/3, which is exactly reversible
    with no pre-scaling.
  * Entropy coding is split into code-blocks: horizontal strips of each
    wavelet band holding at least 64K coefficients. Blocks within a
    resolution layer are independent, so they decode on all cores, and are
    big enough that restarting the adaptive models costs little.
  * The transform runs with 4 extra bits of internal precision (beyond the
    file's 1/4-level coefficient units): rounding inside quantised integer
    lifting otherwise shows up as faint structure that SSIM notices.
  * The transform itself runs in parallel over rows and columns.

The entropy coder, perceptual model, denoising and grain are v4's.
Run for a self-test:  python codec_v5.py
"""
from __future__ import annotations

import struct

import numpy as np

from codec_v2 import HAVE_NUMBA, _ll_pred, _pool, njit
from codec_v3 import BLOCK_SHIFT, MULTS, _masking_map, quality_to_base
from codec_v4 import (
    INIT_CDF, LEAF, M_LL, M_MAP, _band_geometry, _dequantise, _emit,
    _emit_value, _get, _get_value, _grain_sigmas, _hash32, _level_dims,
    _ll_ctx, _models_init, _node_model, _quantise, _rans_encode, _rans_start,
    _raw_get, _raw_put, _shrink_thresholds, auto_denoise_strength, estimate_noise,
    fast, M_MAG,
)

try:
    from numba import prange
except ImportError:
    prange = range

MAGIC = b"FUL\x00"
VERSION = 5
# magic, version, width, height, quality, flags, levels, precision
# (low 4 bits: prescale, high 4 bits: extra), base step
HEADER = struct.Struct("<4sBIIBBBBf")
FLAG_PERCEPTUAL = 1
FLAG_GRAIN = 2
FLAG_97 = 4

MAX_LEVELS = 6
PRESCALE_97 = 2            # coefficient units in the file: 1/4 level
EXTRA_97 = 4               # extra internal precision inside the transform
GRAIN_CELL = 256
BLOCK_MIN = 65536          # minimum coefficients per code-block

LIFT_SHIFT = 10            # keeps GPU 32-bit products in range
_A, _B, _C, _D = -1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971
K97 = 1.230174104914001
# lifting steps: (applies to odd samples?, sign, multiplier, rounding, shift)
STEPS_53 = np.array([[1, -1, 1, 0, 1], [0, 1, 1, 2, 2]], np.int64)
STEPS_97 = np.array([[1, 1, int(round(c * (1 << LIFT_SHIFT))), 1 << (LIFT_SHIFT - 1), LIFT_SHIFT]
                     for c in (_A, _B, _C, _D)], np.int64)
STEPS_97[1, 0] = 0
STEPS_97[3, 0] = 0


CHROMA_WEIGHTS = [16.0, 5.0, 3.0, 2.0, 1.5, 1.5]
LUMA_FINEST = 1.25
DIAGONAL = 2.0


def perceptual_weights(levels: int) -> np.ndarray:
    """Visibility weights W[plane, level, orientation] (level 1 = finest,
    levels + 1 = LL; orientation 1 = diagonal). Fine colour detail is stored
    far more coarsely, diagonal detail somewhat more coarsely. Retuned for
    the 9/7 wavelet by measuring file size at equal SSIM against JPEG."""
    w = np.ones((3, levels + 2, 2))
    if levels >= 1:
        w[0, 1, :] = LUMA_FINEST
    for j in range(1, levels + 1):
        w[1:, j, :] = CHROMA_WEIGHTS[min(j, len(CHROMA_WEIGHTS)) - 1]
        w[:, j, 1] *= DIAGONAL
    w[1:, levels + 1, :] = CHROMA_WEIGHTS[-1]
    return w


def levels_for(w: int, h: int) -> int:
    m = min(w, h)
    return int(max(0, min(MAX_LEVELS, m.bit_length() - 1 - 3)))


# --------------------------------------------------------------------------
# Integer lifting wavelets (5/3 and 9/7), parallel over rows and columns
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _fdiv(a, s):
    """floor(a / 2^s) for any sign."""
    return a >> s if a >= 0 else -((-a + (1 << s) - 1) >> s)


@njit(cache=True, nogil=True)
def _lift_fwd(x, n, steps, out):
    if n <= 1:
        out[0] = x[0]
        return
    nh = n // 2
    nl = n - nh
    e = np.empty(nl, np.int64)
    o = np.empty(nh, np.int64)
    for i in range(nl):
        e[i] = x[2 * i]
    for i in range(nh):
        o[i] = x[2 * i + 1]
    for k in range(steps.shape[0]):
        odd, sg, q, r, s = steps[k, 0], steps[k, 1], steps[k, 2], steps[k, 3], steps[k, 4]
        if odd:
            for i in range(nh):
                right = e[i + 1] if i + 1 < nl else e[i]
                o[i] += sg * _fdiv(q * (e[i] + right) + r, s)
        else:
            for i in range(nl):
                left = o[i - 1] if i > 0 else o[0]
                right = o[i] if i < nh else o[nh - 1]
                e[i] += sg * _fdiv(q * (left + right) + r, s)
    for i in range(nl):
        out[i] = e[i]
    for i in range(nh):
        out[nl + i] = o[i]


@njit(cache=True, nogil=True)
def _lift_inv(y, n, steps, out):
    if n <= 1:
        out[0] = y[0]
        return
    nh = n // 2
    nl = n - nh
    e = np.empty(nl, np.int64)
    o = np.empty(nh, np.int64)
    for i in range(nl):
        e[i] = y[i]
    for i in range(nh):
        o[i] = y[nl + i]
    for k in range(steps.shape[0] - 1, -1, -1):
        odd, sg, q, r, s = steps[k, 0], steps[k, 1], steps[k, 2], steps[k, 3], steps[k, 4]
        if odd:
            for i in range(nh):
                right = e[i + 1] if i + 1 < nl else e[i]
                o[i] -= sg * _fdiv(q * (e[i] + right) + r, s)
        else:
            for i in range(nl):
                left = o[i - 1] if i > 0 else o[0]
                right = o[i] if i < nh else o[nh - 1]
                e[i] -= sg * _fdiv(q * (left + right) + r, s)
    for i in range(nl):
        out[2 * i] = e[i]
    for i in range(nh):
        out[2 * i + 1] = o[i]


@njit(cache=True, nogil=True)
def _fdiv32(a, s):
    return a >> s if a >= 0 else -((-a + (1 << s) - 1) >> s)


@njit(cache=True, parallel=True)
def _rows(a, h, w, steps, inverse):
    """Lifting along each row (rows are contiguous in memory)."""
    if w <= 1:
        return
    nh = w // 2
    nl = w - nh
    for y in prange(h):
        e = np.empty(nl, np.int32)
        o = np.empty(nh, np.int32)
        if inverse:
            for i in range(nl):
                e[i] = a[y, i]
            for i in range(nh):
                o[i] = a[y, nl + i]
        else:
            for i in range(nl):
                e[i] = a[y, 2 * i]
            for i in range(nh):
                o[i] = a[y, 2 * i + 1]
        ns = steps.shape[0]
        for kk in range(ns):
            k = ns - 1 - kk if inverse else kk
            odd, sg, q, r, s = steps[k, 0], steps[k, 1], steps[k, 2], steps[k, 3], steps[k, 4]
            if inverse:
                sg = -sg
            if odd:
                for i in range(nh):
                    right = e[i + 1] if i + 1 < nl else e[i]
                    o[i] += sg * _fdiv32(q * (e[i] + right) + r, s)
            else:
                for i in range(nl):
                    left = o[i - 1] if i > 0 else o[0]
                    right = o[i] if i < nh else o[nh - 1]
                    e[i] += sg * _fdiv32(q * (left + right) + r, s)
        if inverse:
            for i in range(nl):
                a[y, 2 * i] = e[i]
            for i in range(nh):
                a[y, 2 * i + 1] = o[i]
        else:
            for i in range(nl):
                a[y, i] = e[i]
            for i in range(nh):
                a[y, nl + i] = o[i]


@njit(cache=True, parallel=True)
def _col_step_odd(E, O, nl, nh, w, sg, q, r, s):
    for ip in prange(nh):
        i = np.int64(ip)                        # parallel index is unsigned
        i2 = i + 1 if i + 1 < nl else i
        for x in range(w):                      # contiguous: vectorises
            O[i, x] += sg * _fdiv32(q * (E[i, x] + E[i2, x]) + r, s)


@njit(cache=True, parallel=True)
def _col_step_even(E, O, nl, nh, w, sg, q, r, s):
    for ip in prange(nl):
        i = np.int64(ip)
        il = i - 1 if i > 0 else 0
        ir = i if i < nh else nh - 1
        for x in range(w):
            E[i, x] += sg * _fdiv32(q * (O[il, x] + O[ir, x]) + r, s)


def _cols(a, h, w, steps, inverse):
    """Lifting down the columns, done a whole row at a time so memory is
    read in order (walking down single columns defeats the CPU cache)."""
    if h <= 1:
        return
    nh = h // 2
    nl = h - nh
    # explicit copies: a full-width slice would otherwise be a view into `a`
    # and the write-back below would overwrite values still being read
    if inverse:
        E = a[:nl, :w].copy()
        O = a[nl:h, :w].copy()
    else:
        E = a[0:h:2, :w].copy()
        O = a[1:h:2, :w].copy()
    order = range(steps.shape[0] - 1, -1, -1) if inverse else range(steps.shape[0])
    for k in order:
        odd, sg, q, r, s = (int(v) for v in steps[k])
        if inverse:
            sg = -sg
        if odd:
            _col_step_odd(E, O, nl, nh, w, sg, q, r, s)
        else:
            _col_step_even(E, O, nl, nh, w, sg, q, r, s)
    if inverse:
        a[0:h:2, :w] = E
        a[1:h:2, :w] = O
    else:
        a[:nl, :w] = E
        a[nl:h, :w] = O


def forward(a: np.ndarray, levels: int, steps: np.ndarray) -> None:
    h, w = a.shape
    for _ in range(levels):
        _rows(a, h, w, steps, False)
        _cols(a, h, w, steps, False)
        h, w = (h + 1) // 2, (w + 1) // 2


def inverse(a: np.ndarray, levels: int, steps: np.ndarray, stop: int = 0) -> None:
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels - 1, stop - 1, -1):
        _cols(a, hs[j], ws[j], steps, True)
        _rows(a, hs[j], ws[j], steps, True)


# --------------------------------------------------------------------------
# Band gains: the 9/7 lifting is unscaled, so its bands come out K or 1/K
# times larger per dimension than the 5/3 convention the step rules assume.
# --------------------------------------------------------------------------
def band_gains(levels: int, is97: bool, prescale: int) -> np.ndarray:
    g = np.ones((levels + 2, 2))
    if is97:
        k2 = K97 * K97
        for j in range(1, levels + 1):
            g[j, 0] = k2 ** (j - 1)              # HL, LH: one low and one high pass
            g[j, 1] = k2 ** (j - 1) / k2         # HH: two high passes
        g[levels + 1, :] = k2 ** levels          # LL: low passes only
    return g * (1 << prescale)


_noise_cache: dict = {}


def to_coefficients(pix: np.ndarray, levels: int, steps, prescale: int, extra: int) -> np.ndarray:
    """Pixels -> wavelet coefficients in file units (pixel x 2^prescale), with
    the transform itself running `extra` bits more precisely. 32-bit integers
    are enough: values stay below ~400k and lifting products below 2^31."""
    a = np.ascontiguousarray(pix.astype(np.int32) << (prescale + extra))
    forward(a, levels, steps)
    if extra:
        a = (a + (1 << (extra - 1))) >> extra
    return a


def noise_gains(levels: int, is97: bool, prescale: int, extra: int = 0) -> np.ndarray:
    """Standard deviation each band gets from unit white noise. G[level, orient]."""
    key = (levels, is97, prescale, extra)
    if key not in _noise_cache:
        rng = np.random.default_rng(0)
        a = to_coefficients(np.round(rng.normal(0, 64, (512, 512))), levels,
                            STEPS_97 if is97 else STEPS_53, prescale, extra)
        hs, ws = _level_dims(512, 512, levels)
        g = np.ones((levels + 2, 3))
        for j in range(1, levels + 1):
            for o in range(3):
                by0, bx0, bh, bw, *_ = _band_geometry(hs, ws, j, levels, o)
                g[j, o] = a[by0:by0 + bh, bx0:bx0 + bw].std() / 64
        _noise_cache[key] = g
    return _noise_cache[key]


# --------------------------------------------------------------------------
# Code-blocks
# --------------------------------------------------------------------------
def chunk_list(w: int, h: int, levels: int):
    """Every code-block in file order: layer, then plane, then band, then
    strip. Entries: (layer, plane, orient, strip_y0, strip_h). Layer 0 is the
    LL band (one block per plane); layer k holds detail level levels+1-k."""
    hs, ws = _level_dims(h, w, levels)
    out = []
    for p in range(3):
        out.append((0, p, -1, 0, int(hs[levels])))
    for layer in range(1, levels + 1):
        j = levels + 1 - layer
        for p in range(3):
            for o in range(3):
                by0, bx0, bh, bw, *_ = _band_geometry(hs, ws, j, levels, o)
                if bh <= 0 or bw <= 0:
                    continue
                rows = max(64, -(-BLOCK_MIN // max(1, bw)))
                rows = -(-rows // 64) * 64                 # multiple of 64
                for sy in range(0, bh, rows):
                    out.append((layer, p, o, int(sy), int(min(rows, bh - sy))))
    return out


@fast
def _ctx5(a, yy, xx, sy0, bx0, by0, py0, px0, ph, pw):
    """Neighbour + parent context. The upper neighbour only counts inside
    this code-block (strip), so blocks decode independently."""
    left = abs(np.int64(a[yy, xx - 1])) if xx > bx0 else 0
    up = abs(np.int64(a[yy - 1, xx])) if yy > sy0 else 0
    nb = left + up
    nbin = 0 if nb == 0 else (1 if nb == 1 else (2 if nb == 2 else (3 if nb <= 4 else 4)))
    pbin = 0
    if ph > 0 and pw > 0:
        py = min(ph - 1, (yy - by0) >> 1)
        px = min(pw - 1, (xx - bx0) >> 1)
        pbin = min(2, abs(np.int64(a[py0 + py, px0 + px])))
    return M_MAG + nbin * 3 + pbin


@njit(cache=True, nogil=True)
def _region_code(a, sy0, bx0, sh, bw, by0, py0, px0, ph, pw, encode,
                 st, starts, freqs, raw, buf, nwords, x4, cdf, cnt, nsym, stack):
    if sh <= 0 or bw <= 0:
        return
    size = LEAF
    while size < sh or size < bw:
        size *= 2
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack[0, 2] = size
    sp = 1
    while sp > 0:
        sp -= 1
        y, x, s = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        h = min(s, sh - y)
        w = min(s, bw - x)
        if h <= 0 or w <= 0:
            continue
        if encode:
            nz = 0
            for yy in range(sy0 + y, sy0 + y + h):
                for xx in range(bx0 + x, bx0 + x + w):
                    if a[yy, xx] != 0:
                        nz = 1
                        break
                if nz:
                    break
            _emit(st, starts, freqs, cdf, cnt, nsym, _node_model(s), nz)
        else:
            nz = _get(buf, nwords, x4, st, cdf, cnt, nsym, _node_model(s))
        if nz == 0:
            continue
        if s <= LEAF:
            for yy in range(sy0 + y, sy0 + y + h):
                for xx in range(bx0 + x, bx0 + x + w):
                    m = _ctx5(a, yy, xx, sy0, bx0, by0, py0, px0, ph, pw)
                    if encode:
                        v = np.int64(a[yy, xx])
                        _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, m, abs(v))
                        if v != 0:
                            _raw_put(st, raw, 1 if v < 0 else 0, 1)
                    else:
                        v = _get_value(buf, nwords, x4, st, raw, cdf, cnt, nsym, m)
                        if v != 0 and _raw_get(raw, st, 1):
                            v = -v
                        a[yy, xx] = v
        else:
            half = s // 2
            for cy, cx in ((half, half), (half, 0), (0, half), (0, 0)):
                stack[sp, 0] = y + cy
                stack[sp, 1] = x + cx
                stack[sp, 2] = half
                sp += 1


@njit(cache=True, nogil=True)
def _pack(starts, freqs, nsym_count, raw, raw_bits):
    words = np.empty(nsym_count + 16, np.int64)
    p0 = _rans_encode(starts, freqs, nsym_count, words)
    nw = words.size - p0
    nraw = (raw_bits + 7) >> 3
    out = np.empty(4 + 2 * nw + nraw, np.uint8)
    out[0] = nw & 0xFF
    out[1] = (nw >> 8) & 0xFF
    out[2] = (nw >> 16) & 0xFF
    out[3] = (nw >> 24) & 0xFF
    for i in range(nw):
        out[4 + 2 * i] = words[p0 + i] & 0xFF
        out[5 + 2 * i] = words[p0 + i] >> 8
    for i in range(nraw):
        out[4 + 2 * nw + i] = raw[i]
    return out


@njit(cache=True, nogil=True)
def _encode_detail(a, levels, j, orient, sy, sh, init_cdf):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, orient)
    n = sh * bw
    starts = np.empty(2 * n + 1024, np.int64)
    freqs = np.empty(2 * n + 1024, np.int64)
    raw = np.zeros(6 * n + 1024, np.uint8)
    st = np.zeros(3, np.int64)
    stack = np.empty((64, 3), np.int64)
    cdf, cnt, nsym = _models_init(init_cdf)
    dummy = np.zeros(1, np.uint8)
    _region_code(a, by0 + sy, bx0, sh, bw, by0, py0, px0, ph, pw, True,
                 st, starts, freqs, raw, dummy, 0, np.zeros(4, np.int64),
                 cdf, cnt, nsym, stack)
    return _pack(starts, freqs, st[0], raw, st[1])


@njit(cache=True, nogil=True)
def _decode_detail(a, levels, j, orient, sy, sh, buf, init_cdf):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, orient)
    if buf.size < 4:
        return
    nwords = np.int64(buf[0]) | (np.int64(buf[1]) << 8) | (np.int64(buf[2]) << 16) | (np.int64(buf[3]) << 24)
    if 4 + 2 * nwords > buf.size:
        nwords = (buf.size - 4) // 2
    raw = buf[4 + 2 * nwords:]
    st = np.zeros(3, np.int64)
    x4 = np.zeros(4, np.int64)
    stack = np.empty((64, 3), np.int64)
    cdf, cnt, nsym = _models_init(init_cdf)
    _rans_start(buf, nwords, x4, st)
    st[2] = 0
    dummy = np.zeros(1, np.int64)
    _region_code(a, by0 + sy, bx0, sh, bw, by0, py0, px0, ph, pw, False,
                 st, dummy, dummy, raw, buf, nwords, x4, cdf, cnt, nsym, stack)


@njit(cache=True, nogil=True)
def _encode_ll(a, hl, wl, mp, with_map, gs, init_cdf):
    n = hl * wl + mp.size + 64
    starts = np.empty(2 * n + 1024, np.int64)
    freqs = np.empty(2 * n + 1024, np.int64)
    raw = np.zeros(6 * n + gs.size + 1024, np.uint8)
    st = np.zeros(3, np.int64)
    cdf, cnt, nsym = _models_init(init_cdf)
    for i in range(gs.size):
        _raw_put(st, raw, gs.reshape(gs.size)[i], 8)
    if with_map:
        for by in range(mp.shape[0]):
            for bx in range(mp.shape[1]):
                _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, M_MAP, mp[by, bx])
    for y in range(hl):
        for x in range(wl):
            r = np.int64(a[y, x]) - _ll_pred(a, y, x)
            z = (r << 1) if r >= 0 else ((-r) << 1) - 1
            _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, M_LL + _ll_ctx(a, y, x), z)
    return _pack(starts, freqs, st[0], raw, st[1])


@njit(cache=True, nogil=True)
def _decode_ll(a, hl, wl, mp, with_map, gs, buf, init_cdf):
    if buf.size < 4:
        return
    nwords = np.int64(buf[0]) | (np.int64(buf[1]) << 8) | (np.int64(buf[2]) << 16) | (np.int64(buf[3]) << 24)
    if 4 + 2 * nwords > buf.size:
        nwords = (buf.size - 4) // 2
    raw = buf[4 + 2 * nwords:]
    st = np.zeros(3, np.int64)
    x4 = np.zeros(4, np.int64)
    cdf, cnt, nsym = _models_init(init_cdf)
    _rans_start(buf, nwords, x4, st)
    st[2] = 0
    flat = gs.reshape(gs.size)
    for i in range(flat.size):
        flat[i] = _raw_get(raw, st, 8)
    if with_map:
        for by in range(mp.shape[0]):
            for bx in range(mp.shape[1]):
                mp[by, bx] = min(7, _get_value(buf, nwords, x4, st, raw, cdf, cnt, nsym, M_MAP))
    for y in range(hl):
        for x in range(wl):
            z = _get_value(buf, nwords, x4, st, raw, cdf, cnt, nsym, M_LL + _ll_ctx(a, y, x))
            r = (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)
            a[y, x] = _ll_pred(a, y, x) + r


# --------------------------------------------------------------------------
# Final stage: undo the pre-scale, add grain, convert YCoCg -> RGB
# --------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _finish(planes, h, w, shift, gs, grain, amount, out):
    half = (1 << shift) >> 1
    for y in prange(h):
        for x in range(w):
            Y = _fdiv(np.int64(planes[0, y, x]) + half, shift)
            co = _fdiv(np.int64(planes[1, y, x]) + half, shift)
            cg = _fdiv(np.int64(planes[2, y, x]) + half, shift)
            if grain:
                cy, cx = y >> 8, x >> 8
                f = (Y - 32) / 64.0
                if f <= 0:
                    sig = gs[cy, cx, 0] * 1.0
                elif f >= 3:
                    sig = gs[cy, cx, 3] * 1.0
                else:
                    i0 = int(f)
                    sig = gs[cy, cx, i0] + (gs[cy, cx, i0 + 1] - gs[cy, cx, i0]) * (f - i0)
                hv = _hash32((y << 16) ^ x ^ 0x5F3759DF)
                g = ((hv & 0xFF) + ((hv >> 8) & 0xFF) + ((hv >> 16) & 0xFF)
                     + (hv >> 24) - 510.0) / 147.8
                Y += int(np.floor(g * sig * amount / 8.0 + 0.5))
            t = Y - (cg >> 1)
            gg = cg + t
            b = t - (co >> 1)
            r = b + co
            out[y, x, 0] = min(255, max(0, r))
            out[y, x, 1] = min(255, max(0, gg))
            out[y, x, 2] = min(255, max(0, b))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def encode(rgb: np.ndarray, quality: int = 75, perceptual: bool = True,
           denoise: float | str = "auto", grain: bool | str = "auto",
           masking: bool = False, rdo: bool = True,
           weights: np.ndarray | None = None, wavelet: str = "auto") -> bytes:
    """quality 1-100 (100 = exactly lossless). wavelet: "auto" (9/7 for lossy,
    5/3 for lossless), "97" or "53". Other options as in v4."""
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    lossless = quality == 100
    perceptual = perceptual and not lossless
    rdo = rdo and not lossless
    if lossless:
        denoise = 0.0
    elif denoise == "auto":
        denoise = auto_denoise_strength(estimate_noise(rgb))
    else:
        denoise = float(denoise)
    grain = ((denoise > 0) if grain == "auto" else bool(grain)) and not lossless
    is97 = (not lossless) if wavelet == "auto" else (wavelet == "97")
    prescale = PRESCALE_97 if is97 else 0
    extra = EXTRA_97 if is97 else 0
    steps = STEPS_97 if is97 else STEPS_53

    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    pix = np.stack([y, co, cg])                       # pixel-domain planes
    levels = levels_for(w, h)
    base = quality_to_base(quality)
    if weights is not None and perceptual:
        wp = np.asarray(weights, dtype=np.float64)
    else:
        wp = perceptual_weights(levels) if perceptual else np.ones((3, levels + 2, 2))
    wts = wp * band_gains(levels, is97, prescale)[None, :, :]
    wts = wts.astype(np.float32).astype(np.float64)   # exactly what the file stores

    if perceptual and masking:
        mp = _masking_map(np.ascontiguousarray(pix[0]))
    elif perceptual:
        mp = np.full(((h + 15) >> BLOCK_SHIFT, (w + 15) >> BLOCK_SHIFT), 2, np.int32)
    else:
        mp = np.full((1, 1), 2, np.int32)
    gains = noise_gains(levels, is97, prescale, extra)

    coef = np.empty((3, h, w), np.int32)
    def transform(p):
        a = to_coefficients(pix[p], levels, steps, prescale, extra)
        T = _shrink_thresholds(a, levels, gains, denoise)
        a32 = a.astype(np.int32)
        _quantise(a32, levels, base, wts, p, MULTS, mp, rdo, T)
        coef[p] = a32
    for p in range(3):
        transform(p)

    ncy, ncx = -(-h // GRAIN_CELL), -(-w // GRAIN_CELL)
    gs = np.zeros((ncy, ncx, 4), np.int64)
    if grain:
        rec = coef[0].astype(np.int32).copy()
        _dequantise(rec, levels, base, wts, 0, MULTS, mp, 1)
        rec64 = np.ascontiguousarray(rec << extra)
        inverse(rec64, levels, steps)
        sh = prescale + extra
        rec_pix = ((rec64 + ((1 << sh) >> 1)) >> sh).astype(np.int32)
        for cy in range(ncy):
            for cx in range(ncx):
                ys, xs = slice(cy * GRAIN_CELL, (cy + 1) * GRAIN_CELL), slice(cx * GRAIN_CELL, (cx + 1) * GRAIN_CELL)
                gs[cy, cx] = _grain_sigmas(np.ascontiguousarray(pix[0][ys, xs]),
                                           np.ascontiguousarray(rec_pix[ys, xs]))

    hs, ws = _level_dims(h, w, levels)
    chunks = chunk_list(w, h, levels)

    def work(c):
        layer, p, orient, sy, sh = c
        if layer == 0:
            g_here = gs if (p == 0 and grain) else np.zeros((0, 0, 4), np.int64)
            return _encode_ll(coef[p], hs[levels], ws[levels], mp, p == 0 and perceptual,
                              g_here, INIT_CDF)
        return _encode_detail(coef[p], levels, levels + 1 - layer, orient, sy, sh, INIT_CDF)

    payload = list(_pool().map(work, chunks))
    flags = ((FLAG_PERCEPTUAL if perceptual else 0) | (FLAG_GRAIN if grain else 0)
             | (FLAG_97 if is97 else 0))
    header = HEADER.pack(MAGIC, VERSION, w, h, quality, flags, levels,
                         prescale | (extra << 4), base)
    table = np.asarray([len(c) for c in payload], dtype="<u4").tobytes()
    return header + wts.astype("<f4").tobytes() + table + b"".join(c.tobytes() for c in payload)


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL v5 image")
    magic, version, w, h, quality, flags, levels, precision, base = HEADER.unpack_from(data)
    prescale, extra = precision & 15, precision >> 4
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Not a FUL v5 file (version {version})")
    if levels != levels_for(w, h) or prescale > 4 or extra > 6:
        raise ValueError("Corrupt FUL v5 header")
    return {"version": 5, "width": w, "height": h, "quality": quality,
            "perceptual": bool(flags & FLAG_PERCEPTUAL), "grain": bool(flags & FLAG_GRAIN),
            "wavelet": "9/7" if flags & FLAG_97 else "5/3", "levels": levels,
            "prescale": prescale, "extra": extra, "base": base}


def _layout(data: bytes, info: dict):
    levels = info["levels"]
    nw = 3 * (levels + 2) * 2
    wts = np.frombuffer(data, "<f4", nw, HEADER.size).astype(np.float64).reshape(3, levels + 2, 2)
    chunks = chunk_list(info["width"], info["height"], levels)
    table_at = HEADER.size + 4 * nw
    base = table_at + 4 * len(chunks)
    if len(data) < base:
        raise ValueError("FUL v5 file is truncated")
    lengths = np.frombuffer(data, "<u4", len(chunks), table_at).astype(np.int64)
    starts = base + np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    if len(chunks) and starts[-1] + lengths[-1] > len(data):
        raise ValueError("FUL v5 file is truncated")
    return wts, chunks, starts, lengths, base


def layer_count(data: bytes) -> int:
    return read_header(data)["levels"] + 1


def bytes_for_layers(data: bytes, layers: int) -> int:
    info = read_header(data)
    _, chunks, starts, lengths, base = _layout(data, info)
    return int(base + sum(l for c, l in zip(chunks, lengths) if c[0] < layers))


def decode_coefficients(data: bytes, layers: int | None = None):
    """CPU stage: entropy decoding + dequantisation. Returns (coef, grain,
    info, mp) with coef as (3, H, W) int32 wavelet coefficients."""
    info = read_header(data)
    levels, w, h = info["levels"], info["width"], info["height"]
    k = levels + 1 if layers is None else max(1, min(levels + 1, int(layers)))
    wts, chunks, starts, lengths, _ = _layout(data, info)
    buf = np.frombuffer(data, np.uint8)
    coef = np.zeros((3, h, w), np.int32)
    hs, ws = _level_dims(h, w, levels)
    if info["perceptual"]:
        mp = np.zeros(((h + 15) >> BLOCK_SHIFT, (w + 15) >> BLOCK_SHIFT), np.int32)
    else:
        mp = np.full((1, 1), 2, np.int32)
    ncy, ncx = -(-h // GRAIN_CELL), -(-w // GRAIN_CELL)
    gs = np.zeros((ncy, ncx, 4), np.int64)

    for layer in range(k):        # coarse to fine: parents decode first
        todo = [(c, s, l) for c, s, l in zip(chunks, starts, lengths) if c[0] == layer]

        def work(item):
            (lay, p, orient, sy, sh), s, l = item
            piece = np.array(buf[s:s + l])     # writable copy
            if lay == 0:
                g_here = gs if (p == 0 and info["grain"]) else np.zeros((0, 0, 4), np.int64)
                _decode_ll(coef[p], hs[levels], ws[levels], mp, p == 0 and info["perceptual"],
                           g_here, piece, INIT_CDF)
            else:
                _decode_detail(coef[p], levels, levels + 1 - lay, orient, sy, sh, piece, INIT_CDF)

        list(_pool().map(work, todo))
    stop = levels - (k - 1)
    list(_pool().map(lambda p: _dequantise(coef[p], levels, float(info["base"]), wts, p,
                                           MULTS, mp, stop + 1), range(3)))
    return coef, (gs if info["grain"] else None), info, stop


def decode(data: bytes, layers: int | None = None, grain: float = 1.0) -> np.ndarray:
    coef, gs, info, stop = decode_coefficients(data, layers)
    levels = info["levels"]
    steps = STEPS_97 if info["wavelet"] == "9/7" else STEPS_53
    planes = coef << info["extra"]                    # int32
    for p in range(3):
        inverse(planes[p], levels, steps, stop)
    hs, ws = _level_dims(info["height"], info["width"], levels)
    h, w = int(hs[stop]), int(ws[stop])
    shift = info["prescale"] + info["extra"]
    if stop > 0:
        # a preview: the LL band also carries the wavelet's low-pass gain
        gain = (K97 * K97) ** stop if info["wavelet"] == "9/7" else 1.0
        sub = planes[:, :h, :w].astype(np.float64) / gain
        planes = np.floor(sub + 0.5).astype(np.int32)
    out = np.empty((h, w, 3), np.uint8)
    use_grain = gs is not None and grain > 0 and stop == 0
    _finish(np.ascontiguousarray(planes[:, :h, :w]), h, w, shift,
            gs if gs is not None else np.zeros((1, 1, 4), np.int64),
            use_grain, float(grain), out)
    return out


if __name__ == "__main__":
    import time

    import codec_v1
    import codec_v4

    rng = np.random.default_rng(13)
    a = rng.integers(-300, 300, (37, 53)).astype(np.int32)
    for steps in (STEPS_53, STEPS_97):
        b = a.copy(); forward(b, 3, steps); inverse(b, 3, steps)
        assert np.array_equal(a, b), "integer lifting must be exactly reversible"
    for shape in [(1, 1, 3), (5, 70, 3), (37, 53, 3), (300, 260, 3), (513, 771, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(decode(encode(img, 100)), img), shape
        data = encode(img, 60, denoise=1.0, grain=True, masking=True)
        for k in range(1, layer_count(data) + 1):
            decode(data, k)
    print("Integer wavelets reversible; lossless round trips and progressive decodes verified.")
    img = codec_v1.make_test_image()
    for q in (100, 90, 75, 50):
        d4, d5 = codec_v4.encode(img, q), encode(img, q)
        print(f"quality {q:3d}: v4 {len(d4):>9,} bytes   v5 {len(d5):>9,} bytes")
    d = encode(img, 75); decode(d); t = time.perf_counter(); decode(d)
    print(f"v5 decode {(time.perf_counter() - t) * 1e3:.1f} ms")
