"""
FUL v6: v5 plus real-cost rate-distortion optimisation and honeycomb-style
directional lifting.

Real-cost RDO (encoder only; decoders are unaffected):
  For every detail band, the encoder first measures how often each value
  appears in each context, giving the real bit cost of every option. Each
  coefficient then picks between its rounded value, one less, or zero by
  minimising  J = D + lambda * bits  (D in squared quantiser steps), using
  its already-decided neighbours for context. Finally every 4x4 group is kept
  or dropped by comparing the real costs of both options, replacing v3-v5's
  fixed 20-bit guess.

Directional lifting (the honeycomb idea):
  On the two finest levels, the column (vertical) lifting steps can predict
  along a slanted line instead of straight up and down. Every 16x16 block gets
  one of 7 directions: straight, or a horizontal shift of 1/2, 1 or 2 pixels
  per row either way (roughly 27, 45 and 63 degrees, which includes the
  honeycomb's 30 and 60 degree neighbours). Half-pixel positions use the
  floor of the average of two neighbours. The encoder chooses each block's
  direction by which leaves the smallest detail coefficients and stores the
  choices in the file; the decoder just follows them. Any deterministic
  prediction keeps integer lifting exactly reversible.
  Each level now runs columns first, then rows (v5: rows, then columns).

Run for a self-test:  python codec_v6.py
"""
from __future__ import annotations

import struct

import numpy as np

from codec_v2 import _ll_pred, _pool, njit
from codec_v3 import BLOCK_SHIFT, DELTA, MULTS, _masking_map, quality_to_base
from codec_v4 import (
    INIT_CDF, M_LL, M_MAG, M_MAP, _band_geometry, _coef_step, _dequantise,
    _emit_value, _get_value, _grain_sigmas, _level_dims, _ll_ctx, _models_init,
    _rans_start, _raw_get, _raw_put, _shrink_thresholds, auto_denoise_strength,
    estimate_noise,
)
from codec_v5 import (
    EXTRA_97, GRAIN_CELL, PRESCALE_97, STEPS_53, STEPS_97, _cols, _decode_detail,
    _encode_detail, _fdiv32, _finish, _pack, _rows, band_gains, chunk_list,
    levels_for, noise_gains, perceptual_weights,
)

try:
    from numba import prange
except ImportError:
    prange = range

MAGIC = b"FUL\x00"
VERSION = 6
HEADER = struct.Struct("<4sBIIBBBBf")
FLAG_PERCEPTUAL = 1
FLAG_GRAIN = 2
FLAG_97 = 4
FLAG_DIR = 8

# half-pixel shift per row; index 0 = straight down
DIRS = np.array([0, -1, 1, -2, 2, -4, 4], np.int64)
DIR_SHIFT = 4                 # 16x16 direction blocks
DIR_LEVELS = 2                # directional lifting on the two finest levels
DIR_BIAS = 0.97               # prefer straight unless a slant is 3% better
RDO_LAMBDA = 0.12             # distortion (squared steps) per bit
NCTX = 15


# --------------------------------------------------------------------------
# Directional column lifting
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _samp(R, i, x, dh, w):
    """Row i of R at column x + dh/2 (half-pixel units), clamped to the edge.
    Half positions use the floor of the average of the two neighbours."""
    if (dh & 1) == 0:
        c = min(w - 1, max(0, x + (dh >> 1)))
        return np.int64(R[i, c])
    c0 = min(w - 1, max(0, x + ((dh - 1) >> 1)))
    c1 = min(w - 1, max(0, x + ((dh + 1) >> 1)))
    return _fdiv32(np.int64(R[i, c0]) + np.int64(R[i, c1]), 1)


@njit(cache=True, parallel=True)
def _dcol_odd(E, O, nl, nh, w, sg, q, r, s, D, dirs):
    for ip in prange(nh):
        i = np.int64(ip)
        i2 = i + 1 if i + 1 < nl else i
        row = 2 * i + 1
        for x in range(w):
            dh = dirs[D[row >> DIR_SHIFT, x >> DIR_SHIFT]]
            a = _samp(E, i, x, -dh, w)              # above, shifted back
            b = _samp(E, i2, x, dh, w)              # below, shifted forward
            O[i, x] += sg * _fdiv32(q * (a + b) + r, s)


@njit(cache=True, parallel=True)
def _dcol_even(E, O, nl, nh, w, sg, q, r, s, D, dirs):
    for ip in prange(nl):
        i = np.int64(ip)
        il = i - 1 if i > 0 else 0
        ir = i if i < nh else nh - 1
        row = 2 * i
        for x in range(w):
            dh = dirs[D[row >> DIR_SHIFT, x >> DIR_SHIFT]]
            a = _samp(O, il, x, -dh, w)
            b = _samp(O, ir, x, dh, w)
            E[i, x] += sg * _fdiv32(q * (a + b) + r, s)


def _cols6(a, h, w, steps, inverse, D):
    if D is None:
        _cols(a, h, w, steps, inverse)
        return
    if h <= 1:
        return
    nh, nl = h // 2, h - h // 2
    if inverse:
        E, O = a[:nl, :w].copy(), a[nl:h, :w].copy()
    else:
        E, O = a[0:h:2, :w].copy(), a[1:h:2, :w].copy()
    order = range(steps.shape[0] - 1, -1, -1) if inverse else range(steps.shape[0])
    for k in order:
        odd, sg, q, r, s = (int(v) for v in steps[k])
        if inverse:
            sg = -sg
        (_dcol_odd if odd else _dcol_even)(E, O, nl, nh, w, sg, q, r, s, D, DIRS)
    if inverse:
        a[0:h:2, :w], a[1:h:2, :w] = E, O
    else:
        a[:nl, :w], a[nl:h, :w] = E, O


def forward6(a, levels, steps, dirmaps):
    h, w = a.shape
    for j in range(levels):
        _cols6(a, h, w, steps, False, dirmaps[j] if j < len(dirmaps) else None)
        _rows(a, h, w, steps, False)
        h, w = (h + 1) // 2, (w + 1) // 2


def inverse6(a, levels, steps, dirmaps, stop=0):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels - 1, stop - 1, -1):
        _rows(a, hs[j], ws[j], steps, True)
        _cols6(a, hs[j], ws[j], steps, True, dirmaps[j] if j < len(dirmaps) else None)


def dir_map_shapes(w, h, levels, directional):
    if not directional:
        return []
    hs, ws = _level_dims(h, w, levels)
    return [(-(-int(hs[j]) // 16), -(-int(ws[j]) // 16)) for j in range(min(DIR_LEVELS, levels))]


@njit(cache=True)
def _block_cost(hi, bh, bw):
    out = np.zeros((bh, bw))
    for i in range(hi.shape[0]):
        for x in range(hi.shape[1]):
            out[(2 * i + 1) >> DIR_SHIFT, x >> DIR_SHIFT] += abs(hi[i, x])
    return out


def choose_directions(y, levels, steps, prescale, extra):
    """For each 16x16 block on the finest levels, the direction whose column
    lifting leaves the smallest detail coefficients."""
    a = np.ascontiguousarray(y.astype(np.int32) << (prescale + extra))
    h, w = a.shape
    maps = []
    for j in range(min(DIR_LEVELS, levels)):
        bh, bw = -(-h // 16), -(-w // 16)
        nl = h - h // 2
        best = np.zeros((bh, bw), np.int32)
        best_cost = None
        for k in range(len(DIRS)):
            t = a[:h, :w].copy()
            _cols6(t, h, w, steps, False, np.full((bh, bw), k, np.int32))
            cost = _block_cost(np.ascontiguousarray(t[nl:h, :w]), bh, bw)
            if k == 0:
                cost *= DIR_BIAS
                best_cost = cost
            else:
                better = cost < best_cost
                best[better] = k
                best_cost = np.where(better, cost, best_cost)
        maps.append(best)
        _cols6(a, h, w, steps, False, best)
        _rows(a, h, w, steps, False)
        h, w = (h + 1) // 2, (w + 1) // 2
    return maps


# --------------------------------------------------------------------------
# Real-cost rate-distortion optimised quantisation
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _scale_band(a, U, y0, y1, x0, x1, shift, s_base, mults, mp, thr):
    for y in range(y0, y1):
        for x in range(x0, x1):
            c = a[y, x]
            mag = abs(c) - thr
            if mag < 0:
                mag = 0.0
            s = _coef_step(s_base, mults, mp, (y - y0) << shift, (x - x0) << shift)
            U[y, x] = mag / s if c >= 0 else -mag / s


@njit(cache=True, nogil=True)
def _scale_all(a, levels, base, wts, p, mults, mp, T):
    """Every coefficient in units of its own quantiser step (after denoising)."""
    U = np.zeros(a.shape, np.float32)
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels, 0, -1):
        for o in range(3):
            by0, bx0, bh, bw, _, _, _, _ = _band_geometry(hs, ws, j, levels, o)
            sb = base * wts[p, j, 1 if o == 2 else 0] / (1 << (j - 1))
            _scale_band(a, U, by0, by0 + bh, bx0, bx0 + bw, j, sb, mults, mp, T[j, o])
    s_ll = base * wts[p, levels + 1, 0] / (1 << levels)
    _scale_band(a, U, 0, hs[levels], 0, ws[levels], levels, s_ll, mults, mp, 0.0)
    return U


@njit(cache=True, nogil=True)
def _tok_bits(v):
    if v < 4:
        return v, 0
    n = 0
    t = v
    while t > 1:
        t >>= 1
        n += 1
    if n <= 12:
        return 2 + n, n
    return 15, 5 + n


@njit(cache=True, nogil=True)
def _cidx(q, yy, xx, by0, bx0, py0, px0, ph, pw):
    left = abs(np.int64(q[yy, xx - 1])) if xx > bx0 else 0
    up = abs(np.int64(q[yy - 1, xx])) if yy > by0 else 0
    nb = left + up
    nbin = 0 if nb == 0 else (1 if nb == 1 else (2 if nb == 2 else (3 if nb <= 4 else 4)))
    pbin = 0
    if ph > 0 and pw > 0:
        py = min(ph - 1, (yy - by0) >> 1)
        px = min(pw - 1, (xx - bx0) >> 1)
        pbin = min(2, abs(np.int64(q[py0 + py, px0 + px])))
    return nbin * 3 + pbin


@njit(cache=True, nogil=True)
def _rdo_band(q, U, by0, bx0, bh, bw, py0, px0, ph, pw, lam, up):
    if bh <= 0 or bw <= 0:
        return
    # 1) real costs: how often each token appears in each context
    cnt = np.full((NCTX, 16), 0.5)
    leaf = np.full(2, 0.5)
    for y in range(by0, by0 + bh):
        for x in range(bx0, bx0 + bw):
            t, _ = _tok_bits(abs(np.int64(q[y, x])))
            cnt[_cidx(q, y, x, by0, bx0, py0, px0, ph, pw), t] += 1.0
    for ly in range(by0, by0 + bh, 4):
        for lx in range(bx0, bx0 + bw, 4):
            nz = 0
            for y in range(ly, min(by0 + bh, ly + 4)):
                for x in range(lx, min(bx0 + bw, lx + 4)):
                    if q[y, x] != 0:
                        nz = 1
            leaf[nz] += 1.0
    cost = np.empty((NCTX, 16))
    for m in range(NCTX):
        tot = 0.0
        for t in range(16):
            tot += cnt[m, t]
        for t in range(16):
            cost[m, t] = -np.log2(cnt[m, t] / tot)
    lc0 = -np.log2(leaf[0] / (leaf[0] + leaf[1]))
    lc1 = -np.log2(leaf[1] / (leaf[0] + leaf[1]))
    # 2) each coefficient: rounded value, one less, or zero
    for y in range(by0, by0 + bh):
        for x in range(bx0, bx0 + bw):
            uu = U[y, x]
            u = abs(uu)
            m = _cidx(q, y, x, by0, bx0, py0, px0, ph, pw)
            best_v = 0
            best_j = u * u + lam * cost[m, 0]
            lv = int(u)
            for v in (lv + up, lv, lv - 1):
                if v <= 0:
                    continue
                t, nb = _tok_bits(v)
                d = u - (v + DELTA)
                jv = d * d + lam * (cost[m, t] + nb + 1.0)
                if jv < best_j:
                    best_j = jv
                    best_v = v
            q[y, x] = -best_v if uu < 0 else best_v
    # 3) each 4x4 group: keep it, or drop it entirely?
    for ly in range(by0, by0 + bh, 4):
        for lx in range(bx0, bx0 + bw, 4):
            ey, ex = min(by0 + bh, ly + 4), min(bx0 + bw, lx + 4)
            nz = 0
            d_keep = 0.0
            d_zero = 0.0
            bits = lc1
            for y in range(ly, ey):
                for x in range(lx, ex):
                    u = abs(U[y, x])
                    v = abs(np.int64(q[y, x]))
                    d_zero += u * u
                    m = _cidx(q, y, x, by0, bx0, py0, px0, ph, pw)
                    if v == 0:
                        d_keep += u * u
                        bits += cost[m, 0]
                    else:
                        nz += 1
                        e = u - (v + DELTA)
                        d_keep += e * e
                        t, nb = _tok_bits(v)
                        bits += cost[m, t] + nb + 1.0
            if nz and d_zero + lam * lc0 < d_keep + lam * bits:
                for y in range(ly, ey):
                    for x in range(lx, ex):
                        q[y, x] = 0


def quantise(a, levels, base, wts, p, mp, T, rdo, lam, up=0):
    U = _scale_all(a, levels, base, wts, p, MULTS, mp, T)
    q = np.trunc(U).astype(np.int32)
    if rdo:
        hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
        for j in range(levels, 0, -1):          # coarse first: parents decided
            for o in range(3):
                by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, o)
                _rdo_band(q, U, by0, bx0, bh, bw, py0, px0, ph, pw, lam, up)
    return q


# --------------------------------------------------------------------------
# LL block: grain, masking map, direction maps, then the LL band
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _encode_ll6(a, hl, wl, mp, with_map, gs, dirs, init_cdf):
    n = hl * wl + mp.size + dirs.size + 64
    starts = np.empty(2 * n + 1024, np.int64)
    freqs = np.empty(2 * n + 1024, np.int64)
    raw = np.zeros(6 * n + gs.size + 1024, np.uint8)
    st = np.zeros(3, np.int64)
    cdf, cnt, nsym = _models_init(init_cdf)
    flat = gs.reshape(gs.size)
    for i in range(flat.size):
        _raw_put(st, raw, flat[i], 8)
    if with_map:
        for by in range(mp.shape[0]):
            for bx in range(mp.shape[1]):
                _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, M_MAP, mp[by, bx])
    for i in range(dirs.size):
        _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, M_MAP, dirs[i])
    for y in range(hl):
        for x in range(wl):
            r = np.int64(a[y, x]) - _ll_pred(a, y, x)
            z = (r << 1) if r >= 0 else ((-r) << 1) - 1
            _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, M_LL + _ll_ctx(a, y, x), z)
    return _pack(starts, freqs, st[0], raw, st[1])


@njit(cache=True, nogil=True)
def _decode_ll6(a, hl, wl, mp, with_map, gs, dirs, buf, init_cdf):
    if buf.size < 4:
        return
    nwords = (np.int64(buf[0]) | (np.int64(buf[1]) << 8) | (np.int64(buf[2]) << 16)
              | (np.int64(buf[3]) << 24))
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
    for i in range(dirs.size):
        dirs[i] = min(6, _get_value(buf, nwords, x4, st, raw, cdf, cnt, nsym, M_MAP))
    for y in range(hl):
        for x in range(wl):
            z = _get_value(buf, nwords, x4, st, raw, cdf, cnt, nsym, M_LL + _ll_ctx(a, y, x))
            r = (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)
            a[y, x] = _ll_pred(a, y, x) + r


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def encode(rgb: np.ndarray, quality: int = 75, perceptual: bool = True,
           denoise: float | str = "auto", grain: bool | str = "auto",
           masking: bool = False, rdo: bool = True, directional: bool = True,
           lam: float = RDO_LAMBDA, weights: np.ndarray | None = None,
           round_up: bool = True) -> bytes:
    """quality 1-100 (100 = exactly lossless). rdo: real-cost rate-distortion
    optimised quantisation. directional: honeycomb-style directional lifting
    (lossy files only). Other options as in v4/v5."""
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
    is97 = not lossless
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
    pix = np.stack([y, co, cg])
    levels = levels_for(w, h)
    directional = directional and is97 and levels > 0
    base = quality_to_base(quality)
    if weights is not None and perceptual:
        wp = np.asarray(weights, dtype=np.float64)
    else:
        wp = perceptual_weights(levels) if perceptual else np.ones((3, levels + 2, 2))
    wts = (wp * band_gains(levels, is97, prescale)[None, :, :]).astype(np.float32).astype(np.float64)

    if perceptual and masking:
        mp = _masking_map(np.ascontiguousarray(pix[0]))
    elif perceptual:
        mp = np.full(((h + 15) >> BLOCK_SHIFT, (w + 15) >> BLOCK_SHIFT), 2, np.int32)
    else:
        mp = np.full((1, 1), 2, np.int32)
    gains = noise_gains(levels, is97, prescale, extra)
    dirmaps = choose_directions(pix[0], levels, steps, prescale, extra) if directional else []

    coef = np.empty((3, h, w), np.int32)
    for p in range(3):
        a = np.ascontiguousarray(pix[p].astype(np.int32) << (prescale + extra))
        forward6(a, levels, steps, dirmaps)
        if extra:
            a = (a + (1 << (extra - 1))) >> extra
        T = _shrink_thresholds(a.astype(np.int64), levels, gains, denoise)
        coef[p] = quantise(a, levels, base, wts, p, mp, T, rdo, lam, 1 if round_up else 0)

    ncy, ncx = -(-h // GRAIN_CELL), -(-w // GRAIN_CELL)
    gs = np.zeros((ncy, ncx, 4), np.int64)
    if grain:
        rec = coef[0].copy()
        _dequantise(rec, levels, base, wts, 0, MULTS, mp, 1)
        rec = np.ascontiguousarray(rec << extra)
        inverse6(rec, levels, steps, dirmaps)
        sh = prescale + extra
        rec_pix = ((rec.astype(np.int64) + ((1 << sh) >> 1)) >> sh).astype(np.int32)
        for cy in range(ncy):
            for cx in range(ncx):
                ys = slice(cy * GRAIN_CELL, (cy + 1) * GRAIN_CELL)
                xs = slice(cx * GRAIN_CELL, (cx + 1) * GRAIN_CELL)
                gs[cy, cx] = _grain_sigmas(np.ascontiguousarray(pix[0][ys, xs]),
                                           np.ascontiguousarray(rec_pix[ys, xs]))

    hs, ws = _level_dims(h, w, levels)
    dirs_flat = (np.concatenate([m.ravel() for m in dirmaps]).astype(np.int32)
                 if dirmaps else np.zeros(0, np.int32))
    chunks = chunk_list(w, h, levels)

    def work(c):
        layer, p, orient, sy, sh_ = c
        if layer == 0:
            first = p == 0
            return _encode_ll6(coef[p], hs[levels], ws[levels], mp, first and perceptual,
                               gs if (first and grain) else np.zeros((0, 0, 4), np.int64),
                               dirs_flat if first else np.zeros(0, np.int32), INIT_CDF)
        return _encode_detail(coef[p], levels, levels + 1 - layer, orient, sy, sh_, INIT_CDF)

    payload = list(_pool().map(work, chunks))
    flags = ((FLAG_PERCEPTUAL if perceptual else 0) | (FLAG_GRAIN if grain else 0)
             | (FLAG_97 if is97 else 0) | (FLAG_DIR if directional else 0))
    header = HEADER.pack(MAGIC, VERSION, w, h, quality, flags, levels,
                         prescale | (extra << 4), base)
    table = np.asarray([len(c) for c in payload], dtype="<u4").tobytes()
    return header + wts.astype("<f4").tobytes() + table + b"".join(c.tobytes() for c in payload)


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL v6 image")
    magic, version, w, h, quality, flags, levels, precision, base = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Not a FUL v6 file (version {version})")
    prescale, extra = precision & 15, precision >> 4
    if levels != levels_for(w, h) or prescale > 4 or extra > 6:
        raise ValueError("Corrupt FUL v6 header")
    return {"version": 6, "width": w, "height": h, "quality": quality,
            "perceptual": bool(flags & FLAG_PERCEPTUAL), "grain": bool(flags & FLAG_GRAIN),
            "wavelet": "9/7" if flags & FLAG_97 else "5/3",
            "directional": bool(flags & FLAG_DIR), "levels": levels,
            "prescale": prescale, "extra": extra, "base": base}


def _layout(data: bytes, info: dict):
    levels = info["levels"]
    nw = 3 * (levels + 2) * 2
    wts = np.frombuffer(data, "<f4", nw, HEADER.size).astype(np.float64).reshape(3, levels + 2, 2)
    chunks = chunk_list(info["width"], info["height"], levels)
    table_at = HEADER.size + 4 * nw
    base = table_at + 4 * len(chunks)
    if len(data) < base:
        raise ValueError("FUL v6 file is truncated")
    lengths = np.frombuffer(data, "<u4", len(chunks), table_at).astype(np.int64)
    starts = base + np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    if len(chunks) and starts[-1] + lengths[-1] > len(data):
        raise ValueError("FUL v6 file is truncated")
    return wts, chunks, starts, lengths, base


def layer_count(data: bytes) -> int:
    return read_header(data)["levels"] + 1


def bytes_for_layers(data: bytes, layers: int) -> int:
    info = read_header(data)
    _, chunks, starts, lengths, base = _layout(data, info)
    return int(base + sum(l for c, l in zip(chunks, lengths) if c[0] < layers))


def decode_coefficients(data: bytes, layers: int | None = None):
    """CPU stage: entropy decoding + dequantisation. Returns (coef, grain,
    info, stop, dirmaps)."""
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
    gs = np.zeros((-(-h // GRAIN_CELL), -(-w // GRAIN_CELL), 4), np.int64)
    shapes = dir_map_shapes(w, h, levels, info["directional"])
    dirs_flat = np.zeros(sum(a * b for a, b in shapes), np.int32)

    for layer in range(k):
        todo = [(c, s, l) for c, s, l in zip(chunks, starts, lengths) if c[0] == layer]

        def work(item):
            (lay, p, orient, sy, sh_), s, l = item
            piece = np.array(buf[s:s + l])
            if lay == 0:
                first = p == 0
                _decode_ll6(coef[p], hs[levels], ws[levels], mp, first and info["perceptual"],
                            gs if (first and info["grain"]) else np.zeros((0, 0, 4), np.int64),
                            dirs_flat if first else np.zeros(0, np.int32), piece, INIT_CDF)
            else:
                _decode_detail(coef[p], levels, levels + 1 - lay, orient, sy, sh_, piece, INIT_CDF)

        list(_pool().map(work, todo))
    stop = levels - (k - 1)
    list(_pool().map(lambda p: _dequantise(coef[p], levels, float(info["base"]), wts, p,
                                           MULTS, mp, stop + 1), range(3)))
    dirmaps, pos = [], 0
    for a_, b_ in shapes:
        dirmaps.append(dirs_flat[pos:pos + a_ * b_].reshape(a_, b_).copy())
        pos += a_ * b_
    return coef, (gs if info["grain"] else None), info, stop, dirmaps


def decode(data: bytes, layers: int | None = None, grain: float = 1.0) -> np.ndarray:
    coef, gs, info, stop, dirmaps = decode_coefficients(data, layers)
    levels = info["levels"]
    steps = STEPS_97 if info["wavelet"] == "9/7" else STEPS_53
    planes = np.ascontiguousarray(coef << info["extra"])
    for p in range(3):
        plane = np.ascontiguousarray(planes[p])
        inverse6(plane, levels, steps, dirmaps, stop)
        planes[p] = plane
    hs, ws = _level_dims(info["height"], info["width"], levels)
    h, w = int(hs[stop]), int(ws[stop])
    if stop > 0:
        gain = (1.230174104914001 ** 2) ** stop if info["wavelet"] == "9/7" else 1.0
        planes = np.floor(planes[:, :h, :w].astype(np.float64) / gain + 0.5).astype(np.int32)
    out = np.empty((h, w, 3), np.uint8)
    use_grain = gs is not None and grain > 0 and stop == 0
    _finish(np.ascontiguousarray(planes[:, :h, :w]), h, w, info["prescale"] + info["extra"],
            gs if gs is not None else np.zeros((1, 1, 4), np.int64), use_grain, float(grain), out)
    return out


if __name__ == "__main__":
    import time

    import codec_v1
    import codec_v5

    rng = np.random.default_rng(21)
    a = rng.integers(-3000, 3000, (61, 77)).astype(np.int32)
    maps = [rng.integers(0, len(DIRS), (4, 5)).astype(np.int32),
            rng.integers(0, len(DIRS), (2, 3)).astype(np.int32)]
    b = a.copy()
    forward6(b, 3, STEPS_97, maps)
    inverse6(b, 3, STEPS_97, maps)
    assert np.array_equal(a, b), "directional lifting must be exactly reversible"
    for shape in [(1, 1, 3), (5, 70, 3), (37, 53, 3), (300, 260, 3), (513, 771, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(decode(encode(img, 100)), img), shape
        data = encode(img, 60, denoise=1.0, grain=True, masking=True)
        for k in range(1, layer_count(data) + 1):
            decode(data, k)
    print("Directional lifting reversible; lossless round trips and progressive decodes verified.")
    img = codec_v1.make_test_image()
    for q in (90, 75, 50):
        d5, d6 = codec_v5.encode(img, q), encode(img, q)
        print(f"quality {q}: v5 {len(d5):>8,} bytes   v6 {len(d6):>8,} bytes")
    d = encode(img, 75)
    decode(d)
    t0 = time.perf_counter(); decode(d); t1 = time.perf_counter(); encode(img, 75); t2 = time.perf_counter()
    print(f"v6 decode {(t1 - t0) * 1e3:.1f} ms, encode {(t2 - t1) * 1e3:.1f} ms")
