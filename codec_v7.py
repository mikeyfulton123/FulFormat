"""
FUL v7: v6 plus better coding of signs and low bits, a compact header, and
(optionally) dependent quantisation.

Measured before building (on v6's real coefficients, quality 75, 6 images):
  sign context from the left and upper neighbours' signs  -1.31% file size
  coding the first low-order bit with a model              -0.30%
  "edge-aware" richer contexts (orientation, diagonal,
  cross-channel)                                           +0.4% to +1.9% (worse:
  more contexts dilute each model's statistics), so they are not used.

New in v7:
  * Signs are coded by the rANS coder in 9 contexts (sign of the left and of
    the upper neighbour), instead of 1 raw bit each.
  * The first low-order bit of every value >= 4 is coded in a model per
    magnitude class; the remaining low bits stay raw.
  * Compact header: standard step weights are rebuilt by the decoder instead
    of stored (custom weights are still possible), and block lengths are
    variable-length integers (LEB128).
  * Dependent quantisation (on by default, flag in the header): two quantisers
    whose reconstruction points interleave, selected by a 4-state machine
    driven by the parity of each stored value (the scheme used in VVC). The
    encoder chooses values with a Viterbi trellis search; the decoder only
    tracks the state while dequantising, in raster order per band.

Measured at equal SSIM against JPEG (tuning photos / held-out images):
  v6 -30.9% / -22.8%;  v7 without DQ -33.0% / -24.8%;  v7 -34.7% / -28.4%.
  Confirmed at equal PSNR: v6 -43.0%, v7 -44.6%.

Run for a self-test:  python codec_v7.py
"""
from __future__ import annotations

import struct

import numpy as np

from codec_v2 import _pool, njit
from codec_v3 import BLOCK_SHIFT, DELTA, MULTS, _masking_map, quality_to_base
from codec_v4 import (
    INIT_CDF, LEAF, M_MAG, NSYM, PROB_ONE, _adapt, _band_geometry, _coef_step,
    _dequantise, _emit, _get, _grain_sigmas, _level_dims, _node_model,
    _rans_start, _raw_get, _raw_put, _shrink_thresholds, auto_denoise_strength,
    estimate_noise, fast,
)
from codec_v5 import (
    EXTRA_97, GRAIN_CELL, PRESCALE_97, STEPS_53, STEPS_97, _finish, _pack,
    band_gains, chunk_list, levels_for, noise_gains, perceptual_weights,
)
from codec_v6 import (
    DIRS, RDO_LAMBDA, _decode_ll6, _encode_ll6, _scale_all, choose_directions,
    dir_map_shapes, forward6, inverse6, quantise,
)

MAGIC = b"FUL\x00"
VERSION = 7
# magic, version, width, height, quality, flags, levels, precision, base, dq eta x100
HEADER = struct.Struct("<4sBIIBBBBfB")
FLAG_PERCEPTUAL = 1
FLAG_GRAIN = 2
FLAG_97 = 4
FLAG_DIR = 8
FLAG_DQ = 16
FLAG_CUSTOM_WEIGHTS = 32

M_SIGN = 29            # 9 binary models: left sign x upper sign
M_LOW = 38             # 13 binary models: first low bit, per magnitude class
N_MODELS7 = 51
DQ_ETA = 1.0           # dependent-quantisation step (measured best of 0.5-1.2)
DQ_TRANS = np.array([[0, 2], [2, 0], [1, 3], [3, 1]], np.int64)


# --------------------------------------------------------------------------
# Models with the extra sign and low-bit contexts
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _models_init7(init_cdf):
    cdf = np.zeros((N_MODELS7, NSYM + 1), np.int64)
    cnt = np.zeros(N_MODELS7, np.int64)
    nsym = np.zeros(N_MODELS7, np.int64)
    for m in range(N_MODELS7):
        if m < M_MAG or m >= M_SIGN:
            nsym[m] = 2
            cdf[m, 1] = PROB_ONE >> 1
            cdf[m, 2] = PROB_ONE
        else:
            nsym[m] = NSYM
            for i in range(NSYM + 1):
                cdf[m, i] = init_cdf[i]
    return cdf, cnt, nsym


@fast
def _ctx7(a, yy, xx, sy0, bx0, by0, py0, px0, ph, pw):
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


@fast
def _sign_ctx(a, yy, xx, sy0, bx0):
    sl = 1
    su = 1
    if xx > bx0:
        v = a[yy, xx - 1]
        sl = 0 if v < 0 else (2 if v > 0 else 1)
    if yy > sy0:
        v = a[yy - 1, xx]
        su = 0 if v < 0 else (2 if v > 0 else 1)
    return M_SIGN + sl * 3 + su


@njit(cache=True, nogil=True)
def _put_mag(st, starts, freqs, raw, cdf, cnt, nsym, m, v):
    """Token as before; the first low bit through a model, the rest raw."""
    if v < 4:
        _emit(st, starts, freqs, cdf, cnt, nsym, m, v)
        return
    n = 0
    t = v
    while t > 1:
        t >>= 1
        n += 1
    if n > 12:
        _emit(st, starts, freqs, cdf, cnt, nsym, m, 15)
        _raw_put(st, raw, ((n - 13) << n) | (v - (1 << n)), 5 + n)
        return
    _emit(st, starts, freqs, cdf, cnt, nsym, m, 2 + n)
    _emit(st, starts, freqs, cdf, cnt, nsym, M_LOW + n, (v >> (n - 1)) & 1)
    if n > 1:
        _raw_put(st, raw, v & ((1 << (n - 1)) - 1), n - 1)


@njit(cache=True, nogil=True)
def _get_mag(buf, nwords, x4, st, raw, cdf, cnt, nsym, m):
    tok = _get(buf, nwords, x4, st, cdf, cnt, nsym, m)
    if tok < 4:
        return tok
    if tok == 15:
        n = 13 + _raw_get(raw, st, 5)
        return (1 << n) + _raw_get(raw, st, n)
    n = tok - 2
    top = _get(buf, nwords, x4, st, cdf, cnt, nsym, M_LOW + n)
    rest = _raw_get(raw, st, n - 1) if n > 1 else 0
    return (1 << n) | (top << (n - 1)) | rest


@njit(cache=True, nogil=True)
def _region7(a, sy0, bx0, sh, bw, by0, py0, px0, ph, pw, encode,
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
                    m = _ctx7(a, yy, xx, sy0, bx0, by0, py0, px0, ph, pw)
                    sm = _sign_ctx(a, yy, xx, sy0, bx0)
                    if encode:
                        v = np.int64(a[yy, xx])
                        _put_mag(st, starts, freqs, raw, cdf, cnt, nsym, m, abs(v))
                        if v != 0:
                            _emit(st, starts, freqs, cdf, cnt, nsym, sm, 1 if v < 0 else 0)
                    else:
                        v = _get_mag(buf, nwords, x4, st, raw, cdf, cnt, nsym, m)
                        if v != 0 and _get(buf, nwords, x4, st, cdf, cnt, nsym, sm):
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
def _encode_detail7(a, levels, j, orient, sy, sh, init_cdf):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, orient)
    n = sh * bw
    starts = np.empty(4 * n + 1024, np.int64)
    freqs = np.empty(4 * n + 1024, np.int64)
    raw = np.zeros(6 * n + 1024, np.uint8)
    st = np.zeros(3, np.int64)
    stack = np.empty((64, 3), np.int64)
    cdf, cnt, nsym = _models_init7(init_cdf)
    _region7(a, by0 + sy, bx0, sh, bw, by0, py0, px0, ph, pw, True, st, starts, freqs,
             raw, np.zeros(1, np.uint8), 0, np.zeros(4, np.int64), cdf, cnt, nsym, stack)
    return _pack(starts, freqs, st[0], raw, st[1])


@njit(cache=True, nogil=True)
def _decode_detail7(a, levels, j, orient, sy, sh, buf, init_cdf):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, orient)
    if buf.size < 4:
        return
    nwords = (np.int64(buf[0]) | (np.int64(buf[1]) << 8) | (np.int64(buf[2]) << 16)
              | (np.int64(buf[3]) << 24))
    if 4 + 2 * nwords > buf.size:
        nwords = (buf.size - 4) // 2
    raw = buf[4 + 2 * nwords:]
    st = np.zeros(3, np.int64)
    x4 = np.zeros(4, np.int64)
    stack = np.empty((64, 3), np.int64)
    cdf, cnt, nsym = _models_init7(init_cdf)
    _rans_start(buf, nwords, x4, st)
    st[2] = 0
    dummy = np.zeros(1, np.int64)
    _region7(a, by0 + sy, bx0, sh, bw, by0, py0, px0, ph, pw, False, st, dummy, dummy,
             raw, buf, nwords, x4, cdf, cnt, nsym, stack)


# --------------------------------------------------------------------------
# Dependent quantisation
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _dq_recon(state, k, eta):
    """Reconstruction in units of the normal step. States 0-1 use Q0 (even
    multiples of eta), states 2-3 use Q1 (odd multiples, plus zero)."""
    if k == 0:
        return 0.0
    a = abs(k)
    r = 2.0 * a * eta if state < 2 else (2.0 * a - 1.0) * eta
    return r if k > 0 else -r


@njit(cache=True, nogil=True)
def _dq_trellis(U, by0, bx0, bh, bw, eta, lam, cost, out):
    """Viterbi search over the 4-state machine, raster order within the band.
    cost[t] = estimated bits of token t (sign and low bits added here)."""
    n = bh * bw
    if n == 0:
        return
    back_state = np.empty((n, 4), np.int8)
    back_k = np.empty((n, 4), np.int32)
    J = np.full(4, 1e30)
    J[0] = 0.0
    trans = DQ_TRANS
    i = 0
    for y in range(by0, by0 + bh):
        for x in range(bx0, bx0 + bw):
            u = np.float64(U[y, x])
            au = abs(u)
            sg = 1 if u >= 0 else -1
            NJ = np.full(4, 1e30)
            for s in range(4):
                if J[s] >= 1e29:
                    continue
                if s < 2:
                    k0 = int(au / (2.0 * eta))
                else:
                    k0 = int((au / eta + 1.0) / 2.0)
                for cand in range(3):
                    ak = 0 if cand == 0 else (k0 if cand == 1 else k0 + 1)
                    if cand == 2 and ak == 0:
                        continue
                    if cand == 1 and ak == 0:
                        continue
                    k = sg * ak
                    d = u - _dq_recon(s, k, eta)
                    if ak < 4:
                        bits = cost[ak]
                    else:
                        nb = 0
                        t = ak
                        while t > 1:
                            t >>= 1
                            nb += 1
                        bits = cost[min(15, 2 + nb)] + nb
                    if ak:
                        bits += 1.0
                    total = J[s] + d * d + lam * bits
                    ns = trans[s, ak & 1]
                    if total < NJ[ns]:
                        NJ[ns] = total
                        back_state[i, ns] = s
                        back_k[i, ns] = k
            J = NJ
            i += 1
    s = 0
    for t in range(1, 4):
        if J[t] < J[s]:
            s = t
    for i in range(n - 1, -1, -1):
        y = by0 + i // bw
        x = bx0 + i % bw
        out[y, x] = back_k[i, s]
        s = back_state[i, s]


def _dq_token_costs(U, by0, bx0, bh, bw, eta):
    """Rough per-token bit costs for the trellis, from a plain rounding pass."""
    B = np.abs(U[by0:by0 + bh, bx0:bx0 + bw]) / (2.0 * eta)
    k = np.floor(B + 0.3).astype(np.int64).ravel()
    t = np.where(k < 4, k, np.minimum(15, 2 + np.floor(np.log2(np.maximum(k, 1))))).astype(int)
    cnt = np.bincount(t, minlength=16) + 0.5
    return -np.log2(cnt / cnt.sum())


def quantise_dq(a, levels, base, wts, p, mp, T, lam, eta):
    U = _scale_all(a, levels, base, wts, p, MULTS, mp, T)
    q = np.trunc(U).astype(np.int32)                     # LL keeps plain quantisation
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels, 0, -1):
        for o in range(3):
            by0, bx0, bh, bw, *_ = _band_geometry(hs, ws, j, levels, o)
            if bh <= 0 or bw <= 0:
                continue
            cost = _dq_token_costs(U, by0, bx0, bh, bw, eta)
            _dq_trellis(U, by0, bx0, bh, bw, eta, lam, cost, q)
    return q


@njit(cache=True, nogil=True)
def _dq_dequant_band(a, y0, y1, x0, x1, shift, s_base, mults, mp, eta, trans):
    state = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            k = np.int64(a[y, x])
            if k != 0:
                s = _coef_step(s_base, mults, mp, (y - y0) << shift, (x - x0) << shift)
                r = _dq_recon(state, k, eta) * s
                a[y, x] = int(np.floor(r + 0.5))
            state = trans[state, abs(k) & 1]


@njit(cache=True, nogil=True)
def _dequantise_dq(a, levels, base, wts, p, mults, mp, last_level, eta, trans):
    """Detail bands through the state machine; the LL band as before."""
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels, last_level - 1, -1):
        for o in range(3):
            by0, bx0, bh, bw, _, _, _, _ = _band_geometry(hs, ws, j, levels, o)
            sb = base * wts[p, j, 1 if o == 2 else 0] / (1 << (j - 1))
            _dq_dequant_band(a, by0, by0 + bh, bx0, bx0 + bw, j, sb, mults, mp, eta, trans)
    s_ll = base * wts[p, levels + 1, 0] / (1 << levels)
    for y in range(hs[levels]):
        for x in range(ws[levels]):
            qv = a[y, x]
            if qv != 0:
                s = _coef_step(s_ll, mults, mp, y << levels, x << levels)
                v = int((abs(qv) + DELTA) * s + 0.5)
                a[y, x] = v if qv > 0 else -v


# --------------------------------------------------------------------------
# Compact header pieces
# --------------------------------------------------------------------------
def _varints(values) -> bytes:
    out = bytearray()
    for v in values:
        v = int(v)
        while True:
            b = v & 0x7F
            v >>= 7
            out.append(b | (0x80 if v else 0))
            if not v:
                break
    return bytes(out)


def _read_varints(data: bytes, pos: int, count: int):
    vals = []
    for _ in range(count):
        v = shift = 0
        while True:
            if pos >= len(data):
                raise ValueError("FUL v7 file is truncated")
            b = data[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        vals.append(v)
    return vals, pos


def standard_weights(levels, perceptual, is97, prescale):
    wp = perceptual_weights(levels) if perceptual else np.ones((3, levels + 2, 2))
    return (wp * band_gains(levels, is97, prescale)[None, :, :]).astype(np.float32).astype(np.float64)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def encode(rgb: np.ndarray, quality: int = 75, perceptual: bool = True,
           denoise: float | str = "auto", grain: bool | str = "auto",
           masking: bool = False, rdo: bool = True, directional: bool = True,
           dq: bool = True, lam: float = RDO_LAMBDA, eta: float = DQ_ETA,
           weights: np.ndarray | None = None) -> bytes:
    """quality 1-100 (100 = exactly lossless). dq: dependent quantisation
    (replaces the per-coefficient RDO when on). Other options as in v6."""
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    lossless = quality == 100
    perceptual = perceptual and not lossless
    rdo = rdo and not lossless
    dq = dq and not lossless
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
    eta_byte = int(round(eta * 100)) if dq else 0
    eta = eta_byte / 100.0

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
    custom = weights is not None and perceptual
    if custom:
        wts = (np.asarray(weights, dtype=np.float64) * band_gains(levels, is97, prescale)[None]
               ).astype(np.float32).astype(np.float64)
    else:
        wts = standard_weights(levels, perceptual, is97, prescale)

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
        if dq:
            coef[p] = quantise_dq(a, levels, base, wts, p, mp, T, lam, eta)
        else:
            coef[p] = quantise(a, levels, base, wts, p, mp, T, rdo, lam, 1)

    ncy, ncx = -(-h // GRAIN_CELL), -(-w // GRAIN_CELL)
    gs = np.zeros((ncy, ncx, 4), np.int64)
    if grain:
        rec = coef[0].copy()
        if dq:
            _dequantise_dq(rec, levels, base, wts, 0, MULTS, mp, 1, eta, DQ_TRANS)
        else:
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
        return _encode_detail7(coef[p], levels, levels + 1 - layer, orient, sy, sh_, INIT_CDF)

    payload = list(_pool().map(work, chunks))
    flags = ((FLAG_PERCEPTUAL if perceptual else 0) | (FLAG_GRAIN if grain else 0)
             | (FLAG_97 if is97 else 0) | (FLAG_DIR if directional else 0)
             | (FLAG_DQ if dq else 0) | (FLAG_CUSTOM_WEIGHTS if custom else 0))
    header = HEADER.pack(MAGIC, VERSION, w, h, quality, flags, levels,
                         prescale | (extra << 4), base, eta_byte)
    weight_bytes = wts.astype("<f4").tobytes() if custom else b""
    table = _varints(len(c) for c in payload)
    return header + weight_bytes + table + b"".join(c.tobytes() for c in payload)


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL v7 image")
    magic, version, w, h, quality, flags, levels, precision, base, eta_byte = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Not a FUL v7 file (version {version})")
    prescale, extra = precision & 15, precision >> 4
    if levels != levels_for(w, h) or prescale > 4 or extra > 6:
        raise ValueError("Corrupt FUL v7 header")
    return {"version": 7, "width": w, "height": h, "quality": quality,
            "perceptual": bool(flags & FLAG_PERCEPTUAL), "grain": bool(flags & FLAG_GRAIN),
            "wavelet": "9/7" if flags & FLAG_97 else "5/3",
            "directional": bool(flags & FLAG_DIR), "dq": bool(flags & FLAG_DQ),
            "custom_weights": bool(flags & FLAG_CUSTOM_WEIGHTS), "eta": eta_byte / 100.0,
            "levels": levels, "prescale": prescale, "extra": extra, "base": base}


def _layout(data: bytes, info: dict):
    levels = info["levels"]
    pos = HEADER.size
    if info["custom_weights"]:
        nw = 3 * (levels + 2) * 2
        wts = np.frombuffer(data, "<f4", nw, pos).astype(np.float64).reshape(3, levels + 2, 2)
        pos += 4 * nw
    else:
        wts = standard_weights(levels, info["perceptual"], info["wavelet"] == "9/7", info["prescale"])
    chunks = chunk_list(info["width"], info["height"], levels)
    lengths, base = _read_varints(data, pos, len(chunks))
    lengths = np.asarray(lengths, np.int64)
    starts = base + np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    if len(chunks) and starts[-1] + lengths[-1] > len(data):
        raise ValueError("FUL v7 file is truncated")
    return wts, chunks, starts, lengths, base


def layer_count(data: bytes) -> int:
    return read_header(data)["levels"] + 1


def bytes_for_layers(data: bytes, layers: int) -> int:
    info = read_header(data)
    _, chunks, starts, lengths, base = _layout(data, info)
    return int(base + sum(l for c, l in zip(chunks, lengths) if c[0] < layers))


def decode_coefficients(data: bytes, layers: int | None = None):
    """CPU stage: entropy decoding + dequantisation. Returns (coef, grain,
    info, stop, dirmaps), the same shape as v6 so the GPU decoder can use it."""
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
                _decode_detail7(coef[p], levels, levels + 1 - lay, orient, sy, sh_, piece, INIT_CDF)

        list(_pool().map(work, todo))
    stop = levels - (k - 1)
    base = float(info["base"])
    if info["dq"]:
        list(_pool().map(lambda p: _dequantise_dq(coef[p], levels, base, wts, p, MULTS, mp,
                                                  stop + 1, info["eta"], DQ_TRANS), range(3)))
    else:
        list(_pool().map(lambda p: _dequantise(coef[p], levels, base, wts, p, MULTS, mp,
                                               stop + 1), range(3)))
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
    import codec_v1
    import codec_v6

    rng = np.random.default_rng(31)
    for shape in [(1, 1, 3), (5, 70, 3), (37, 53, 3), (300, 260, 3), (513, 771, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(decode(encode(img, 100)), img), shape
        for dq in (False, True):
            data = encode(img, 60, denoise=1.0, grain=True, masking=True, dq=dq)
            for k in range(1, layer_count(data) + 1):
                decode(data, k)
    print("Lossless round trips and progressive decodes verified (with and without DQ).")
    img = codec_v1.make_test_image()
    for q in (90, 75, 50):
        d6 = codec_v6.encode(img, q)
        d7a = encode(img, q, dq=False)
        same = np.array_equal(codec_v6.decode(d6, grain=0), decode(d7a, grain=0))
        d7 = encode(img, q)
        print(f"quality {q}: v6 {len(d6):>8,} bytes | v7 without DQ {len(d7a):>8,} "
              f"(identical pixels: {same}) | v7 {len(d7):>8,}")
