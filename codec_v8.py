"""
FUL v8: v7 plus an encoder-guided post-filter and learned starting
probabilities ("priors") for the entropy coder.

Noise estimator (encoder update, v8.1; no format change):
  Automatic denoising now measures noise only in the image's flattest
  regions, where any remaining fine variation must be noise. It catches the
  slightly blurred noise phone cameras produce, which v4's estimator (finest
  wavelet band only) missed. Images with no flat regions fall back to the old
  estimator. Measured on simulated low-light phone photos: -63% vs JPEG at
  equal closeness to the clean scene (was -56%); clean photos are left alone.

Post-filter (edge-aware deringing):
  After decoding, an edge-aware (bilateral) filter smooths each pixel with
  neighbours of similar brightness only, cleaning the faint ringing and
  mottling that quantisation leaves near edges without blurring across them.
  The encoder has the original image, so for every 32x32 block it tries the
  filter at four strengths (or off) and stores the best choice; the decoder
  just applies it. Integer arithmetic with lookup tables keeps the GPU
  decoder bit-identical. Not used when grain is regenerated (the encoder
  cannot judge random grain against the original).

v7's adaptive probability models start every code-block from the same generic
guess and must learn the image's statistics from scratch, which costs bits
(and is why extra contexts used to lose). v8's decoder ships a small table of
starting probabilities for every model, learned from natural photos and
indexed by plane (brightness or colour) and wavelet level. Models start close
to the right answer and also start more confident (they adapt more slowly at
first). Nothing extra is stored in the file and pixels are identical to v7's:
it is a pure entropy-coding gain.

The power-law idea from the research panel motivated this: natural images
have predictable statistics at every scale, so they can be learned once.

Priors are trained with `python codec_v8.py --train` (on the tuning photos)
and stored in priors_v8.npy next to this file.
"""
from __future__ import annotations

import struct

import numpy as np

from codec_v2 import _pool, njit

try:
    from numba import prange
except ImportError:
    prange = range
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
VERSION = 8
# magic, version, width, height, quality, flags, levels, precision, base, dq eta x100
HEADER = struct.Struct("<4sBIIBBBBfB")
FLAG_PERCEPTUAL = 1
FLAG_GRAIN = 2
FLAG_97 = 4
FLAG_DIR = 8
FLAG_DQ = 16
FLAG_CUSTOM_WEIGHTS = 32
FLAG_FILTER = 64

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
def _encode_detail8(a, levels, j, orient, sy, sh, prior, cnt0):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, orient)
    n = sh * bw
    starts = np.empty(4 * n + 1024, np.int64)
    freqs = np.empty(4 * n + 1024, np.int64)
    raw = np.zeros(6 * n + 1024, np.uint8)
    st = np.zeros(3, np.int64)
    stack = np.empty((64, 3), np.int64)
    cdf, cnt, nsym = _models_init8(prior, cnt0)
    _region7(a, by0 + sy, bx0, sh, bw, by0, py0, px0, ph, pw, True, st, starts, freqs,
             raw, np.zeros(1, np.uint8), 0, np.zeros(4, np.int64), cdf, cnt, nsym, stack)
    return _pack(starts, freqs, st[0], raw, st[1])


@njit(cache=True, nogil=True)
def _decode_detail8(a, levels, j, orient, sy, sh, buf, prior, cnt0):
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
    cdf, cnt, nsym = _models_init8(prior, cnt0)
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
                raise ValueError("FUL v8 file is truncated")
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
# Post-filter
# --------------------------------------------------------------------------
FILTER_SIGMAS = (2.0, 4.0, 7.0, 11.0)       # strengths 1-4 (range sigma, brightness levels); 0 = off
FILTER_SHIFT = 5                             # 32x32 blocks
SPATIAL_LUT = np.array([[int(round(64 * np.exp(-(dy * dy + dx * dx) / 2.0))) for dx in range(-2, 3)]
                        for dy in range(-2, 3)], np.int64)
RANGE_LUT = np.array([[64] + [0] * 255] + [[int(round(64 * np.exp(-(d * d) / (2 * s * s)))) for d in range(256)]
                      for s in FILTER_SIGMAS], np.int64)


@njit(cache=True, parallel=True)
def _bilateral(img, smap, spatial, rlut, out):
    """5x5 edge-aware filter, all integer: weight = spatial x brightness-similarity."""
    h, w = img.shape[0], img.shape[1]
    for yp in prange(h):
        y = np.int64(yp)
        for x in range(w):
            k = smap[y >> FILTER_SHIFT, x >> FILTER_SHIFT]
            if k == 0:
                for c in range(3):
                    out[y, x, c] = img[y, x, c]
                continue
            yc = (77 * np.int64(img[y, x, 0]) + 150 * np.int64(img[y, x, 1]) + 29 * np.int64(img[y, x, 2])) >> 8
            n0 = 0
            n1 = 0
            n2 = 0
            den = 0
            for dy in range(-2, 3):
                yy = min(h - 1, max(0, y + dy))
                for dx in range(-2, 3):
                    xx = min(w - 1, max(0, x + dx))
                    r = np.int64(img[yy, xx, 0])
                    g = np.int64(img[yy, xx, 1])
                    b = np.int64(img[yy, xx, 2])
                    yn = (77 * r + 150 * g + 29 * b) >> 8
                    wgt = spatial[dy + 2, dx + 2] * rlut[k, abs(yn - yc)]
                    n0 += wgt * r
                    n1 += wgt * g
                    n2 += wgt * b
                    den += wgt
            out[y, x, 0] = (n0 + (den >> 1)) // den
            out[y, x, 1] = (n1 + (den >> 1)) // den
            out[y, x, 2] = (n2 + (den >> 1)) // den


def post_filter(img, smap):
    out = np.empty_like(img)
    _bilateral(np.ascontiguousarray(img), np.ascontiguousarray(smap.astype(np.int64)),
               SPATIAL_LUT, RANGE_LUT, out)
    return out


def choose_filter(original, decoded):
    """Per 32x32 block, the strength (0 = off) with the smallest error."""
    h, w = decoded.shape[:2]
    bh, bw = -(-h // 32), -(-w // 32)
    o = original.astype(np.float64)
    best = np.zeros((bh, bw), np.int64)
    best_err = None
    for k in range(len(FILTER_SIGMAS) + 1):
        f = decoded if k == 0 else post_filter(decoded, np.full((bh, bw), k, np.int64))
        e = ((f.astype(np.float64) - o) ** 2).sum(axis=2)
        pad = np.zeros((bh * 32, bw * 32)); pad[:h, :w] = e
        err = pad.reshape(bh, 32, bw, 32).sum(axis=(1, 3))
        if best_err is None:
            best_err = err
        else:
            better = err < best_err
            best[better] = k
            best_err = np.where(better, err, best_err)
    return best


@njit(cache=True, nogil=True)
def _encode_map(vals, init_cdf):
    n = vals.size + 16
    starts = np.empty(2 * n, np.int64)
    freqs = np.empty(2 * n, np.int64)
    raw = np.zeros(4 * n + 64, np.uint8)
    st = np.zeros(3, np.int64)
    cdf, cnt, nsym = _models_init7(init_cdf)
    prev = 0
    for i in range(vals.size):
        _emit(st, starts, freqs, cdf, cnt, nsym, M_MAG + prev, vals[i])   # context: previous block
        prev = vals[i]
    return _pack(starts, freqs, st[0], raw, st[1])


@njit(cache=True, nogil=True)
def _decode_map(buf, count, init_cdf):
    out = np.zeros(count, np.int64)
    if buf.size < 4:
        return out
    nwords = (np.int64(buf[0]) | (np.int64(buf[1]) << 8) | (np.int64(buf[2]) << 16)
              | (np.int64(buf[3]) << 24))
    if 4 + 2 * nwords > buf.size:
        nwords = (buf.size - 4) // 2
    st = np.zeros(3, np.int64)
    x4 = np.zeros(4, np.int64)
    cdf, cnt, nsym = _models_init7(init_cdf)
    _rans_start(buf, nwords, x4, st)
    prev = 0
    for i in range(count):
        v = _get(buf, nwords, x4, st, cdf, cnt, nsym, M_MAG + prev)
        out[i] = min(4, v)
        prev = out[i]
    return out


# --------------------------------------------------------------------------
# Priors: learned starting probabilities
# --------------------------------------------------------------------------
import os as _os

PRIOR_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "priors_v8.npy")
PRIOR_CNT = 0           # start from the priors but adapt at full speed (measured best of 0, 16, 32)
_PRIORS = None


@njit(cache=True, nogil=True)
def _models_init8(prior, cnt0):
    cdf, cnt, nsym = _models_init7(INIT_CDF)
    for m in range(prior.shape[0]):
        if prior[m, nsym[m]] == PROB_ONE:          # a trained row is present
            for i in range(nsym[m] + 1):
                cdf[m, i] = prior[m, i]
            cnt[m] = cnt0
    return cdf, cnt, nsym


def _flat_table():
    cdf, _, nsym = _models_init7(INIT_CDF)
    return np.asarray(cdf), np.asarray(nsym)


def prior_for(p, j):
    global _PRIORS
    if _PRIORS is None:
        if _os.path.exists(PRIOR_PATH):
            _PRIORS = np.load(PRIOR_PATH)
        else:                                     # untrained: behave exactly like v7
            flat, _ = _flat_table()
            _PRIORS = np.broadcast_to(flat, (2, 5) + flat.shape).copy()
    return np.ascontiguousarray(_PRIORS[min(p, 1), min(j, 4)])


def _counts_to_cdf(counts, n):
    f = counts[:n].astype(np.float64) + 0.5
    f = np.maximum(1, np.floor(f / f.sum() * (PROB_ONE - n))).astype(np.int64)
    f[np.argmax(f)] += PROB_ONE - f.sum()
    row = np.zeros(NSYM + 1, np.int64)
    row[1:n + 1] = np.cumsum(f)
    return row


def train_priors(images, quality_values=(40, 60, 75, 90)):
    """Count tokens, signs and first low bits per model on training images,
    using the same contexts the coder uses, and turn them into CDF rows."""
    from codec_v4 import _band_geometry as geom
    flat, nsym = _flat_table()
    counts = np.zeros((2, 5, flat.shape[0], NSYM))
    def nbin(v, edges):
        return np.digitize(v, edges)
    for img in images:
        for q in quality_values:
            info_coef = _coefficients_for_training(img, q)
            if info_coef is None:
                continue
            coef, L = info_coef
            hs, ws = _level_dims(img.shape[0], img.shape[1], L)
            for p in range(3):
                for j in range(1, L + 1):
                    for o in range(3):
                        by0, bx0, bh, bw, py0, px0, ph, pw = geom(hs, ws, j, L, o)
                        if bh <= 0 or bw <= 0:
                            continue
                        B = coef[p, by0:by0 + bh, bx0:bx0 + bw].astype(np.int64)
                        A = np.abs(B)
                        left = np.zeros_like(A); left[:, 1:] = A[:, :-1]
                        up = np.zeros_like(A); up[1:, :] = A[:-1, :]
                        if ph > 0 and pw > 0:
                            P_ = np.abs(coef[p, py0:py0 + ph, px0:px0 + pw]).astype(np.int64)
                            par = np.minimum(P_[np.minimum(np.arange(bh) >> 1, ph - 1)][:, np.minimum(np.arange(bw) >> 1, pw - 1)], 2)
                        else:
                            par = np.zeros_like(A)
                        ctx = M_MAG + nbin(left + up, [1, 2, 3, 5]) * 3 + par
                        tokv = np.where(A < 4, A, np.minimum(15, 2 + np.floor(np.log2(np.maximum(A, 1))))).astype(np.int64)
                        K = counts[min(p, 1), min(j, 4)]
                        np.add.at(K, (ctx.ravel(), tokv.ravel()), 1)
                        sl = np.ones_like(B); sl[:, 1:] = np.sign(B[:, :-1]) + 1
                        su = np.ones_like(B); su[1:, :] = np.sign(B[:-1, :]) + 1
                        nz = B != 0
                        np.add.at(K, ((M_SIGN + sl * 3 + su)[nz], (B < 0)[nz].astype(np.int64)), 1)
                        big = (A >= 4) & (A < (1 << 13))
                        if big.any():
                            n = np.floor(np.log2(A[big])).astype(np.int64)
                            np.add.at(K, (M_LOW + n, (A[big] >> (n - 1)) & 1), 1)
    table = np.broadcast_to(flat, (2, 5) + flat.shape).copy()
    for pc in range(2):
        for jc in range(1, 5):
            for m in range(M_MAG, flat.shape[0]):
                if counts[pc, jc, m].sum() >= 50:
                    table[pc, jc, m] = _counts_to_cdf(counts[pc, jc, m], int(nsym[m]))
    return table


def _coefficients_for_training(img, q):
    """The exact quantised coefficients the encoder would code (v7/v8 pipeline)."""
    rgb = np.ascontiguousarray(img[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    L = levels_for(w, h)
    if L < 1:
        return None
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    co = r - b; t = b + (co >> 1); cg = g - t; y = t + (cg >> 1)
    pix = np.stack([y, co, cg])
    wts = standard_weights(L, True, True, PRESCALE_97)
    mp = np.full(((h + 15) >> BLOCK_SHIFT, (w + 15) >> BLOCK_SHIFT), 2, np.int32)
    dirs = choose_directions(pix[0], L, STEPS_97, PRESCALE_97, EXTRA_97)
    out = []
    for p in range(3):
        a = np.ascontiguousarray(pix[p].astype(np.int32) << (PRESCALE_97 + EXTRA_97))
        forward6(a, L, STEPS_97, dirs)
        a = (a + (1 << (EXTRA_97 - 1))) >> EXTRA_97
        out.append(quantise_dq(a, L, quality_to_base(q), wts, p, mp,
                               np.zeros((L + 2, 3)), RDO_LAMBDA, DQ_ETA))
    return np.stack(out), L


# --------------------------------------------------------------------------
# Noise estimation (v8.1)
# --------------------------------------------------------------------------
_FLAT_CAL = {}


def _flat_level(y, B=16, pct=10):
    """Fine-scale variation in the quietest blocks, and how flat those blocks are."""
    from scipy.ndimage import gaussian_filter
    r = y - gaussian_filter(y, 1.0)
    sm = gaussian_filter(y, 2.0)
    gx = np.abs(np.diff(sm, axis=1, append=sm[:, -1:]))
    gy = np.abs(np.diff(sm, axis=0, append=sm[-1:]))
    H, W = y.shape
    H2, W2 = H // B * B, W // B * B
    rb = r[:H2, :W2].reshape(H2 // B, B, W2 // B, B).std(axis=(1, 3))
    gb = (gx + gy)[:H2, :W2].reshape(H2 // B, B, W2 // B, B).mean(axis=(1, 3))
    sel = gb <= np.percentile(gb, 25)
    if not sel.any():
        return 0.0, np.inf
    return float(np.percentile(rb[sel], pct)), float(np.median(gb[sel]))


def estimate_noise_v2(rgb: np.ndarray) -> float:
    """Noise level (brightness levels) from the image's flattest regions."""
    from scipy.ndimage import gaussian_filter
    rgb = np.asarray(rgb)[..., :3].astype(np.float64)
    if min(rgb.shape[:2]) < 64:
        return 0.0
    if rgb.shape[0] * rgb.shape[1] > 4_000_000:            # large photos: measure a centre crop
        h, w = rgb.shape[:2]
        rgb = rgb[h // 2 - 1000:h // 2 + 1000, w // 2 - 1000:w // 2 + 1000]
    y = (rgb[..., 0] + 2 * rgb[..., 1] + rgb[..., 2]) / 4
    if not _FLAT_CAL:
        n = np.random.default_rng(0).normal(0, 1, (512, 512))
        _FLAT_CAL["k"] = (_flat_level(n)[0] + _flat_level(gaussian_filter(n, 0.6))[0]) / 2
    level, flatness = _flat_level(y)
    if flatness < 2.5:
        return level / _FLAT_CAL["k"]
    return estimate_noise(np.clip(rgb, 0, 255).astype(np.uint8)) * (1.4 / 3.0)   # no flat regions: old method


def auto_denoise_strength_v2(noise: float) -> float:
    """Clean photos read below ~1.4; low-light phone noise 2.4-3.6."""
    return float(np.clip((noise - 1.6) / 1.6, 0.0, 1.0))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def encode(rgb: np.ndarray, quality: int = 75, perceptual: bool = True,
           denoise: float | str = "auto", grain: bool | str = "auto",
           masking: bool = False, rdo: bool = True, directional: bool = True,
           dq: bool = True, lam: float = RDO_LAMBDA, eta: float = DQ_ETA,
           weights: np.ndarray | None = None, post_filter_on: bool = True) -> bytes:
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
        denoise = auto_denoise_strength_v2(estimate_noise_v2(rgb))
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
        jj = levels + 1 - layer
        return _encode_detail8(coef[p], levels, jj, orient, sy, sh_, prior_for(p, jj), PRIOR_CNT)

    payload = list(_pool().map(work, chunks))
    flags = ((FLAG_PERCEPTUAL if perceptual else 0) | (FLAG_GRAIN if grain else 0)
             | (FLAG_97 if is97 else 0) | (FLAG_DIR if directional else 0)
             | (FLAG_DQ if dq else 0) | (FLAG_CUSTOM_WEIGHTS if custom else 0))
    weight_bytes = wts.astype("<f4").tobytes() if custom else b""
    body = b"".join(c.tobytes() for c in payload)

    def assemble(flags_, filter_bytes):
        header = HEADER.pack(MAGIC, VERSION, w, h, quality, flags_, levels,
                             prescale | (extra << 4), base, eta_byte)
        lengths = [len(c) for c in payload] + ([len(filter_bytes)] if filter_bytes else [])
        return header + weight_bytes + _varints(lengths) + body + filter_bytes

    data = assemble(flags, b"")
    if post_filter_on and not grain and not lossless:
        smap = choose_filter(rgb, decode(data, grain=0))
        if smap.any():
            fb = _encode_map(smap.ravel(), INIT_CDF).tobytes()
            data = assemble(flags | FLAG_FILTER, fb)
    return data


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL v8 image")
    magic, version, w, h, quality, flags, levels, precision, base, eta_byte = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Not a FUL v8 file (version {version})")
    prescale, extra = precision & 15, precision >> 4
    if levels != levels_for(w, h) or prescale > 4 or extra > 6:
        raise ValueError("Corrupt FUL v8 header")
    return {"version": 8, "width": w, "height": h, "quality": quality,
            "perceptual": bool(flags & FLAG_PERCEPTUAL), "grain": bool(flags & FLAG_GRAIN),
            "wavelet": "9/7" if flags & FLAG_97 else "5/3",
            "directional": bool(flags & FLAG_DIR), "dq": bool(flags & FLAG_DQ),
            "custom_weights": bool(flags & FLAG_CUSTOM_WEIGHTS), "eta": eta_byte / 100.0,
            "filter": bool(flags & FLAG_FILTER),
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
    lengths, base = _read_varints(data, pos, len(chunks) + (1 if info["filter"] else 0))
    info["filter_len"] = lengths.pop() if info["filter"] else 0
    lengths = np.asarray(lengths, np.int64)
    starts = base + np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    if len(chunks) and starts[-1] + lengths[-1] + info["filter_len"] > len(data):
        raise ValueError("FUL v8 file is truncated")
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
                jj = levels + 1 - lay
                _decode_detail8(coef[p], levels, jj, orient, sy, sh_, piece, prior_for(p, jj), PRIOR_CNT)

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
    if info["filter"] and stop == 0:
        out = post_filter(out, filter_map(data, info))
    return out


def filter_map(data: bytes, info: dict) -> np.ndarray:
    """The encoder's per-block filter strengths (0 = off), from the end of the file."""
    h, w = info["height"], info["width"]
    bh, bw = -(-h // 32), -(-w // 32)
    n = info["filter_len"]
    buf = np.frombuffer(data, np.uint8, n, len(data) - n).copy()
    return _decode_map(buf, bh * bw, INIT_CDF).reshape(bh, bw)


if __name__ == "__main__":
    import sys

    import codec_v1
    import codec_v7

    if "--train" in sys.argv:
        from skimage import data as skdata
        imgs = [np.ascontiguousarray(getattr(skdata, n)()[..., :3])
                for n in ("astronaut", "coffee", "chelsea", "rocket")]
        np.save(PRIOR_PATH, train_priors(imgs))
        print(f"Saved {PRIOR_PATH}")
        sys.exit()
    rng = np.random.default_rng(41)
    for shape in [(1, 1, 3), (5, 70, 3), (37, 53, 3), (300, 260, 3), (513, 771, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(decode(encode(img, 100)), img), shape
        data = encode(img, 60, denoise=1.0, grain=True, masking=True)
        for k in range(1, layer_count(data) + 1):
            decode(data, k)
    print("Lossless round trips and progressive decodes verified.")
    img = codec_v1.make_test_image()
    for q in (90, 75, 50):
        d7, d8 = codec_v7.encode(img, q, denoise=0.0), encode(img, q, denoise=0.0)
        same = np.array_equal(codec_v7.decode(d7, grain=0), decode(d8, grain=0))
        d8n = encode(img, q, post_filter_on=False, denoise=0.0)
        same = np.array_equal(codec_v7.decode(d7, grain=0), decode(d8n, grain=0))
        print(f"quality {q}: v7 {len(d7):>8,} bytes | v8 without filter {len(d8n):>8,} "
              f"(identical pixels: {same}) | v8 {len(d8):>8,} (filter: {read_header(d8)['filter']})")
