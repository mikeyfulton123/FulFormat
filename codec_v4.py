"""
FUL v4: v3's perceptual wavelets with a new entropy coder, denoising and
grain regeneration.

Entropy coder (replaces v2/v3's Rice codes):
  * rANS (asymmetric numeral systems) with FOUR interleaved states. Symbols
    take turns between the four states, so a CPU can work on several at once.
  * Probabilities adapt as the data is read, but always sum to exactly 2^15,
    so decoding needs only shifts, a multiply and a small table scan: no
    division.
  * Context modelling: the probabilities used for a coefficient depend on its
    already-decoded left and upper neighbours and on its "parent" (the same
    spot in the next coarser wavelet band). Quiet neighbourhoods predict
    zeros, busy ones predict large values.
  * Values are split into a coded "token" (roughly the magnitude class) and
    raw extra bits. Raw bits and signs are close to random, so they skip the
    coder and go to a separate plain bit stream that is read directly.

Denoising (encoder only):
  * BayesShrink: the encoder estimates the noise level of each tile from its
    finest diagonal wavelet band, then shrinks every detail coefficient
    towards zero by an amount that balances noise against signal in that
    band. Noise costs bits and isn't detail anyone wants.

Grain regeneration (decoder, optional):
  * The encoder measures what it removed (noise + quantisation loss) in four
    brightness ranges and stores four numbers per tile. The decoder adds
    matching pseudo-random grain, computed per pixel from a hash of its
    position, so it is deterministic and trivially parallel (GPU-friendly).

Quality 100 is exactly lossless. Run for a self-test:  python codec_v4.py
"""
from __future__ import annotations

import struct

import numpy as np

from codec_v2 import (
    HAVE_NUMBA, TILE_SIZES, _fwd2d, _inv2d, _level_dims, _ll_pred,
    _pool, _tile_grid, levels_for_tile, njit,
)
from codec_v3 import (
    BLOCK_SHIFT, DELTA, MULTS, RDO_LAMBDA, RDO_LEAF_BITS, _masking_map,
    perceptual_weights, quality_to_base,
)

def _make_fast():
    """Decorator for small hot helpers that never allocate memory. Numba's
    _nrt=False skips the atomic reference counting it otherwise does for
    every array argument, which made each symbol ~3x slower. Falls back to
    a normal compile if this Numba version doesn't support the option."""
    if not HAVE_NUMBA:
        return njit
    try:
        @njit(nogil=True, _nrt=False)
        def _probe(a):
            return a[0]
        _probe(np.zeros(1))
        return lambda f: njit(cache=True, nogil=True, _nrt=False)(f)
    except Exception:
        return lambda f: njit(cache=True, nogil=True)(f)


fast = _make_fast()

MAGIC = b"FUL\x00"
VERSION = 4
HEADER = struct.Struct("<4sBIIBBBHf")
FLAG_PERCEPTUAL = 1
FLAG_GRAIN = 2

PROB_BITS = 15
PROB_ONE = 1 << PROB_BITS
NSYM = 16                         # token alphabet
LEAF = 4

# Model slots, reset at the start of every chunk
M_SIG = 0                         # 10 binary models: quadtree node size
M_MAG = 10                        # 15 models: 5 neighbour bins x 3 parent bins
M_LL = 25                         # 3 models: local activity in the LL band
M_MAP = 28                        # 1 model: masking map values
N_MODELS = 29


def _initial_cdf() -> np.ndarray:
    """Start every token model with a geometric guess: small values likely."""
    f = 0.5 ** np.arange(NSYM)
    f = np.maximum(1, np.floor(f / f.sum() * (PROB_ONE - NSYM))).astype(np.int64) + 0
    f[0] += PROB_ONE - f.sum()
    return np.concatenate([[0], np.cumsum(f)]).astype(np.int64)


INIT_CDF = _initial_cdf()


@fast
def _put_bit(buf, pos, bit):
    if bit:
        buf[pos >> 3] = buf[pos >> 3] | (0x80 >> (pos & 7))
    return pos + 1


@fast
def _get_bit(buf, pos):
    i = pos >> 3
    if i >= buf.size:
        return 0
    return (np.int64(buf[i]) >> (7 - (pos & 7))) & 1


# --------------------------------------------------------------------------
# Per-coefficient steps (same rules as v3) and dequantisation
# --------------------------------------------------------------------------
@fast
def _coef_step(s_base, mults, mp, py, px):
    by = min(mp.shape[0] - 1, py >> BLOCK_SHIFT)
    bx = min(mp.shape[1] - 1, px >> BLOCK_SHIFT)
    s = s_base * mults[mp[by, bx]]
    return s if s > 1.0 else 1.0


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
def _dequantise(a, levels, base, wts, p, mults, mp, last_level):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels, last_level - 1, -1):
        h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
        s0 = base * wts[p, j, 0] / (1 << (j - 1))
        s1 = base * wts[p, j, 1] / (1 << (j - 1))
        _dequant_band(a, 0, hl, wl, w, j, s0, mults, mp)
        _dequant_band(a, hl, h, 0, wl, j, s0, mults, mp)
        _dequant_band(a, hl, h, wl, w, j, s1, mults, mp)
    s_ll = base * wts[p, levels + 1, 0] / (1 << levels)
    _dequant_band(a, 0, hs[levels], 0, ws[levels], levels, s_ll, mults, mp)


# --------------------------------------------------------------------------
# Adaptive models
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _models_init(init_cdf):
    cdf = np.zeros((N_MODELS, NSYM + 1), np.int64)
    cnt = np.zeros(N_MODELS, np.int64)
    nsym = np.zeros(N_MODELS, np.int64)
    for m in range(N_MODELS):
        if m < M_MAG:
            nsym[m] = 2
            cdf[m, 0] = 0
            cdf[m, 1] = PROB_ONE >> 1
            cdf[m, 2] = PROB_ONE
        else:
            nsym[m] = NSYM
            for i in range(NSYM + 1):
                cdf[m, i] = init_cdf[i]
    return cdf, cnt, nsym


@fast
def _adapt(cdf, cnt, nsym, m, s):
    """Move the model towards symbol s. Totals stay exactly 2^15 and every
    symbol keeps a probability of at least 1/2^15."""
    n = nsym[m]
    c = cnt[m]
    rate = 4
    if c > 15:
        rate += 1
    if c > 31:
        rate += 1
    if n > 2:
        rate += 1
    for i in range(1, s + 1):              # boundaries below s move down
        cdf[m, i] += (i - cdf[m, i]) >> rate
    top = PROB_ONE - n
    for i in range(s + 1, n):              # boundaries above s move up
        cdf[m, i] += (top + i - cdf[m, i]) >> rate
    if c < 32:
        cnt[m] = c + 1


# --------------------------------------------------------------------------
# Tokens: magnitude class (coded) + extra bits (raw)
# --------------------------------------------------------------------------
@fast
def _token(v):
    """Tokens 0-3 are the values themselves. Tokens 4-14 mean 'between 2^n
    and 2^(n+1)' (n = token - 2), with the n low bits sent raw. Token 15 is an
    escape for anything larger, with n itself sent in 5 raw bits."""
    if v < 4:
        return v, 0, 0
    n = 0
    t = v
    while t > 1:
        t >>= 1
        n += 1
    if n <= 12:
        return 2 + n, n, v - (1 << n)
    return 15, 5 + n, ((n - 13) << n) | (v - (1 << n))


@fast
def _token_extra_bits(token):
    return 0 if token < 4 else (token - 2 if token < 15 else -1)


@fast
def _untoken(token, extra):
    if token < 4:
        return token
    n = token - 2
    return (1 << n) + extra


# --------------------------------------------------------------------------
# Encoder side: forward pass records symbols, then rANS runs backwards
# --------------------------------------------------------------------------
@fast
def _emit(st, starts, freqs, cdf, cnt, nsym, m, s):
    i = st[0]
    starts[i] = cdf[m, s]
    freqs[i] = cdf[m, s + 1] - cdf[m, s]
    st[0] = i + 1
    _adapt(cdf, cnt, nsym, m, s)


@fast
def _raw_put(st, raw, value, nbits):
    pos = st[1]
    for b in range(nbits - 1, -1, -1):
        pos = _put_bit(raw, pos, (value >> b) & 1)
    st[1] = pos


@fast
def _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, m, v):
    tok, nb, extra = _token(v)
    _emit(st, starts, freqs, cdf, cnt, nsym, m, tok)
    if nb:
        _raw_put(st, raw, extra, nb)


@njit(cache=True, nogil=True)
def _rans_encode(starts, freqs, n, words):
    """Four interleaved rANS states; symbol i uses state i % 4. Encoding runs
    backwards so the decoder can run forwards. Returns where the stream
    starts in `words`."""
    x = np.full(4, 1 << 16, np.int64)
    p = words.size
    for i in range(n - 1, -1, -1):
        k = i & 3
        f = freqs[i]
        xi = x[k]
        if xi >= (f << 17):
            p -= 1
            words[p] = xi & 0xFFFF
            xi >>= 16
        x[k] = ((xi // f) << PROB_BITS) + (xi % f) + starts[i]
    for k in range(3, -1, -1):
        p -= 1
        words[p] = x[k] & 0xFFFF
        p -= 1
        words[p] = x[k] >> 16
    return p


# --------------------------------------------------------------------------
# Decoder side
# --------------------------------------------------------------------------
@fast
def _word(buf, nwords, i):
    if i >= nwords:
        return 0
    return np.int64(buf[4 + 2 * i]) | (np.int64(buf[5 + 2 * i]) << 8)


@njit(cache=True, nogil=True)
def _rans_start(buf, nwords, x, st):
    for k in range(4):
        x[k] = (_word(buf, nwords, 2 * k) << 16) | _word(buf, nwords, 2 * k + 1)
    st[0] = 0          # symbol counter
    st[1] = 8          # next word


@fast
def _get(buf, nwords, x, st, cdf, cnt, nsym, m):
    k = st[0] & 3
    st[0] += 1
    xi = x[k]
    slot = xi & (PROB_ONE - 1)
    s = 0
    n = nsym[m]
    while s < n - 1 and cdf[m, s + 1] <= slot:
        s += 1
    start = cdf[m, s]
    xi = (cdf[m, s + 1] - start) * (xi >> PROB_BITS) + slot - start
    if xi < (1 << 16):
        xi = (xi << 16) | _word(buf, nwords, st[1])
        st[1] += 1
    x[k] = xi
    _adapt(cdf, cnt, nsym, m, s)
    return s


@fast
def _raw_get(raw, st, nbits):
    v = 0
    pos = st[2]
    for _ in range(nbits):
        v = (v << 1) | _get_bit(raw, pos)
        pos += 1
    st[2] = pos
    return v


@fast
def _get_value(buf, nwords, x, st, raw, cdf, cnt, nsym, m):
    tok = _get(buf, nwords, x, st, cdf, cnt, nsym, m)
    if tok < 4:
        return tok
    if tok < 15:
        n = tok - 2
        return (1 << n) + _raw_get(raw, st, n)
    n = 13 + _raw_get(raw, st, 5)              # escape: very large value
    return (1 << n) + _raw_get(raw, st, n)


# --------------------------------------------------------------------------
# Contexts (identical on both sides)
# --------------------------------------------------------------------------
@fast
def _ll_ctx(a, y, x):
    if y == 0 or x == 0:
        return 0
    act = abs(int(a[y, x - 1]) - int(a[y - 1, x - 1])) + abs(int(a[y - 1, x]) - int(a[y - 1, x - 1]))
    return 0 if act <= 1 else (1 if act <= 6 else 2)


@fast
def _mag_ctx(a, yy, xx, by0, bx0, py0, px0, ph, pw):
    left = abs(int(a[yy, xx - 1])) if xx > bx0 else 0
    up = abs(int(a[yy - 1, xx])) if yy > by0 else 0
    nb = left + up
    nbin = 0 if nb == 0 else (1 if nb == 1 else (2 if nb == 2 else (3 if nb <= 4 else 4)))
    pbin = 0
    if ph > 0 and pw > 0:
        py = min(ph - 1, (yy - by0) >> 1)
        px = min(pw - 1, (xx - bx0) >> 1)
        pbin = min(2, abs(int(a[py0 + py, px0 + px])))
    return M_MAG + nbin * 3 + pbin


@fast
def _node_model(s):
    n = 0
    while (LEAF << n) < s:
        n += 1
    return M_SIG + min(n, 9)


# --------------------------------------------------------------------------
# Band coding: quadtree for zero regions, context-modelled leaves
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _band_encode(a, by0, bx0, bh, bw, py0, px0, ph, pw, st, starts, freqs, raw,
                 cdf, cnt, nsym, stack):
    if bh <= 0 or bw <= 0:
        return
    size = LEAF
    while size < bh or size < bw:
        size *= 2
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack[0, 2] = size
    sp = 1
    while sp > 0:
        sp -= 1
        y, x, s = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        h = min(s, bh - y)
        w = min(s, bw - x)
        if h <= 0 or w <= 0:
            continue
        nz = 0
        for yy in range(by0 + y, by0 + y + h):
            for xx in range(bx0 + x, bx0 + x + w):
                if a[yy, xx] != 0:
                    nz = 1
                    break
            if nz:
                break
        _emit(st, starts, freqs, cdf, cnt, nsym, _node_model(s), nz)
        if nz == 0:
            continue
        if s <= LEAF:
            for yy in range(by0 + y, by0 + y + h):
                for xx in range(bx0 + x, bx0 + x + w):
                    v = int(a[yy, xx])
                    m = _mag_ctx(a, yy, xx, by0, bx0, py0, px0, ph, pw)
                    _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, m, abs(v))
                    if v != 0:
                        _raw_put(st, raw, 1 if v < 0 else 0, 1)
        else:
            half = s // 2
            for cy, cx in ((half, half), (half, 0), (0, half), (0, 0)):
                stack[sp, 0] = y + cy
                stack[sp, 1] = x + cx
                stack[sp, 2] = half
                sp += 1


@njit(cache=True, nogil=True)
def _band_decode(a, by0, bx0, bh, bw, py0, px0, ph, pw, buf, nwords, x4, st, raw,
                 cdf, cnt, nsym, stack):
    if bh <= 0 or bw <= 0:
        return
    size = LEAF
    while size < bh or size < bw:
        size *= 2
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack[0, 2] = size
    sp = 1
    while sp > 0:
        sp -= 1
        y, x, s = stack[sp, 0], stack[sp, 1], stack[sp, 2]
        h = min(s, bh - y)
        w = min(s, bw - x)
        if h <= 0 or w <= 0:
            continue
        if _get(buf, nwords, x4, st, cdf, cnt, nsym, _node_model(s)) == 0:
            continue
        if s <= LEAF:
            for yy in range(by0 + y, by0 + y + h):
                for xx in range(bx0 + x, bx0 + x + w):
                    m = _mag_ctx(a, yy, xx, by0, bx0, py0, px0, ph, pw)
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
def _band_geometry(hs, ws, j, levels, orient):
    """Origin and size of a detail band and of its parent band."""
    h, w, hl, wl = hs[j - 1], ws[j - 1], hs[j], ws[j]
    if orient == 0:
        by0, bx0, bh, bw = 0, wl, hl, w - wl            # HL
    elif orient == 1:
        by0, bx0, bh, bw = hl, 0, h - hl, wl            # LH
    else:
        by0, bx0, bh, bw = hl, wl, h - hl, w - wl       # HH
    if j < levels:
        ph_, pw_, phl, pwl = hs[j], ws[j], hs[j + 1], ws[j + 1]
        if orient == 0:
            py0, px0, ph, pw = 0, pwl, phl, pw_ - pwl
        elif orient == 1:
            py0, px0, ph, pw = phl, 0, ph_ - phl, pwl
        else:
            py0, px0, ph, pw = phl, pwl, ph_ - phl, pw_ - pwl
    else:
        py0, px0, ph, pw = 0, 0, 0, 0
    return by0, bx0, bh, bw, py0, px0, ph, pw


# --------------------------------------------------------------------------
# Denoising and grain estimation (encoder)
# --------------------------------------------------------------------------
_noise_gain_cache: dict[int, np.ndarray] = {}


def noise_gains(levels: int) -> np.ndarray:
    """Standard deviation each band gets from unit white noise, measured by
    pushing noise through the same wavelet. G[level, orientation 0/1/2]."""
    if levels not in _noise_gain_cache:
        rng = np.random.default_rng(0)
        a = np.round(rng.normal(0, 64, (512, 512))).astype(np.int32)
        _fwd2d(a, levels)
        hs, ws = _level_dims(512, 512, levels)
        g = np.ones((levels + 2, 3))
        for j in range(1, levels + 1):
            for o in range(3):
                by0, bx0, bh, bw, *_ = _band_geometry(hs, ws, j, levels, o)
                g[j, o] = a[by0:by0 + bh, bx0:bx0 + bw].std() / 64
        _noise_gain_cache[levels] = g
    return _noise_gain_cache[levels]


@njit(cache=True, nogil=True)
def _shrink_thresholds(a, levels, gains, strength):
    """BayesShrink: threshold = noise variance / signal deviation, per band."""
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    T = np.zeros((levels + 2, 3))
    by0, bx0, bh, bw, _, _, _, _ = _band_geometry(hs, ws, 1, levels, 2)
    if bh * bw < 16 or strength <= 0:
        return T
    sigma = np.median(np.abs(a[by0:by0 + bh, bx0:bx0 + bw].astype(np.float64))) / 0.6745
    sigma /= gains[1, 2]
    for j in range(1, levels + 1):
        for o in range(3):
            by0, bx0, bh, bw, _, _, _, _ = _band_geometry(hs, ws, j, levels, o)
            if bh <= 0 or bw <= 0:
                continue
            band = a[by0:by0 + bh, bx0:bx0 + bw].astype(np.float64)
            sn = sigma * gains[j, o]
            var = np.mean(band * band)
            sx = np.sqrt(max(var - sn * sn, 0.0))
            t = 3.0 * sn if sx < 1e-6 else sn * sn / sx
            T[j, o] = strength * min(t, 3.0 * sn)
    return T


@njit(cache=True, nogil=True)
def _quant_band(a, y0, y1, x0, x1, shift, s_base, mults, mp, rdo, thr):
    for ly in range(y0, y1, 4):
        for lx in range(x0, x1, 4):
            ey, ex = min(y1, ly + 4), min(x1, lx + 4)
            score = 0.0
            nz = 0
            big = False
            for y in range(ly, ey):
                for x in range(lx, ex):
                    c = a[y, x]
                    mag = abs(c) - thr                 # soft threshold (denoise)
                    if mag < 0:
                        mag = 0.0
                    s = _coef_step(s_base, mults, mp, (y - y0) << shift, (x - x0) << shift)
                    u = mag / s
                    qv = int(u)
                    if qv >= 2:
                        big = True
                    elif qv == 1:
                        nz += 1
                        score += 3.0 * u - 2.25
                    a[y, x] = qv if c >= 0 else -qv
            if rdo and not big and 0 < nz <= 2 and score < RDO_LAMBDA * RDO_LEAF_BITS:
                for y in range(ly, ey):
                    for x in range(lx, ex):
                        a[y, x] = 0


@njit(cache=True, nogil=True)
def _quantise(a, levels, base, wts, p, mults, mp, rdo, T):
    hs, ws = _level_dims(a.shape[0], a.shape[1], levels)
    for j in range(levels, 0, -1):
        for o in range(3):
            by0, bx0, bh, bw, _, _, _, _ = _band_geometry(hs, ws, j, levels, o)
            sb = base * wts[p, j, 1 if o == 2 else 0] / (1 << (j - 1))
            _quant_band(a, by0, by0 + bh, bx0, bx0 + bw, j, sb, mults, mp, rdo, T[j, o])
    s_ll = base * wts[p, levels + 1, 0] / (1 << levels)
    _quant_band(a, 0, hs[levels], 0, ws[levels], levels, s_ll, mults, mp, False, 0.0)


@njit(cache=True, nogil=True)
def _grain_sigmas(orig, recon):
    """Robust grain strength in four brightness ranges (x8, as bytes)."""
    th, tw = orig.shape
    hp = np.zeros((4, th * tw))
    count = np.zeros(4, np.int64)
    for y in range(1, th - 1):
        for x in range(1, tw - 1):
            box = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    box += orig[y + dy, x + dx] - recon[y + dy, x + dx]
            r = (orig[y, x] - recon[y, x]) - box / 9.0
            b = min(3, max(0, recon[y, x] >> 6))
            hp[b, count[b]] = abs(r)
            count[b] += 1
    out = np.zeros(4, np.int64)
    for b in range(4):
        if count[b] >= 64:
            sigma = 1.4826 * np.median(hp[b, :count[b]]) / 0.943
            out[b] = min(255, int(sigma * 8 + 0.5))
    # fill empty ranges from their neighbours
    for b in range(4):
        if count[b] < 64:
            best = -1
            for d in range(1, 4):
                for c in (b - d, b + d):
                    if 0 <= c < 4 and count[c] >= 64 and best < 0:
                        best = out[c]
            out[b] = max(best, 0)
    return out


# --------------------------------------------------------------------------
# Tile encode / decode
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _encode_tile(planes, levels, base, wts, mults, perceptual, masking, rdo,
                 strength, gains, grain, init_cdf):
    _, th, tw = planes.shape
    n = th * tw
    out = np.zeros(3 * (n * 12) + 16384, np.uint8)
    spans = np.zeros((3, levels + 1, 2), np.int64)
    starts = np.empty(2 * n + 4096, np.int64)
    freqs = np.empty(2 * n + 4096, np.int64)
    words = np.empty(2 * n + 4096 + 16, np.int64)
    raw = np.zeros(6 * n + 4096, np.uint8)
    stack = np.empty((64, 3), np.int64)
    st = np.zeros(3, np.int64)
    if perceptual and masking:
        mp = _masking_map(planes[0])
    elif perceptual:
        mp = np.full(((th + 15) >> BLOCK_SHIFT, (tw + 15) >> BLOCK_SHIFT), 2, np.int32)
    else:
        mp = np.full((1, 1), 2, np.int32)
    hs, ws = _level_dims(th, tw, levels)
    gsig = np.zeros(4, np.int64)
    pos = 0
    for p in range(3):
        a = planes[p].copy()
        _fwd2d(a, levels)
        T = _shrink_thresholds(a, levels, gains, strength)
        _quantise(a, levels, base, wts, p, mults, mp, rdo, T)
        if p == 0 and grain:
            rec = a.copy()
            _dequantise(rec, levels, base, wts, 0, mults, mp, 1)
            _inv2d(rec, levels, 0)
            gsig = _grain_sigmas(planes[0], rec)
        for layer in range(levels + 1):
            cdf, cnt, nsym = _models_init(init_cdf)
            st[0] = 0
            st[1] = 0
            raw[:] = 0
            if layer == 0:
                if p == 0 and grain:
                    for b in range(4):
                        _raw_put(st, raw, gsig[b], 8)
                if p == 0 and perceptual:
                    for by in range(mp.shape[0]):
                        for bx in range(mp.shape[1]):
                            _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, M_MAP, mp[by, bx])
                for y in range(hs[levels]):
                    for x in range(ws[levels]):
                        r = int(a[y, x]) - _ll_pred(a, y, x)
                        z = (r << 1) if r >= 0 else ((-r) << 1) - 1
                        _emit_value(st, starts, freqs, raw, cdf, cnt, nsym, M_LL + _ll_ctx(a, y, x), z)
            else:
                j = levels + 1 - layer
                for o in range(3):
                    by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, o)
                    _band_encode(a, by0, bx0, bh, bw, py0, px0, ph, pw, st,
                                 starts, freqs, raw, cdf, cnt, nsym, stack)
            p0 = _rans_encode(starts, freqs, st[0], words)
            nw = words.size - p0
            # chunk: word count (u32) | words (u16 little-endian) | raw bits
            start = pos
            out[pos] = nw & 0xFF
            out[pos + 1] = (nw >> 8) & 0xFF
            out[pos + 2] = (nw >> 16) & 0xFF
            out[pos + 3] = (nw >> 24) & 0xFF
            pos += 4
            for i in range(nw):
                wv = words[p0 + i]
                out[pos] = wv & 0xFF
                out[pos + 1] = wv >> 8
                pos += 2
            nraw = (st[1] + 7) >> 3
            for i in range(nraw):
                out[pos + i] = raw[i]
            pos += nraw
            spans[p, layer, 0] = start
            spans[p, layer, 1] = pos
    return out, spans


@njit(cache=True, nogil=True)
def _hash32(x):
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


@njit(cache=True, nogil=True)
def _decode_tile_coeffs(data, offs, th, tw, levels, nlayers, base, wts, mults,
                        perceptual, grain, init_cdf):
    """CPU stage: entropy decoding + dequantisation. Returns the tile's
    wavelet coefficients (3 planes) and its grain statistics."""
    stop = levels - (nlayers - 1)
    planes = np.zeros((3, th, tw), np.int32)
    stack = np.empty((64, 3), np.int64)
    st = np.zeros(3, np.int64)
    x4 = np.zeros(4, np.int64)
    hs, ws = _level_dims(th, tw, levels)
    if perceptual:
        mp = np.zeros(((th + 15) >> BLOCK_SHIFT, (tw + 15) >> BLOCK_SHIFT), np.int32)
    else:
        mp = np.full((1, 1), 2, np.int32)
    gsig = np.zeros(4, np.int64)
    for p in range(3):
        a = planes[p]
        for layer in range(nlayers):
            cs = offs[p, layer, 0]
            length = offs[p, layer, 1]
            if length < 4:
                continue
            buf = data[cs:cs + length]
            nwords = np.int64(buf[0]) | (np.int64(buf[1]) << 8) | (np.int64(buf[2]) << 16) | (np.int64(buf[3]) << 24)
            if 4 + 2 * nwords > length:
                nwords = (length - 4) // 2
            raw = buf[4 + 2 * nwords:]
            cdf, cnt, nsym = _models_init(init_cdf)
            _rans_start(buf, nwords, x4, st)
            st[2] = 0
            if layer == 0:
                if p == 0 and grain:
                    for b in range(4):
                        gsig[b] = _raw_get(raw, st, 8)
                if p == 0 and perceptual:
                    for by in range(mp.shape[0]):
                        for bx in range(mp.shape[1]):
                            mp[by, bx] = min(7, _get_value(buf, nwords, x4, st, raw, cdf, cnt, nsym, M_MAP))
                for y in range(hs[levels]):
                    for x in range(ws[levels]):
                        z = _get_value(buf, nwords, x4, st, raw, cdf, cnt, nsym, M_LL + _ll_ctx(a, y, x))
                        r = (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)
                        a[y, x] = _ll_pred(a, y, x) + r
            else:
                j = levels + 1 - layer
                for o in range(3):
                    by0, bx0, bh, bw, py0, px0, ph, pw = _band_geometry(hs, ws, j, levels, o)
                    _band_decode(a, by0, bx0, bh, bw, py0, px0, ph, pw, buf, nwords,
                                 x4, st, raw, cdf, cnt, nsym, stack)
        _dequantise(a, levels, base, wts, p, mults, mp, stop + 1)
    return planes, gsig


@njit(cache=True, nogil=True)
def _decode_tile(data, offs, th, tw, levels, nlayers, base, wts, mults,
                 perceptual, grain, grain_amount, init_cdf, out, oy, ox):
    stop = levels - (nlayers - 1)
    planes, gsig = _decode_tile_coeffs(data, offs, th, tw, levels, nlayers, base,
                                       wts, mults, perceptual, grain, init_cdf)
    for p in range(3):
        _inv2d(planes[p], levels, stop)
    hs, ws = _level_dims(th, tw, levels)
    add_grain = grain and grain_amount > 0 and stop == 0
    h, w = hs[stop], ws[stop]
    for y in range(h):
        for x in range(w):
            Y = planes[0, y, x]
            if add_grain:
                # grain strength for this brightness (linear between ranges)
                f = (Y - 32) / 64.0
                if f <= 0:
                    sig = gsig[0]
                elif f >= 3:
                    sig = gsig[3]
                else:
                    i0 = int(f)
                    sig = gsig[i0] + (gsig[i0 + 1] - gsig[i0]) * (f - i0)
                hv = _hash32(((oy + y) << 16) ^ (ox + x) ^ 0x5F3759DF)
                g = ((hv & 0xFF) + ((hv >> 8) & 0xFF) + ((hv >> 16) & 0xFF)
                     + (hv >> 24) - 510.0) / 147.8
                Y += int(np.floor(g * sig * grain_amount / 8.0 + 0.5))
            co = planes[1, y, x]
            cg = planes[2, y, x]
            t = Y - (cg >> 1)
            gg = cg + t
            b = t - (co >> 1)
            r = b + co
            out[oy + y, ox + x, 0] = min(255, max(0, r))
            out[oy + y, ox + x, 1] = min(255, max(0, gg))
            out[oy + y, ox + x, 2] = min(255, max(0, b))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def estimate_noise(rgb: np.ndarray) -> float:
    """Noise level of the brightness channel (standard deviation, in levels),
    from the finest diagonal wavelet band: real detail is sparse there, noise
    is everywhere, so the median magnitude mostly measures noise."""
    rgb = np.asarray(rgb)[..., :3]
    y = ((rgb[..., 0].astype(np.int32) + 2 * rgb[..., 1].astype(np.int32)
          + rgb[..., 2].astype(np.int32)) >> 2).astype(np.int32)
    if min(y.shape) < 8:
        return 0.0
    a = np.ascontiguousarray(y)
    _fwd2d(a, 1)
    h, w = a.shape
    hh = a[(h + 1) // 2:, (w + 1) // 2:].astype(np.float64)
    # mean of the central 50% of magnitudes: like a median, but not
    # rounded to whole levels
    mags = np.sort(np.abs(hh).ravel())
    core = mags[len(mags) // 4: 3 * len(mags) // 4]
    med = float(core.mean()) if core.size else 0.0
    return med / 0.6745 / noise_gains(5)[1, 2]


def auto_denoise_strength(noise: float) -> float:
    """Clean photos measure ~0-3, visibly noisy ones 4+. Ramp from 0 at 3 to
    full strength at 5 (calibrated on test photos with added sensor noise)."""
    return float(np.clip((noise - 3.0) / 2.0, 0.0, 1.0))


def encode(rgb: np.ndarray, quality: int = 75, perceptual: bool = True,
           tile_size: int = 256, denoise: float | str = "auto",
           grain: bool | str = "auto",
           masking: bool = False, rdo: bool = True,
           weights: np.ndarray | None = None) -> bytes:
    """quality 1-100 (100 = exactly lossless). denoise: "auto" (the encoder
    measures the image's noise and decides), or a BayesShrink strength
    (0 = off, 1 = standard). grain: store grain statistics (about 4 bytes per
    tile) so decoders can regenerate the removed texture; "auto" stores them
    whenever the encoder denoises."""
    if tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {TILE_SIZES}")
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
    grain = (denoise > 0) if grain == "auto" else bool(grain)
    grain = grain and not lossless

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
    wts = wts.astype(np.float32).astype(np.float64)
    gains = noise_gains(levels)
    tiles = _tile_grid(w, h, tile_size)

    def work(tile):
        ty, tx = tile
        sub = np.ascontiguousarray(planes[:, ty * tile_size:(ty + 1) * tile_size,
                                          tx * tile_size:(tx + 1) * tile_size])
        return _encode_tile(sub, levels, base, wts, MULTS, perceptual, masking,
                            rdo, denoise, gains, grain, INIT_CDF)

    results = list(_pool().map(work, tiles))
    lengths, payload = [], []
    for layer in range(levels + 1):
        for buf, spans in results:
            for p in range(3):
                s, e = spans[p, layer]
                lengths.append(e - s)
                payload.append(buf[s:e].tobytes())
    flags = (FLAG_PERCEPTUAL if perceptual else 0) | (FLAG_GRAIN if grain else 0)
    header = HEADER.pack(MAGIC, VERSION, w, h, quality, flags, levels, tile_size, base)
    return (header + wts.astype("<f4").tobytes()
            + np.asarray(lengths, dtype="<u4").tobytes() + b"".join(payload))


def read_header(data: bytes) -> dict:
    if len(data) < HEADER.size:
        raise ValueError("File is too short to be a FUL v4 image")
    magic, version, w, h, quality, flags, levels, tile, base = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    if version != VERSION:
        raise ValueError(f"Not a FUL v4 file (version {version})")
    if tile not in TILE_SIZES or levels != levels_for_tile(tile):
        raise ValueError("Corrupt FUL v4 header")
    return {"version": 4, "width": w, "height": h, "quality": quality,
            "perceptual": bool(flags & FLAG_PERCEPTUAL),
            "grain": bool(flags & FLAG_GRAIN), "levels": levels,
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
        raise ValueError("FUL v4 file is truncated")
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


@njit(cache=True, nogil=True)
def _coeffs_into(data, offs, th, tw, levels, base, wts, mults, perceptual, grain,
                 init_cdf, coef, oy, ox, gsig_out, overflow):
    planes, gsig = _decode_tile_coeffs(data, offs, th, tw, levels, levels + 1, base,
                                       wts, mults, perceptual, grain, init_cdf)
    for y in range(th):
        for x in range(tw):
            for p in range(3):
                v = planes[p, y, x]
                if v > 32767 or v < -32768:
                    overflow[0] = 1
                coef[oy + y, ox + x, p] = v
    for b in range(4):
        gsig_out[b] = gsig[b]


def decode_coefficients(data: bytes):
    """First half of decoding, for GPU decoders: entropy decoding and
    dequantisation on the CPU. Returns (coef, grain, info) where coef is an
    (H, W, 3) int16 array of Y, Co, Cg wavelet coefficients (each tile in its
    own wavelet layout; 6 bytes per pixel, the compact form sent to the GPU),
    and grain is (tiles_y, tiles_x, 4) grain strengths x8, or None."""
    info = read_header(data)
    levels, tile = info["levels"], info["tile_size"]
    w, h = info["width"], info["height"]
    k = levels + 1
    wts, tiles, starts, lengths, _ = _layout(data, info)
    if (starts + lengths).max() > len(data):
        raise ValueError("FUL v4 file is truncated")
    coef = np.zeros((h, w, 3), np.int16)
    overflow = np.zeros(1, np.int64)
    nty, ntx = (h + tile - 1) // tile, (w + tile - 1) // tile
    grain = np.zeros((nty, ntx, 4), np.int64)
    buf = np.frombuffer(data, np.uint8)
    base = float(info["base"])

    def work(i):
        ty, tx = tiles[i]
        th = min(tile, h - ty * tile)
        tw = min(tile, w - tx * tile)
        offs = np.empty((3, k, 2), np.int64)
        offs[:, :, 0] = starts[:k, i, :].T
        offs[:, :, 1] = lengths[:k, i, :].T
        _coeffs_into(buf, offs, th, tw, levels, base, wts, MULTS, info["perceptual"],
                     info["grain"], INIT_CDF, coef, ty * tile, tx * tile, grain[ty, tx],
                     overflow)

    list(_pool().map(work, range(len(tiles))))
    if overflow[0]:
        raise ValueError("Coefficients exceed 16 bits; use the CPU decoder")
    return coef, (grain if info["grain"] else None), info


def decode(data: bytes, layers: int | None = None, grain: float = 1.0) -> np.ndarray:
    """Decode to (H, W, 3) uint8. layers < all gives a smaller preview.
    grain scales any stored grain (0 = off, 1 = as measured)."""
    info = read_header(data)
    levels, tile = info["levels"], info["tile_size"]
    w, h = info["width"], info["height"]
    k = levels + 1 if layers is None else max(1, min(levels + 1, int(layers)))
    wts, tiles, starts, lengths, _ = _layout(data, info)
    if (starts[:k] + lengths[:k]).max() > len(data):
        raise ValueError("FUL v4 file is truncated")
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
                     info["perceptual"], info["grain"], float(grain), INIT_CDF,
                     out, ty * tile_s, tx * tile_s)

    list(_pool().map(work, range(len(tiles))))
    return out


if __name__ == "__main__":
    import codec_v1
    import codec_v3

    rng = np.random.default_rng(7)
    for shape in [(1, 1, 3), (5, 70, 3), (37, 53, 3), (300, 260, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        for tile in TILE_SIZES:
            assert np.array_equal(decode(encode(img, 100, tile_size=tile)), img), (shape, tile)
            data = encode(img, 60, True, tile, denoise=1.0, grain=True, masking=True)
            for k in range(1, layer_count(data) + 1):
                decode(data, k)
    print("Lossless round trips and progressive decodes verified.")
    img = codec_v1.make_test_image()
    for q in (100, 90, 75, 50):
        d3 = codec_v3.encode(img, q)
        d4 = encode(img, q)
        print(f"quality {q:3d}: v3 {len(d3):>9,} bytes   v4 {len(d4):>9,} bytes "
              f"({(len(d4) / len(d3) - 1) * 100:+.1f}%)")
