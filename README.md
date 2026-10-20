# FUL

An experimental image format built from first principles: whole-image integer
wavelets with honeycomb-style directional prediction, perceptual quantisation,
real-cost rate-distortion optimisation, context-modelled rANS entropy coding,
automatic denoising with grain regeneration, and a GPU decoder that is
bit-identical to the CPU decoder.

At equal SSIM, FUL v6 files are about 31% smaller than JPEG on the tuning
photos and 23% smaller on held-out images (WebP ~30%, JPEG XL ~28%,
JPEG 2000 ~21%, AVIF ~46%).

## Quick start

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python viewer.py                   # FUL Lab: compare FUL with other formats
    python codec_v6.py                 # self-test
    python bench/run.py                # benchmark against JPEG

Contributions and research: see `CONTRIBUTING.md`.

## Files

- `codec_v1.py` ... `codec_v6.py`: every version of the format. Each keeps its
  own decoder so older `.ful` files always open. v6 is the current format.
- `ful.py`: opens any `.ful` file by reading its version byte.
- `gpu_decode.py`: CPU entropy decoding + OpenGL decoding of the wavelet,
  colour and grain stages (v4 to v6).
- `formats.py`: JPEG, WebP, AVIF, JPEG XL and JPEG 2000 for comparison.
- `metrics.py`: PSNR and colour SSIM.
- `viewer.py`: FUL Lab, the PyQt6 test bench.
- `bench/run.py`: reproducible benchmark (size at equal SSIM vs JPEG).

## History and what has been tried

- v1: pixel prediction (MED) + adaptive Rice codes.
- v2: integer 5/3 wavelets + quadtree, tiled and progressive.
- v3: perceptual quantisation. Helped: coarse fine-colour and diagonal steps,
  reconstruction offset 0.4 (~10%). Didn't help: texture masking and Weber's-law
  scaling (lower SSIM per byte).
- v4: context-modelled 4-state rANS (~10%), BayesShrink denoising (about half
  the size of JPEG on noisy photos), grain regeneration.
- v5: whole-image transform instead of 256px tiles (+7 points), 9/7 wavelet for
  lossy files (+8), 4 extra bits of internal transform precision.
- v6: directional lifting (+4 on photos, ~0 on edge-poor images), real-cost RDO
  with rounding up allowed (+1.7). Didn't help: a third directional level,
  removing the straight-prediction bias, RDO that only rounds down.
- Measured, not built: sign contexts (~0.9% of file), coding the first low
  bit (~0.4%), compact header/table (most of the 5% on small thumbnails).

## Setup (VS Code)

1. Open this folder in VS Code (File > Open Folder).
2. In the terminal:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Ctrl+Shift+P > "Python: Select Interpreter" > choose the `.venv` one.
5. Run and Debug (Ctrl+Shift+D) > "FUL Lab (viewer)" or one of the self-tests.

Numba is optional but strongly recommended: without it everything still works,
only much slower, and v2 can't use multiple cores.

## File layout

Every file starts with the magic bytes `FUL\0` and a version byte.
All integers are little-endian.

### Version 1

    magic | version=1 u8 | width u32 | height u32 | q_luma u8 | q_chroma u8 | flags u8
    then 3 planes (Y, Co, Cg), each: byte length u32 | Rice-coded residuals

### Version 2

    magic | version=2 u8 | width u32 | height u32 | q_luma u8 | q_chroma u8
          | flags u8 | levels u8 | tile_size u16
    chunk length table: u32 per chunk, ordered layer > tile > plane
    chunk data in the same order

Layer 0 holds each tile's coarsest wavelet band; each later layer doubles the
resolution. Reading only the first k layers gives the whole image at
1 / 2^(levels + 1 - k) scale. Flag bit 0 = finest chroma detail not stored.

### Version 3

    magic | version=3 u8 | width u32 | height u32 | quality u8 | flags u8
          | levels u8 | tile_size u16 | base_step f32
    band weights: f32 x 3 planes x (levels + 2) x 2 orientations
    chunk length table and chunk data as in version 2

Coefficient step = base_step x weight[plane, level, orientation] / 2^(level-1)
x block multiplier (from the map, when present), floored at 1.
Flag bit 0 = perceptual (a 16x16-block multiplier map precedes each tile's
luma LL band). The decoder reconstructs nonzero values at (|q| + 0.4) x step.

### Version 4

Header, band weights and chunk table as in version 3 (version byte 4).
Flag bit 1 = grain statistics present.

Each chunk:

    rANS word count u32 | rANS words (u16 little-endian) | raw bit stream

The rANS stream uses four interleaved 32-bit states (symbol i uses state
i mod 4), 15-bit probabilities and 16-bit renormalisation. Models adapt after
every symbol and reset at each chunk. Coded symbols are quadtree node bits and
magnitude tokens (0-3 literal, 4-14 = magnitude class with raw low bits,
15 = escape). Signs, low bits and grain statistics go in the raw stream.
Coefficient contexts: left + upper neighbour magnitude (5 bins) x parent
coefficient magnitude (3 bins).

## GPU decoding

`gpu_decode.py` splits decoding: the CPU entropy-decodes and dequantises, then
uploads the coefficients as a 16-bit, 3-channel texture. The GPU undoes the
wavelet in 4 full-screen shader passes per level (even and odd lifting steps,
columns then rows, all tiles at once), then a final pass converts YCoCg to RGB
and adds grain. Integer maths matches the CPU decoder exactly, so the output
is bit-identical; the lab checks this on every decode ("GPU check").

Needs OpenGL 3.3 core (macOS provides 4.1). On a headless Linux machine, set
`FUL_GPU_EGL=1` to use EGL instead of Qt for the context.

### Version 5

    magic | version=5 u8 | width u32 | height u32 | quality u8 | flags u8
          | levels u8 | precision u8 | base_step f32
    band weights: f32 x 3 planes x (levels + 2) x 2 orientations
    code-block length table: u32 per block
    code-blocks (same internal format as version 4 chunks)

Flags: bit 0 perceptual (masking map in the luma LL block), bit 1 grain
(4 bytes per 256x256 cell, first in the luma LL block), bit 2 = 9/7 wavelet
(otherwise 5/3). Precision: low 4 bits = prescale (coefficients are in units
of 1/2^prescale level), high 4 bits = extra internal transform bits.

The wavelet covers the whole image. Levels = min(6, log2(min side) - 3).
Lifting steps are integer: x += sign * floor((m * (a + b) + r) / 2^s), with the
9/7 multipliers at 10-bit precision; its scale factors are folded into the
band weights. Code-blocks: layer 0 = one LL block per plane; each later layer
= per plane, per band (HL, LH, HH), horizontal strips of >= 64K coefficients
(height a multiple of 64). Upper-neighbour contexts don't cross strip edges.

### Version 6

Same layout as version 5 (version byte 6), plus flag bit 3 = directional
lifting. Each level is transformed columns first, then rows.

With the directional flag, the luma LL block also carries direction maps
(after the masking map, same model): for each of the two finest levels, one
value 0-6 per 16x16 block of that level's region, meaning a column shift of
0, -1, +1, -2, +2, -4, +4 half pixels per row. In the column lifting steps a
sample at row r, column x predicts from the row above at x - d and the row
below at x + d (half positions: floor of the average of the two neighbours;
columns clamped to the region). Rows mirror at the edges as before.

Real-cost RDO is encoder-only and needs no format change: decoders read v6
files exactly as v5 apart from the directions.
