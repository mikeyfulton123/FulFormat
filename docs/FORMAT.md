# How FUL works

A complete description of the FUL image format as of **v9** (file format
version 9). It is written for anyone proposing changes, including research
agents: read it alongside `docs/EXPERIMENTS.md`, which lists every idea that
has already been measured.

Code references point to the Python reference implementation in this
repository (`codec_v9.py` and the modules it builds on).

---

## 1. Design principles

1. **The encoder thinks, the decoder follows.** Any expensive search (choosing
   directions, quantised values, filter strengths) happens in the encoder, and
   only the decisions are stored. The decoder never searches.
2. **Light, exact decoding.** The decoder uses integer arithmetic only, so a GPU
   decoder (`gpu_decode.py`) produces bit-identical output to the CPU decoder.
   No neural networks.
3. **Every file is progressive.** Data is ordered coarse to fine, so any prefix
   of the file decodes to a complete, smaller image.
4. **Quality 100 is exactly lossless.**
5. **Measured, not assumed.** Every feature was kept because it measured better
   on the benchmark (`bench/run.py`); see `docs/EXPERIMENTS.md`.

---

## 2. The encoder, step by step

### 2.1 Colour transform
RGB is converted to **YCoCg-R**, an exactly reversible integer transform:

    Co = R - B;  t = B + (Co >> 1);  Cg = G - t;  Y = t + (Cg >> 1)

Y is brightness (luma); Co and Cg are colour (chroma). Colour is kept at full
resolution (no 4:2:0 subsampling); fine colour detail is instead quantised very
coarsely (section 2.5).

### 2.2 Noise estimation and denoising (encoder only)
`estimate_noise_v2` measures noise in the **flattest 25% of 16x16 blocks**: the
fine-scale residual (image minus a Gaussian blur of sigma 1) has its 10th
percentile standard deviation taken there, calibrated against synthetic white
and slightly blurred noise. If the image has no flat regions (the selected
blocks' median gradient is 2.5 or more), it falls back to the v4 estimator
(median magnitude of the finest diagonal wavelet band).

Denoising strength = clip((noise - 1.6) / 1.6, 0, 1). Clean photos read below
~1.4 and are left alone. When strength > 0, **BayesShrink** soft thresholding
is applied to every detail band during quantisation: threshold = strength x
noise_variance_in_band / signal_deviation_in_band (capped at 3x the band noise).

If the encoder denoised, it also stores **grain statistics** (section 4.6).

### 2.3 Wavelet transform
A whole-image (untiled) **integer lifting wavelet**:

- **Lossy files:** CDF 9/7 (the filter of lossy JPEG 2000) as four integer
  lifting steps with 10-bit multipliers:
  `x += sign * floor((m * (a + b) + r) / 2^s)`. The 9/7's scaling factors are not
  applied in the transform; they are folded into the step sizes (2.5).
- **Lossless files:** LeGall 5/3, exactly reversible.
- **Levels:** min(6, log2(min(width, height)) - 3).
- **Precision:** pixels are pre-scaled by 2^2 (file units) and the transform runs
  with 4 further bits of internal precision (`EXTRA_97`); results are rounded
  back to file units. Without this, rounding inside the lifting steps produced
  visible structured error.
- **Order per level:** columns first, then rows (the inverse: rows, then columns).
- **Edges:** symmetric extension.

#### Directional lifting (2 finest levels)
In the column pass of the two finest levels, each sample can be predicted along
a **slanted line** instead of straight down. Each 16x16 block (in that level's
coordinates) has one of 7 directions: a horizontal shift of 0, +-0.5, +-1 or
+-2 pixels per row (about 0, 27, 45 and 63 degrees). Half-pixel positions use
the floor of the average of the two neighbours; columns are clamped at the
region edge. Any deterministic prediction keeps integer lifting exactly
reversible.

The encoder chooses each block's direction by running the column pass with each
candidate and keeping the one with the smallest sum of high-band magnitudes
(straight is preferred unless a slant is 3% better).

### 2.4 Band layout
After L levels, each plane holds the coarsest LL band plus, for each level j
(1 = finest), three detail bands: **HL** (horizontal detail), **LH** (vertical
detail), **HH** (diagonal detail), in the usual Mallat layout.

### 2.5 Step sizes and quantisation
Each coefficient's quantiser step:

    step = base x W[plane, level, orientation] x gain[level, orientation]
           / 2^(level - 1) x block_multiplier, floored at 1

- **base** = 2^((100 - quality) / 10).
- **W** (perceptual weights, `codec_v5.perceptual_weights`): luma 1.25 at the
  finest level, 1 elsewhere; chroma 16, 5, 3, 2, 1.5, 1.5 from finest to
  coarsest (and 1.5 for the chroma LL); diagonal (HH) bands x2.
- **gain** folds in the 9/7's unapplied scaling factors (K^2 per level) and the
  pre-scale (`codec_v5.band_gains`).
- **block_multiplier** is 1 unless the optional masking map is used (off by
  default; it lowered SSIM per byte).

The 1/2^(level - 1) term gives each level roughly equal squared error per
coefficient, the classic equal-error allocation.

**Detail bands: dependent quantisation (DQ).** Two quantisers with interleaved
reconstruction points are selected by a 4-state machine driven by the parity of
each stored value (next state from parity: 0->{0,2}, 1->{2,0}, 2->{1,3},
3->{3,1}). States 0-1 reconstruct 2|k| x eta x step; states 2-3 reconstruct
(2|k| - 1) x eta x step, with eta = 1.0. The encoder picks values with a
**Viterbi trellis** in raster order per band, minimising squared error (in step
units) + lambda x estimated bits (lambda = 0.12; bit costs from token
frequencies of a first pass).

**LL band:** truncation, reconstructed at (|q| + 0.4) x step.

(With DQ off, detail bands use v6's real-cost rate-distortion optimisation:
each value chooses between rounded-up, rounded, one less or zero, then whole
4x4 groups are kept or dropped by comparing real costs.)

### 2.6 Code-blocks
For parallel decoding, coefficients are coded in independent **code-blocks**,
listed in file order by `chunk_list`:

1. **Layer 0:** one block per plane holding the LL band (plane 0's block also
   carries side information: grain, masking map, direction maps).
2. **Layers 1..L:** detail levels from coarsest to finest. Within a layer: per
   plane (Y, Co, Cg), per band (HL, LH, HH), horizontal strips of at least
   65,536 coefficients, height a multiple of 64.

Because layers are ordered coarse to fine, a file prefix holds a complete
lower-resolution image.

### 2.7 Entropy coding
Each code-block is one **rANS** stream plus one **raw bit stream**:

- **rANS:** 4 interleaved 32-bit states (symbol i uses state i mod 4), 15-bit
  probabilities (totals exactly 2^15), 16-bit renormalisation, encoded in
  reverse so the decoder runs forwards. Decoding needs no division.
- **Adaptive models:** each model is a 16-symbol (or binary) CDF that moves
  toward each coded symbol at rate 2^-(4..7) (faster at first), always keeping
  every probability >= 1/2^15. Models reset at every code-block.
- **Priors:** models start from learned tables (`priors_v9.npy`), indexed by
  plane class (luma / chroma) and level class (1, 2, 3, 4+), trained on photos
  (`python codec_v9.py --train`). Starting confidence 0 (adapt at full speed).

**Detail band symbols, per strip:**

1. **Quadtree:** the strip is covered by a square of side >= 4; each node sends
   one coded bit, "is anything inside non-zero?" (context: node size). Zero
   nodes end there; non-zero nodes split into 4 children down to 4x4 leaves.
2. **Leaves:** each coefficient's magnitude is coded as a **token**:
   - 0-3: the value itself;
   - 4-14: magnitude class n = floor(log2 v) (token = 2 + n); the first bit
     below the leading 1 is coded with a model per n; the remaining n - 1 bits
     go to the raw stream;
   - 15: escape for v >= 2^13 (raw).
   **Context (v9, 15 per model set):** one local-spread estimate, in the spirit
   of Gaussian scale mixtures:

       S = 2|left| + |left2| + 2|up| + |up-left| + |up-right| + |up2|
           + 2|parent| + sum of |sibling at the same position|
       spread = S / (10 + number of siblings)

   binned at 0 (S = 0), then edges 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12,
   16, 24 (computed in integers as 4S >= 4e x norm). The parent is the
   co-located coefficient one level coarser; siblings are the same position in
   the bands of the same level decoded earlier (HL for LH; HL and LH for HH).
   Neighbours above only count inside the strip; up-right only within the same
   column of 4x4 leaves (across a leaf edge it may not be decoded yet).
   (v4-v8: bin(|left| + |up|) into 5 bins x min(|parent|, 2).)
3. **Signs:** coded with 9 contexts: sign of the left neighbour x sign of the
   upper neighbour (negative / zero / positive).

**LL band:** MED (median edge detector) prediction from left, upper and
upper-left; the residual is zig-zag mapped and coded as a token with 3 contexts
from local gradient activity.

**Side information** (in plane 0's LL block): grain statistics (raw bytes),
masking map (if any), direction maps (one value 0-6 per 16x16 block for the 2
finest levels), all coded with one model.

### 2.8 Post-filter decision (encoder only)
After encoding, the encoder decodes its own file and, for each 32x32 block,
tries the post-filter (section 4.5) at strengths 0-4, keeping the one with the
smallest squared error against the original. The choices are coded (context:
previous block's strength) and stored at the end of the file. Not used when
grain is regenerated.

---

## 3. File layout (version 9; version 8 is identical apart from the version byte, contexts and priors)

All integers are little-endian.

| Field | Size | Meaning |
|---|---|---|
| magic | 4 | `FUL\0` |
| version | 1 | 9 |
| width, height | 4 + 4 | pixels |
| quality | 1 | 1-100 |
| flags | 1 | bit 0 perceptual, 1 grain, 2 9/7 wavelet, 3 directional, 4 DQ, 5 custom weights, 6 post-filter |
| levels | 1 | wavelet levels |
| precision | 1 | low 4 bits prescale, high 4 bits extra internal bits |
| base | 4 | float32 base step |
| dq eta | 1 | eta x 100 |
| weights | 4 x 3 x (L+2) x 2 | float32, only if flag bit 5 |
| block table | varies | LEB128 length of every code-block, then of the filter block if flag bit 6 |
| code-blocks | varies | in `chunk_list` order |
| filter block | varies | post-filter strengths (if flag bit 6) |

**Each code-block:** u32 rANS word count | rANS words (u16) | raw bit stream.

Older versions (1-7) remain readable through `ful.py`; their layouts are in
the README.

---

## 4. The decoder, step by step

1. **Header and block table.** Read the header; rebuild the standard step
   weights (or read custom ones); read the block lengths.
2. **Entropy decoding (CPU).** Layer by layer, coarse to fine (parents must
   exist before children's contexts). Within a layer, three phases: all HL
   blocks, then LH, then HH (siblings must exist first); blocks within a phase
   decode in parallel.
3. **Dequantisation (CPU).** Detail bands through the DQ state machine in
   raster order per band; LL at (|q| + 0.4) x step.
4. **Inverse wavelet (CPU or GPU).** Level by level: rows, then columns
   (directional at the 2 finest levels). The GPU decoder runs one shader pass
   per lifting step.
5. **Post-filter (CPU or GPU, if flagged).** A 5x5 integer bilateral filter:
   weight = spatial[dy][dx] x range[strength][|Y(n) - Y(c)|] with
   Y = (77R + 150G + 29B) >> 8; output = (sum(w x v) + den/2) / den. Tables
   `SPATIAL_LUT`, `RANGE_LUT` in `codec_v8.py`.
6. **Grain and colour (CPU or GPU).** Undo the pre-scale; if grain is present,
   add pseudo-random grain: strength interpolated from the cell's 4 numbers by
   brightness, noise from a 32-bit hash of the pixel position (deterministic,
   parallel); convert YCoCg back to RGB.

**Progressive previews:** decoding only the first k layers and stopping the
inverse transform early gives the image at 1/2^(L+1-k) size.

---

## 5. Where the bytes go

Typical photo at quality 75 (measured on FUL v7; v8 is similar):

| Part | Share of file |
|---|---|
| Luma, finest detail level | ~33% |
| Luma, second finest level | ~26% |
| Luma, coarser levels | ~18% |
| Chroma, all levels | ~20% |
| LL bands, maps, header, block table | ~3% |
| (of which raw low bits, across all parts) | ~5% |

A change can never save more than the share it touches.

---

## 6. What can change without a new file version

Anything that only alters **encoder decisions** produces files existing decoders
read correctly: noise estimation and denoise strength, direction choices,
quantised values (trellis, lambda), post-filter choices, and custom weights
(stored in the file). These are the cheapest ideas to deploy.

Anything that changes **what the decoder computes** needs a new version: new
contexts or models, new transforms or filters, new side information.

---

## 7. Benchmarks used to judge changes

`bench/run.py`: file size at equal colour SSIM (YCbCr, weighted 6:1:1),
relative to JPEG (Pillow, optimised Huffman), interpolated at the SSIM values
JPEG reaches at qualities 50, 75 and 90, on tuning photos and held-out images.
Gains should be confirmed with a second metric (MS-SSIM on brightness, PSNR, or
CIEDE2000 for colour) and, for anything visual, side-by-side images.
