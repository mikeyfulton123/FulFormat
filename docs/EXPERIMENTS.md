# FUL experiment ledger

Every idea that has been **measured** for FUL, with its result and verdict.
Check here before proposing anything: if an idea (or a close variant) is
listed, a new proposal must explain what is genuinely different about it.

## Rules for this file

- **Only measured results go here.** Predictions, estimates from agents, and
  "conceptual" tests do not. Each entry says how it was measured.
- **Negative results stay.** They are as valuable as wins: they stop repeats.
- **Add new entries at the end of the relevant section,** with the next ID.
- **Unless stated otherwise,** "vs JPEG" means file size relative to JPEG at
  equal colour SSIM (`bench/run.py` method), negative = smaller. "Tuning" =
  astronaut, coffee, chelsea, rocket; "held-out" = hubble_deep_field,
  immunohistochemistry; "textures" = grass, gravel, brick, cat fur crop,
  astronaut hair crop. "Offline" = ideal code length estimated on FUL's real
  coefficients, not a full implementation.

Verdicts: **Adopted** (in the format), **Lead** (promising, not built yet),
**Small** (real but minor), **Dropped** (no gain or worse), **Duplicate**
(already in FUL in another form).

---

## Current reference points (v8.1; v9 files are a further ~0.7-1.3% smaller)

| Measure | FUL | JPEG XL | JPEG 2000 | AVIF |
|---|---|---|---|---|
| Clean photos, equal SSIM vs JPEG | -35% | -28% | -21% | -46% |
| Real camera photo crops, SSIM | -46% | -40% | -30% | -35% |
| Real camera photo crops, MS-SSIM | -17% | -29% | +6% | -17% |
| Simulated low-light phone noise, vs clean scene | -63% | +37% | +262% | +30% |

---

## Colour

| ID | Idea | Measured | Result | Verdict |
|---|---|---|---|---|
| C01 | YCoCg-R reversible colour transform | v1 onwards | Basis of all versions | Adopted |
| C02 | Very coarse fine-colour steps (x16 finest, then 5, 3, 2, 1.5) | v3/v5 weight searches | Clear SSIM gain | Adopted |
| C03 | Brightness as a context for colour coefficients (separate contexts) | Offline, v6 coefficients | +1.33% file (worse: dilution) | Dropped |
| C04 | Brightness folded into the neighbour sum for colour | Offline | +0.07% (neutral; varies by image) | Dropped |
| C05 | Colour predicted from brightness per 16x16 block (chroma-from-luma) | Offline, 6 images | -0.45% file (rocket -1.7%) | Small |
| C06 | Colour steps 1.6x coarser | Benchmark, photos | MS-SSIM vs JPEG +5.1% -> -3.9%; SSIM -34.5% -> -33.6%; colour share 20% -> 14% | Lead (judgement call) |
| C07 | Colour steps 2.5x coarser | Benchmark, photos | MS-SSIM -6.6%; SSIM -30.6% | Dropped (too costly in SSIM) |
| C08 | Colour budget vs AVIF | Greyscale-encode estimate | AVIF ~17% of bytes on colour, FUL ~20%, equal CIEDE2000 | Not the AVIF gap |

## Noise and grain

| ID | Idea | Measured | Result | Verdict |
|---|---|---|---|---|
| N01 | BayesShrink denoising, fixed strength | Clean photos; noisy (sigma 8) vs clean scene | Hurts clean photos; ~-52% vs JPEG on noisy | Adopted (automatic) |
| N02 | Automatic denoise from finest-band noise estimate (v4) | Calibration on clean/noisy | Clean vs noisy separated for uniform noise | Superseded by N06 |
| N03 | Grain regeneration (4 numbers per tile/cell, hash noise) | Noisy photos | Restores texture energy (e.g. 8.5 vs original 9.3); lowers SSIM; ~20 bytes | Adopted (optional viewing) |
| N04 | Non-local means denoising in the encoder | Noisy photos vs clean scene | -52.5% vs BayesShrink's -63.9% (much better on rocket, worse on others) | Dropped |
| N05 | Second-band noise estimate | Calibration | Real detail dominates; clean photos read up to 7.8 | Dropped |
| N06 | Flat-region noise estimator (v8.1) | Simulated phone noise | Low light vs clean scene -56.0% -> -63.1%; daylight -45.3% -> -43.5%; clean photos untouched | Adopted |
| N07 | Denoise ramp starting at 2.0 instead of 1.6 | Simulated phone noise | Low light -53.0% (loses most of N06's gain) | Dropped |

## Transform

| ID | Idea | Measured | Result | Verdict |
|---|---|---|---|---|
| T01 | 5/3 wavelet + quadtree (v2) vs pixel prediction (v1) | Photos | Smaller files and better quality at strong compression; lossless ~5% smaller | Adopted |
| T02 | 9/7 wavelet for lossy files | Experiment, then v5 | ~+8 points vs JPEG | Adopted |
| T03 | Whole-image transform instead of 256px tiles | Experiment, then v5 | ~+7 points (tile edges and model resets) | Adopted |
| T04 | Extra internal transform precision (4 bits) | v5 | Recovered SSIM lost to lifting rounding (0.9353 -> 0.9389 at q60) | Adopted |
| T05 | 6 levels vs 5 | v5 | No difference | n/a |
| T06 | Directional lifting, 7 angles, column pass, 2 finest levels (v6) | Benchmark | +4 points photos; ~0 on edge-poor images | Adopted |
| T07 | A third directional level | Benchmark | +0.2 points | Dropped |
| T08 | No bias toward straight prediction | Benchmark | No change | Dropped |
| T09 | Directional row pass too | Benchmark | Photos -31.47% vs -31.95%; textures -27.25% vs -27.94% (maps cost more than saved) | Dropped |
| T10 | 11 directions instead of 7 | Benchmark | No change (-31.94% vs -31.95%) | Dropped |
| T11 | Wavelet packets (best-basis split of finest bands) | Offline | Photos +0.9% bytes for -4.8% error (under 1% net); brick -2.6% bytes and -7.7% error | Small (regular textures only) |
| T12 | Hexagonal sampling lattice | Resampling experiment | Slightly better per sample (SSIM +0.002-0.004) but needs resampling | Dropped (for now) |
| T13 | DCT (8-32 px blocks) instead of wavelet at medium levels | Offline, equal error | DCT needs 35-62% more bits | Wavelet confirmed |

## Prediction

| ID | Idea | Measured | Result | Verdict |
|---|---|---|---|---|
| P01 | Lateral inhibition: subtract neighbour average from coefficients | Offline | +3.2% file (worse) | Dropped |
| P02 | Fixed-weight neighbour prediction (neuroscience "predictive coding") | Offline | +20.8% file (worse); as proposed (with right neighbour) +19.6% and not decodable | Dropped |
| P03 | Cross-scale prediction from upscaled half-size image (cubic) | Offline | ~0 | Dropped |
| P04 | Cross-scale edge-directed prediction, per-block strength | Offline | -0.28% photos, -0.60% textures | Small |
| P05 | AVIF-style intra prediction, finest detail (8x8 blocks) | Offline vs FUL's coarse prediction | Intra alone +72% photos / +61% textures; hybrid -0.6% / -1.6% (brick -7.1%) | Dropped |
| P06 | Intra prediction at medium levels (16-64 px blocks) | Offline | Intra alone +13% to +49%; hybrid 0 to -2.7% of those levels | Dropped |
| P07 | DNA-style pattern reuse (copy repeated 4x4 coefficient blocks) | Offline | Photos ~0.5% of file; starfield ~5% | Lead for screenshots/text only |

## Quantisation

| ID | Idea | Measured | Result | Verdict |
|---|---|---|---|---|
| Q01 | Reconstruction offset 0.4 instead of 0.5 | v3 search | ~10% | Adopted |
| Q02 | Diagonal (HH) steps x1.5-2.0 | Weight searches | Gain (x2 best with 9/7) | Adopted |
| Q03 | Finest luma steps x1.25 | Weight search | ~1 point | Adopted |
| Q04 | Brightness band weights by contrast sensitivity | v3 ablation | Worse | Dropped |
| Q05 | Texture masking (coarser steps in busy blocks) | v3 ablation | +19.3% vs JPEG instead of +13.5% (worse) | Dropped (optional switch) |
| Q06 | Weber's-law steps by brightness | Within Q05 | Worse | Dropped |
| Q07 | Fixed-rule dropping of sparse 4x4 groups (20-bit guess) | v3 ablation | ~4% | Superseded by Q08 |
| Q08 | Real-cost RDO, rounding down only | Benchmark | +0.2 / +1.3 points | Superseded |
| Q09 | Real-cost RDO with rounding up allowed (lambda 0.12) | Benchmark | +1.6 / +1.7 points; lambda 0.08 and 0.18 worse | Adopted |
| Q10 | Dependent quantisation (4-state trellis), eta 0.5-1.2 | Benchmark | eta 1.0 best: -34.7% / -28.4% (from -33.0% / -24.8%); confirmed at equal PSNR | Adopted |
| Q11 | Power-law companding of large coefficients | Real codec | gamma 0.85: +-0.2 points; gamma 0.7: -2 points | Dropped |
| Q12 | Finer steps for coarse levels (x0.6) | Benchmark | SSIM -2.4 points, MS-SSIM +1 point | Dropped |
| Q13 | Smooth tilt of steps toward coarse levels (0.8 per level) | Benchmark | SSIM -2.2, MS-SSIM +2.0 (pure trade-off) | Dropped |
| Q14 | Less eager RDO at coarse levels (lambda x0.5 per level) | Benchmark | Both metrics worse | Dropped |
| Q15 | Conditional-mean (centroid) dequantisation, encoder-sent offsets | Real codec | +-0.1% | Dropped (already near optimal) |
| Q16 | Simulated annealing for quantisation | Reasoning | Cannot beat the exact Viterbi search | Dropped |

## Entropy coding

| ID | Idea | Measured | Result | Verdict |
|---|---|---|---|---|
| E01 | Context-modelled adaptive rANS (v4) replacing Rice codes | Benchmark | 11-45% smaller at same quality; -14.9% vs JPEG (from -4.5%) | Adopted |
| E02 | 16-symbol vs 32-symbol token alphabet | Benchmark | +0.2% size, 7% faster | Adopted (16) |
| E03 | Sign contexts (left and upper sign) | Offline, then v7 | -1.31% file | Adopted |
| E04 | First low bit coded in a model | Offline, then v7 | -0.30% file | Adopted |
| E05 | Compact header (rebuilt weights, LEB128 table) | v7 | Most of the 5% overhead on thumbnails | Adopted |
| E06 | Edge-aware / orientation-weighted contexts | Offline | +0.38% (worse) | Dropped |
| E07 | Diagonal neighbours in the context | Offline | -0.15% | Dropped |
| E08 | Orientation axis (sibling bands) folded into neighbour sum | Offline | +0.55% average (varies by image) | Dropped |
| E09 | Learned starting probabilities (priors) | Offline -2.75%; real coder | Real: -0.84% held-out, -1.37% textures; confidence 0 best | Adopted (v8) |
| E10 | Texture-class split of models | Offline | +1.33% without priors; no gain beyond priors | Dropped |
| E11 | Brightness-range split of models | Offline | +5.11% without priors; -1.39% with (worse than priors alone) | Dropped |
| E12 | Gaussian-scale-mixture context: one continuous spread estimate from neighbours + parent + siblings, finely binned | Offline; then real coder (v9) vs v8, same pixels | Offline -2.26% photos / -1.37% textures; real: -1.31% tuning, -0.69% held-out, -0.77% textures (brick +0.6%); decode +8% time | Adopted (v9) |
| E13 | Huffman coding instead of rANS | Offline, per symbol | Roughly 3x the bits (Huffman needs >= 1 bit per symbol) | Dropped |
| E14 | Power-law (Pareto) probability model per context | Offline | +9.2% vs counting what occurs (Laplacian +7.3%) | Dropped (FUL already counts, it assumes no shape) |
| E15 | Hidden Markov trees across scale | Covered by E12 | Parent and sibling dependencies captured there | Folded into E12 |
| E16 | E12's context without priors (flat start) | Real coder vs v8 | +0.7% to +1.4% (worse): finer bins learn slowly from scratch | Context needs its priors |

## Post-processing

| ID | Idea | Measured | Result | Verdict |
|---|---|---|---|---|
| F01 | Encoder-chosen 5x5 bilateral post-filter, per 32x32 block | Benchmark | +2.6 points photos, +4.2 textures; MS-SSIM +2.2 / +3.6; visually cleaner | Adopted (v8) |
| F02 | Least-squares restoration (Wiener / ALF-style) 7x7 filter, 1 or 9 classes | Real codec | SSIM +0.1-0.25 points; MS-SSIM slightly worse; medium-scale error unchanged | Dropped |
| F03 | Texture synthesis for the most textured blocks | Real codec + visual | Worse SSIM and MS-SSIM; grass looked speckled | Dropped |

## Diagnoses (measurements that explain, rather than test an idea)

| ID | Question | Result |
|---|---|---|
| D01 | Where FUL's bytes go | Finest two luma levels ~59%, colour ~20%, raw low bits ~5% |
| D02 | Texture share | Most textured 20% of the image = ~36% of the file |
| D03 | Where AVIF beats FUL (equal size) | Medium scales: FUL has 2-4x more error at 4-16 px; similar at finest; less at coarsest; same pattern in smooth, edge and textured regions |
| D04 | Is FUL's entropy coder efficient? | Within 1-3% of an ideal coder for its own model |
| D05 | MS-SSIM vs SSIM gap | Largely FUL's colour spending (MS-SSIM here is brightness only) |
| D06 | Real camera photos | FUL best on SSIM (-46%); JPEG XL best on MS-SSIM (-29% vs FUL -17%) |

---

## Open leads (in priority order)

1. **C06** Colour budget (large MS-SSIM gain for small SSIM cost; needs a
   perceptual decision).
2. **C05** Colour from brightness (-0.45%).
3. **T11** Wavelet packets for regular textures; **P07** pattern reuse for
   screenshots and text (both need suitable test images first).
4. Tuning E12's spread estimate (neighbour weights, bin edges) from data rather
   than by hand.
