# Contributing to FUL

FUL is an experimental image format. The goal is the smallest files at equal
visual quality, with a light, exact decoder. Every change is judged by
measurement, and negative results are as welcome as positive ones.

## Read first

1. `docs/FORMAT.md`: how the format works today.
2. `docs/EXPERIMENTS.md`: what has already been measured. If your idea, or a
   close variant, is there, explain what is genuinely different.

## Workflow

1. Branch from `main` (one idea per branch).
2. Make one change at a time and measure it alone before combining.
3. Open a pull request with the report described below.
4. Add the measured result to `docs/EXPERIMENTS.md` (wins and losses alike).

New format versions go in a new file (`codec_v10.py`, ...). Never change how an
existing version decodes: files written by it exist and must keep opening.
Register new versions in `ful.py`.

## Constraints

- Decoding stays light: the encoder may search heavily, the decoder only
  follows stored decisions. No neural networks in the decoder.
- Integer arithmetic in the decoder, so the GPU decoder stays bit-identical to
  the CPU decoder (`python gpu_decode.py` checks this).
- Quality 100 stays exactly lossless (each codec's self-test checks this).
- Progressive decoding from a file prefix keeps working.
- Report decode and encode time changes in milliseconds.

## Measuring

    python bench/run.py --codec v10               # tuning + held-out sets
    python bench/run.py --codec v10 --formats     # include WebP, AVIF, JPEG XL, JPEG 2000
    python bench/run.py --codec v10 --dir photos  # your own held-out photos

The primary number is file size relative to JPEG at equal colour SSIM
(YCbCr, weighted 6:1:1), interpolated at the SSIM values JPEG reaches at
qualities 50, 75 and 90.

- Report tuning and held-out results separately; never tune on held-out images.
- Confirm claimed gains with a second perceptual metric (for example MS-SSIM,
  SSIMULACRA 2 or Butteraugli), since SSIM alone can be gamed.
- When comparing with AVIF, state the encoder speed used (the reference
  numbers use libaom speed 6; its slowest speed compresses better).

## Report format (pull request description)

- **Idea:** one paragraph, with the maths. Cite prior work and check the
  reference exists.
- **Results:** tuning set, held-out set, second metric, encode and decode time.
- **Verdict:** adopt, needs work, or drop.

## Current state (v9)

About 35% smaller than JPEG on photos and 32% on textures at equal SSIM.
Always report a second metric too: at equal MS-SSIM FUL photos are currently
~5% larger than JPEG (see README, Open issues). Reference points: WebP ~30%, JPEG XL ~28%, JPEG 2000 ~21%,
AVIF ~46%.

Already tried: see "Already tried" in the README's history section.
