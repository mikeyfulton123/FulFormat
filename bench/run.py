"""
FUL benchmark: file size at equal image quality, relative to JPEG.

For each test image, FUL (and optionally other formats) is encoded across a
range of quality settings. Each format's size-vs-SSIM curve is interpolated
at the SSIM values JPEG reaches at qualities 50, 75 and 90, and the result is
reported as file size relative to JPEG (negative = smaller than JPEG).

Usage (from the repository root):
    python bench/run.py                       # FUL v6 on tuning + held-out images
    python bench/run.py --formats             # also WebP, AVIF, JPEG XL, JPEG 2000
    python bench/run.py --codec v5            # an older FUL version
    python bench/run.py --dir path/to/photos  # your own held-out images
    python bench/run.py --quick               # fewer quality steps, faster

Results are printed and saved as JSON in bench/results/.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import formats  # noqa: E402
import metrics  # noqa: E402

TUNING = ["astronaut", "coffee", "chelsea", "rocket"]
HELD_OUT = ["hubble_deep_field", "immunohistochemistry"]
JPEG_REFERENCE = (50, 75, 90)


def load_builtin(names):
    from skimage import data
    return {n: np.ascontiguousarray(getattr(data, n)()[..., :3]) for n in names}


def load_dir(path):
    from PIL import Image
    out = {}
    for f in sorted(Path(path).iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            out[f.name] = np.asarray(Image.open(f).convert("RGB"))
    return out


def curve(img, encode, decode, settings):
    """(bits per pixel, SSIM) points, sorted by SSIM."""
    n = img.shape[0] * img.shape[1]
    pts = []
    for s in settings:
        data = encode(img, s)
        pts.append((len(data) * 8 / n, metrics.ssim(img, decode(data))))
    return sorted(pts, key=lambda p: p[1])


def bpp_at(pts, target):
    """Interpolated bits per pixel at a target SSIM, or None if out of range."""
    ssims = [p[1] for p in pts]
    if target < ssims[0] or target > ssims[-1]:
        return None
    return float(np.interp(target, ssims, [p[0] for p in pts]))


def ful_codec(version):
    codec = importlib.import_module(f"codec_{version}")

    def encode(img, q):
        return codec.encode(img, int(q))

    def decode(data):
        try:
            return codec.decode(data, grain=0)       # measure without grain
        except TypeError:
            return codec.decode(data)
    return encode, decode


def run(images, contenders, quick):
    jpeg = next(f for f in formats.available_formats() if f.name == "JPEG")
    per_image = {}
    for name, img in images.items():
        n = img.shape[0] * img.shape[1]
        refs = []
        for jq in JPEG_REFERENCE:
            d = jpeg.encode(img, jq)
            refs.append((metrics.ssim(img, jpeg.decode(d)), len(d) * 8 / n))
        per_image[name] = {}
        for label, (enc, dec, settings) in contenders.items():
            t0 = time.perf_counter()
            pts = curve(img, enc, dec, settings[::2] if quick else settings)
            ratios = [b / jb for s, jb in refs if (b := bpp_at(pts, s)) is not None]
            per_image[name][label] = {
                "vs_jpeg_percent": (float(np.mean(ratios)) - 1) * 100 if ratios else None,
                "points": len(ratios), "seconds": round(time.perf_counter() - t0, 1)}
        row = "  ".join(f"{k} {v['vs_jpeg_percent']:+.1f}%" if v["vs_jpeg_percent"] is not None
                        else f"{k} n/a" for k, v in per_image[name].items())
        print(f"  {name:24} {row}", flush=True)
    return per_image


def summarise(per_image, names):
    labels = {k for v in per_image.values() for k in v}
    out = {}
    for label in sorted(labels):
        vals = [per_image[n][label]["vs_jpeg_percent"] for n in names
                if per_image[n].get(label, {}).get("vs_jpeg_percent") is not None]
        out[label] = float(np.mean(vals)) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", default="v6", help="FUL codec version to test (v1-v6)")
    ap.add_argument("--formats", action="store_true", help="also test WebP, AVIF, JPEG XL, JPEG 2000")
    ap.add_argument("--dir", help="folder of extra held-out images")
    ap.add_argument("--quick", action="store_true", help="half the quality steps")
    args = ap.parse_args()

    enc, dec = ful_codec(args.codec)
    contenders = {f"FUL {args.codec}": (enc, dec, list(range(10, 100, 3)))}
    if args.formats:
        for f in formats.available_formats():
            if f.name == "JPEG":
                continue
            idx = np.unique(np.linspace(0, len(f.settings) - 1, 24).astype(int))
            contenders[f.name] = (f.encode, f.decode, [f.settings[i] for i in idx])

    groups = {"tuning": load_builtin(TUNING), "held-out": load_builtin(HELD_OUT)}
    if args.dir:
        groups["your images"] = load_dir(args.dir)

    report = {"codec": args.codec, "quick": args.quick, "groups": {}}
    for group, images in groups.items():
        print(f"\n{group} ({len(images)} images): file size vs JPEG at equal SSIM")
        per_image = run(images, contenders, args.quick)
        summary = summarise(per_image, list(images))
        report["groups"][group] = {"per_image": per_image, "average": summary}
        print("  average: " + "  ".join(f"{k} {v:+.1f}%" for k, v in summary.items() if v is not None))

    out_dir = ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.codec}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
