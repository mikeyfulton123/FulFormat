"""
FUL from the command line.

    python ful_cli.py encode photo.jpg                 # writes photo.ful (quality 75)
    python ful_cli.py encode photos/ -q 80 -o out/     # a whole folder
    python ful_cli.py decode photo.ful                 # writes photo_ful.png to view anywhere
    python ful_cli.py info photo.ful                   # version, size, settings

Encoding uses the current format (v8). Decoding opens any FUL version.
Photos are rotated using their EXIF orientation; HEIC needs `pip install pillow-heif`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import ful

PHOTO_TYPES = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp"}


def read_photo(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(im).convert("RGB")))


def human(n: float) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.2f} MB"


def encode(paths, quality: int, out_dir: Path | None):
    import codec_v8
    total_in = total_out = 0
    for p in paths:
        try:
            rgb = read_photo(p)
        except Exception as exc:
            print(f"  skip {p.name}: {exc}")
            continue
        t0 = time.perf_counter()
        data = codec_v8.encode(rgb, quality)
        secs = time.perf_counter() - t0
        target = (out_dir or p.parent) / (p.stem + ".ful")
        target.write_bytes(data)
        size_in = p.stat().st_size
        total_in += size_in
        total_out += len(data)
        noise = codec_v8.estimate_noise_v2(rgb)
        print(f"  {p.name}: {rgb.shape[1]}x{rgb.shape[0]}, {human(size_in)} -> {target.name} "
              f"{human(len(data))} ({len(data) / size_in * 100:.0f}%), {secs:.1f} s, noise {noise:.1f}"
              + (" (denoised)" if codec_v8.auto_denoise_strength_v2(noise) > 0 else ""))
    if total_in:
        print(f"Total: {human(total_in)} -> {human(total_out)} ({total_out / total_in * 100:.0f}%)")


def decode(paths, out_dir: Path | None):
    from PIL import Image
    for p in paths:
        rgb = ful.decode(p.read_bytes())
        target = (out_dir or p.parent) / (p.stem + "_ful.png")
        Image.fromarray(rgb).save(target)
        print(f"  {p.name} -> {target.name} ({rgb.shape[1]}x{rgb.shape[0]})")


def info(paths):
    for p in paths:
        data = p.read_bytes()
        h = ful.read_header(data)
        keys = [k for k in ("quality", "wavelet", "directional", "dq", "filter", "grain", "levels") if k in h]
        print(f"  {p.name}: FUL v{h['version']}, {h['width']}x{h['height']}, {human(len(data))}, "
              + ", ".join(f"{k} {h[k]}" for k in keys))


def expand(items, suffixes):
    out = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in suffixes)
        elif p.exists():
            out.append(p)
        else:
            print(f"  not found: {item}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="photos -> .ful")
    e.add_argument("inputs", nargs="+")
    e.add_argument("-q", "--quality", type=int, default=75, help="1-100 (100 = lossless), default 75")
    e.add_argument("-o", "--out", type=Path, help="output folder (default: next to each photo)")
    d = sub.add_parser("decode", help=".ful -> PNG")
    d.add_argument("inputs", nargs="+")
    d.add_argument("-o", "--out", type=Path)
    i = sub.add_parser("info", help="describe .ful files")
    i.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    if getattr(args, "out", None):
        args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "encode":
        encode(expand(args.inputs, PHOTO_TYPES), args.quality, args.out)
    elif args.cmd == "decode":
        decode(expand(args.inputs, {".ful"}), args.out)
    else:
        info(expand(args.inputs, {".ful"}))


if __name__ == "__main__":
    sys.exit(main())
