"""
Version-aware entry point for FUL files.

Every .ful file starts with the magic bytes b"FUL\\x00" followed by a version
byte, so one function can open files written by any codec version.
"""
from __future__ import annotations

import numpy as np

import codec_v1
import codec_v2
import codec_v3
import codec_v4
import codec_v5
import codec_v6
import codec_v7
import codec_v8
import codec_v9

MAGIC = b"FUL\x00"
LATEST = 9


def peek_version(data: bytes) -> int:
    if len(data) < 5 or data[:4] != MAGIC:
        raise ValueError("Not a FUL file (bad magic bytes)")
    return data[4]


def read_header(data: bytes) -> dict:
    version = peek_version(data)
    if version == 1:
        return {"version": 1, **codec_v1.read_header(data)}
    if version == 2:
        return codec_v2.read_header(data)
    if version == 3:
        return codec_v3.read_header(data)
    if version == 4:
        return codec_v4.read_header(data)
    if version == 5:
        return codec_v5.read_header(data)
    if version == 6:
        return codec_v6.read_header(data)
    if version == 7:
        return codec_v7.read_header(data)
    if version == 8:
        return codec_v8.read_header(data)
    if version == 9:
        return codec_v9.read_header(data)
    raise ValueError(f"Unsupported FUL version {version}")


def encode(rgb: np.ndarray, version: int = LATEST, **settings) -> bytes:
    if version == 1:
        settings.pop("tile_size", None)
        return codec_v1.encode(rgb, **settings)
    if version == 2:
        return codec_v2.encode(rgb, **settings)
    if version == 3:
        return codec_v3.encode(rgb, **settings)
    if version == 4:
        return codec_v4.encode(rgb, **settings)
    if version == 5:
        settings.pop("tile_size", None)
        return codec_v5.encode(rgb, **settings)
    if version == 6:
        settings.pop("tile_size", None)
        return codec_v6.encode(rgb, **settings)
    if version == 7:
        settings.pop("tile_size", None)
        return codec_v7.encode(rgb, **settings)
    if version == 8:
        settings.pop("tile_size", None)
        return codec_v8.encode(rgb, **settings)
    if version == 9:
        settings.pop("tile_size", None)
        return codec_v9.encode(rgb, **settings)
    raise ValueError(f"Unsupported FUL version {version}")


def decode(data: bytes, layers: int | None = None, grain: float = 1.0) -> np.ndarray:
    """Decode any FUL file. `layers` applies to v2+ (progressive); `grain`
    scales regenerated grain in v4 to v9 files (0 = off)."""
    version = peek_version(data)
    if version == 1:
        return codec_v1.decode(data)
    if version == 2:
        return codec_v2.decode(data, layers)
    if version == 3:
        return codec_v3.decode(data, layers)
    if version == 4:
        return codec_v4.decode(data, layers, grain)
    if version == 5:
        return codec_v5.decode(data, layers, grain)
    if version == 6:
        return codec_v6.decode(data, layers, grain)
    if version == 7:
        return codec_v7.decode(data, layers, grain)
    if version == 8:
        return codec_v8.decode(data, layers, grain)
    if version == 9:
        return codec_v9.decode(data, layers, grain)
    raise ValueError(f"Unsupported FUL version {version}")
