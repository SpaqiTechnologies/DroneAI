"""Image encoders.

Pure stdlib: PPM (P6) and PNG. Optional: JPEG via Pillow if installed.

PNG is the right default for inspectable artifacts on Windows because
File Explorer + every browser can preview them natively, whereas PPM
requires GIMP / IrfanView / ImageMagick.
"""

from __future__ import annotations

import io
import os
import struct
import zlib
from typing import Optional

try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    PIL_AVAILABLE = False


_PPM_MAGIC = b"P6"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def encode_ppm(rgb: bytes, width: int, height: int) -> bytes:
    """Encode raw RGB bytes (3 * width * height) as a binary PPM (P6) blob."""
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError(
            f"rgb length {len(rgb)} does not match {width}x{height}*3 = {expected}"
        )
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb


def write_ppm(path: str, rgb: bytes, width: int, height: int) -> int:
    """Write a PPM image. Returns bytes written."""
    blob = encode_ppm(rgb, width, height)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


def write_jpeg_if_possible(
    path: str, rgb: bytes, width: int, height: int, quality: int = 85
) -> Optional[int]:
    """Write a JPEG using Pillow if installed. Returns bytes written, or None."""
    if not PIL_AVAILABLE:
        return None
    img = Image.frombytes("RGB", (width, height), rgb)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data)


def is_ppm(path: str) -> bool:
    """Quick sanity check that a file is a P6 PPM (used by tests)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == _PPM_MAGIC
    except OSError:
        return False


def is_png(path: str) -> bool:
    """Quick sanity check that a file is a PNG."""
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == _PNG_MAGIC
    except OSError:
        return False


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build one PNG chunk: length || type || data || crc32(type || data)."""
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def encode_png(
    rgb: bytes,
    width: int,
    height: int,
    compression_level: int = 6,
) -> bytes:
    """Encode raw RGB bytes as a PNG.

    Uses filter type 0 ("None") for every scanline. Spec-compliant, no
    external deps. Output is decodable by any PNG reader.
    """
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError(
            f"rgb length {len(rgb)} does not match {width}x{height}*3 = {expected}"
        )
    if not (0 <= compression_level <= 9):
        raise ValueError("compression_level must be 0..9")

    stride = width * 3
    # Prepend filter byte (0 = None) to each scanline
    rows = bytearray(height * (1 + stride))
    out = 0
    for y in range(height):
        rows[out] = 0
        out += 1
        src = y * stride
        rows[out:out + stride] = rgb[src:src + stride]
        out += stride

    compressed = zlib.compress(bytes(rows), compression_level)

    # IHDR: width(4) height(4) bit_depth(1) color_type(1) compression(1)
    # filter(1) interlace(1)
    #   color_type 2 = truecolor RGB
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    return (
        _PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def write_png(
    path: str,
    rgb: bytes,
    width: int,
    height: int,
    compression_level: int = 6,
) -> int:
    """Write a PNG file. Returns bytes written."""
    blob = encode_png(rgb, width, height, compression_level=compression_level)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)
