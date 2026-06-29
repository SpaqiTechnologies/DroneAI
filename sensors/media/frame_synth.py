"""Synthesize RGB pixel bytes for a simulated camera frame.

Real cameras hand us a buffer; in simulation we need *something* on disk so
the recording/snapshot artifacts are inspectable. We render a coarse test
pattern with embedded bounding boxes for any detections passed in.
"""

from __future__ import annotations

from typing import Iterable, Tuple


_VISION_MODE_TINT = {
    "normal":              (1.00, 1.00, 1.00),
    "thermal":             (1.30, 0.55, 0.20),
    "night_vision":        (0.30, 1.20, 0.30),
    "obstacle_detection":  (1.00, 1.10, 0.85),
    "marker_detection":    (1.05, 1.05, 1.20),
}


def synthesize_rgb_bytes(
    width: int,
    height: int,
    vision_mode: str = "normal",
    detections: Iterable[Tuple[int, int, int, int]] = (),
    timestamp: float = 0.0,
    coarse_step: int = 8,
) -> bytes:
    """Build a width*height*3 byte buffer with a gradient + detection boxes.

    `detections` is an iterable of (x, y, w, h) in pixel coordinates.
    `coarse_step` controls the gradient block size; lower = smoother but slower.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    tint_r, tint_g, tint_b = _VISION_MODE_TINT.get(vision_mode, (1.0, 1.0, 1.0))
    t_offset = int(timestamp * 32) & 0xFF

    box_set: set[tuple[int, int]] = set()
    edge_set: set[tuple[int, int]] = set()
    for (bx, by, bw, bh) in detections:
        if bw <= 0 or bh <= 0:
            continue
        x0 = max(0, bx)
        y0 = max(0, by)
        x1 = min(width - 1, bx + bw - 1)
        y1 = min(height - 1, by + bh - 1)
        if x0 > x1 or y0 > y1:
            continue
        for xx in range(x0, x1 + 1):
            edge_set.add((xx, y0))
            edge_set.add((xx, y1))
        for yy in range(y0, y1 + 1):
            edge_set.add((x0, yy))
            edge_set.add((x1, yy))
        for xx in range(x0, x1 + 1):
            for yy in range(y0, y1 + 1):
                box_set.add((xx, yy))

    buf = bytearray(width * height * 3)
    step = max(1, coarse_step)

    for y in range(height):
        base_g = ((y // step) * step * 255 // max(1, height - 1)) & 0xFF
        row_off = y * width * 3
        for x in range(width):
            base_r = ((x // step) * step * 255 // max(1, width - 1)) & 0xFF
            base_b = (t_offset + ((x ^ y) // step)) & 0xFF
            r = int(base_r * tint_r) & 0xFF
            g = int(base_g * tint_g) & 0xFF
            b = int(base_b * tint_b) & 0xFF
            if (x, y) in edge_set:
                r, g, b = 255, 32, 32
            elif (x, y) in box_set and ((x + y) & 7) == 0:
                r, g, b = 255, 255, 32
            o = row_off + x * 3
            buf[o] = r
            buf[o + 1] = g
            buf[o + 2] = b

    return bytes(buf)
