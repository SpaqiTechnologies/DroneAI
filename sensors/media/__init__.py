"""Real on-disk media I/O for the camera sensor.

Provides photo capture (PPM, JPEG if Pillow is available) and video recording
(directory-bundle with per-frame PPM + manifest.json) using stdlib only.
"""

from sensors.media.storage import MediaStorage, MediaArtifact
from sensors.media.encoder import (
    encode_ppm,
    write_ppm,
    encode_png,
    write_png,
    write_jpeg_if_possible,
    is_ppm,
    is_png,
    PIL_AVAILABLE,
)
from sensors.media.frame_synth import synthesize_rgb_bytes
from sensors.media.recorder import VideoRecorder, RecorderState, RecordingSummary

__all__ = [
    "MediaStorage",
    "MediaArtifact",
    "VideoRecorder",
    "RecorderState",
    "RecordingSummary",
    "encode_ppm",
    "write_ppm",
    "encode_png",
    "write_png",
    "write_jpeg_if_possible",
    "is_ppm",
    "is_png",
    "synthesize_rgb_bytes",
    "PIL_AVAILABLE",
]
