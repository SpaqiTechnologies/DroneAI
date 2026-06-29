"""Tests for the camera media I/O layer."""

import json
import os
import tempfile
import time

import pytest

from sensors.media import (
    MediaStorage,
    VideoRecorder,
    RecorderState,
    synthesize_rgb_bytes,
    encode_ppm,
    encode_png,
    write_ppm,
    write_png,
    is_ppm,
    is_png,
)
from sensors.camera_sensor import CameraSensor


# ----------------------------- encoder --------------------------------------


def test_synthesize_rgb_bytes_size_matches_dimensions() -> None:
    buf = synthesize_rgb_bytes(64, 36)
    assert len(buf) == 64 * 36 * 3


def test_synthesize_rgb_bytes_rejects_bad_dims() -> None:
    with pytest.raises(ValueError):
        synthesize_rgb_bytes(0, 10)
    with pytest.raises(ValueError):
        synthesize_rgb_bytes(10, -1)


def test_encode_ppm_has_p6_header() -> None:
    rgb = b"\x00\x00\x00" * 4
    blob = encode_ppm(rgb, 2, 2)
    assert blob.startswith(b"P6\n")
    assert b"2 2" in blob.split(b"\n", 3)[1]
    assert blob.endswith(rgb)


def test_encode_ppm_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        encode_ppm(b"\x00\x01", 4, 4)


def test_write_ppm_creates_parent_dir(tmp_path) -> None:
    target = tmp_path / "nested" / "deep" / "x.ppm"
    rgb = synthesize_rgb_bytes(8, 8)
    n = write_ppm(str(target), rgb, 8, 8)
    assert target.exists()
    assert n > 8 * 8 * 3
    assert is_ppm(str(target))


def test_encode_png_has_signature_and_chunks() -> None:
    rgb = synthesize_rgb_bytes(16, 12)
    blob = encode_png(rgb, 16, 12)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR chunk type at offset 12 (length=4 bytes before)
    assert blob[12:16] == b"IHDR"
    assert b"IDAT" in blob
    assert b"IEND" in blob


def test_encode_png_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        encode_png(b"\x00" * 5, 4, 4)


def test_write_png_creates_valid_file(tmp_path) -> None:
    target = tmp_path / "image.png"
    rgb = synthesize_rgb_bytes(32, 24)
    n = write_png(str(target), rgb, 32, 24)
    assert target.exists()
    assert n > 0
    assert is_png(str(target))
    assert not is_ppm(str(target))


def test_png_is_decodable_by_zlib_round_trip(tmp_path) -> None:
    import struct
    import zlib
    rgb = synthesize_rgb_bytes(8, 6)
    target = tmp_path / "test.png"
    write_png(str(target), rgb, 8, 6)
    data = target.read_bytes()
    # IHDR at offset 8; length(4) | type(4) | data(13) | crc(4)
    ihdr = data[16:29]
    w, h, depth, ctype = struct.unpack(">IIBB", ihdr[:10])
    assert (w, h, depth, ctype) == (8, 6, 8, 2)
    # Walk chunks, collect all IDAT payloads, decompress
    pos = 8
    idats: list[bytes] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if chunk_type == b"IDAT":
            idats.append(payload)
        if chunk_type == b"IEND":
            break
        pos += 12 + length
    raw = zlib.decompress(b"".join(idats))
    stride = 8 * 3
    rebuilt = bytearray()
    for y in range(6):
        row_start = y * (stride + 1)
        assert raw[row_start] == 0  # filter type "None"
        rebuilt += raw[row_start + 1:row_start + 1 + stride]
    assert bytes(rebuilt) == rgb


# ----------------------------- storage --------------------------------------


def test_media_storage_creates_dirs(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    assert os.path.isdir(storage.photos_dir)
    assert os.path.isdir(storage.videos_dir)
    assert storage.session_id


def test_media_storage_allocates_unique_paths(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    a = storage.allocate_photo_path()
    b = storage.allocate_photo_path()
    assert a != b
    assert a.endswith(".ppm")


def test_media_storage_registers_and_persists_index(tmp_path) -> None:
    root = str(tmp_path / "media")
    storage = MediaStorage(root=root)
    path = storage.allocate_photo_path()
    write_ppm(path, synthesize_rgb_bytes(8, 8), 8, 8)
    from sensors.media import MediaArtifact
    storage.register(MediaArtifact(
        kind="photo", path=path, width=8, height=8,
        timestamp=time.time(), bytes_written=os.path.getsize(path),
    ))
    assert os.path.isfile(os.path.join(root, "index.json"))
    storage2 = MediaStorage(root=root)
    arts = storage2.list_artifacts()
    assert len(arts) == 1
    assert arts[0].kind == "photo"


# ----------------------------- recorder -------------------------------------


def test_video_recorder_writes_manifest_and_frames(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    rec = VideoRecorder(storage=storage, width=16, height=12, target_fps=30.0)
    rec.start()
    rgb = synthesize_rgb_bytes(16, 12)
    for _ in range(5):
        assert rec.add_frame(rgb)
    summary = rec.stop()
    assert summary.frame_count == 5
    assert summary.width == 16 and summary.height == 12
    assert os.path.isfile(summary.manifest_path)
    with open(summary.manifest_path) as fh:
        manifest = json.load(fh)
    assert manifest["frame_count"] == 5
    assert manifest["frame_format"] == "png"
    for path in summary.frame_paths:
        assert is_png(path)


def test_video_recorder_supports_ppm_frame_format(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    rec = VideoRecorder(storage=storage, width=16, height=12, target_fps=10.0, frame_format="ppm")
    rec.start()
    rec.add_frame(synthesize_rgb_bytes(16, 12))
    summary = rec.stop()
    assert summary.frame_count == 1
    assert is_ppm(summary.frame_paths[0])


def test_video_recorder_rejects_unknown_frame_format(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    with pytest.raises(ValueError):
        VideoRecorder(storage=storage, width=8, height=8, frame_format="webp")


def test_video_recorder_does_not_record_when_idle(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    rec = VideoRecorder(storage=storage, width=8, height=8, target_fps=10.0)
    assert rec.state == RecorderState.IDLE
    assert not rec.add_frame(synthesize_rgb_bytes(8, 8))


def test_video_recorder_max_frames_cap(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    rec = VideoRecorder(storage=storage, width=8, height=8, target_fps=10.0, max_frames=3)
    rec.start()
    rgb = synthesize_rgb_bytes(8, 8)
    accepted = sum(1 for _ in range(10) if rec.add_frame(rgb))
    rec.stop()
    assert accepted == 3


# ----------------------------- integration ----------------------------------


def test_camera_snapshot_writes_real_file(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    cam = CameraSensor(media_storage=storage, synth_resolution=(32, 18))
    cam.start()
    ok, msg = cam.take_snapshot()
    assert ok, msg
    arts = storage.list_artifacts(kind="photo")
    assert len(arts) == 1
    art = arts[0]
    # Primary artifact is PNG; PPM sidecar is also present.
    assert is_png(art.path)
    ppm_path = art.extra.get("ppm_path")
    assert ppm_path and is_ppm(ppm_path)
    assert art.bytes_written > 0


def test_camera_recording_writes_bundle(tmp_path) -> None:
    storage = MediaStorage(root=str(tmp_path / "media"))
    cam = CameraSensor(media_storage=storage, synth_resolution=(32, 18), fps=20)
    cam.start()
    cam.start_streaming()
    ok, _ = cam.start_recording()
    assert ok
    time.sleep(0.35)
    ok, _ = cam.stop_recording()
    assert ok
    summary = cam.get_last_recording_summary()
    assert summary is not None
    assert summary["frame_count"] >= 1
    assert os.path.isfile(summary["manifest_path"])
    cam.stop_streaming()
    cam.stop()


def test_camera_without_storage_returns_path_stub() -> None:
    cam = CameraSensor()
    cam.start()
    ok, msg = cam.take_snapshot()
    assert ok
    assert "Snapshot saved" in msg
    cam.stop()
