"""Smoke test for the media I/O layer."""

import os
import sys
import json
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.media import MediaStorage, PIL_AVAILABLE
from sensors.media.encoder import is_ppm
from sensors.camera_sensor import CameraSensor


def main() -> int:
    print("PIL_AVAILABLE:", PIL_AVAILABLE)
    with tempfile.TemporaryDirectory() as td:
        storage = MediaStorage(root=td)
        cam = CameraSensor(media_storage=storage, synth_resolution=(64, 36))
        ok, msg = cam.start()
        print("start:", ok, msg)

        ok, msg = cam.take_snapshot()
        print("snapshot:", ok, msg)

        cam.start_streaming()
        cam.start_recording()
        time.sleep(0.5)
        cam.stop_recording()
        cam.stop_streaming()

        summary = cam.get_last_recording_summary()
        print("recording summary:", json.dumps(summary, indent=2, default=str))

        arts = storage.list_artifacts()
        print("artifact count:", len(arts), "total_bytes:", storage.total_bytes())
        for a in arts:
            print("  ", a.kind, os.path.basename(a.path), a.bytes_written, "bytes")

        photos = storage.list_artifacts(kind="photo")
        if photos:
            ppm_path = photos[0].extra.get("ppm_path") or photos[0].path
            print("photo is_ppm:", is_ppm(ppm_path))

        videos = storage.list_artifacts(kind="video")
        if videos:
            manifest = videos[0].extra.get("manifest")
            with open(manifest, "r", encoding="utf-8") as fh:
                print("manifest frame_count:", json.load(fh).get("frame_count"))

        cam.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
