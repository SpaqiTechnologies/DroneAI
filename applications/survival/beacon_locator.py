"""RSSI-based emergency beacon locator.

Personnel-recovery use case: a downed soldier broadcasts an emergency
beacon. As a drone flies, it records RSSI samples at known positions and
estimates the beacon's location by trilateration.

Algorithm:
  - RSSI samples are converted to distance estimates via the
    log-distance path loss model.
  - With >= 3 samples that are not collinear, we solve the
    over-determined linear system that linearises the trilateration
    equations (subtracting one reference equation cancels the squared
    terms). For fewer samples we fall back to weighted centroid.
  - A covariance-style residual is reported so callers know whether the
    fix is tight.

Pure stdlib; no numpy dependency.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def _latlon_to_local_xy(
    point: Tuple[float, float], origin: Tuple[float, float]
) -> Tuple[float, float]:
    """Equirectangular projection (m) — fine for areas under ~5 km."""
    lat0 = math.radians(origin[0])
    dx = math.radians(point[1] - origin[1]) * 6371000.0 * math.cos(lat0)
    dy = math.radians(point[0] - origin[0]) * 6371000.0
    return dx, dy


def _local_xy_to_latlon(
    xy: Tuple[float, float], origin: Tuple[float, float]
) -> Tuple[float, float]:
    lat0 = math.radians(origin[0])
    dlat = (xy[1] / 6371000.0) * (180.0 / math.pi)
    dlon = (xy[0] / (6371000.0 * math.cos(lat0))) * (180.0 / math.pi)
    return origin[0] + dlat, origin[1] + dlon


def rssi_to_distance(
    rssi_dbm: float,
    tx_power_dbm: float = -40.0,
    path_loss_exponent: float = 2.5,
    reference_distance_m: float = 1.0,
) -> float:
    """Log-distance path loss model.

    ``tx_power_dbm`` is the RSSI at ``reference_distance_m`` from the
    transmitter (a calibration constant for the beacon hardware).
    """
    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent must be positive")
    if reference_distance_m <= 0:
        raise ValueError("reference_distance_m must be positive")
    exponent = (tx_power_dbm - rssi_dbm) / (10.0 * path_loss_exponent)
    return float(reference_distance_m * (10.0 ** exponent))


@dataclass
class BeaconReading:
    latitude: float
    longitude: float
    altitude: float
    rssi_dbm: float
    timestamp: float = field(default_factory=time.time)
    distance_m: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "rssi_dbm": self.rssi_dbm,
            "timestamp": self.timestamp,
            "distance_m": self.distance_m,
        }


@dataclass
class BeaconFix:
    latitude: float
    longitude: float
    confidence: float           # 0..1
    residual_m: float           # mean residual between predicted/observed distance
    sample_count: int
    method: str                 # "trilateration" | "centroid" | "single"
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "confidence": self.confidence,
            "residual_m": self.residual_m,
            "sample_count": self.sample_count,
            "method": self.method,
            "computed_at": self.computed_at,
        }


class BeaconLocator:
    """Accumulates RSSI samples and computes a beacon position fix."""

    DEFAULT_TX_POWER_DBM = -40.0
    DEFAULT_PLE = 2.5

    def __init__(
        self,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        path_loss_exponent: float = DEFAULT_PLE,
        min_sample_distance_m: float = 5.0,
        max_samples: int = 200,
    ) -> None:
        self._tx_power = float(tx_power_dbm)
        self._ple = float(path_loss_exponent)
        self._min_sample_distance_m = float(min_sample_distance_m)
        self._max_samples = int(max_samples)
        self._readings: List[BeaconReading] = []
        self._lock = threading.Lock()

    @property
    def readings(self) -> List[BeaconReading]:
        with self._lock:
            return list(self._readings)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._readings)

    def reset(self) -> None:
        with self._lock:
            self._readings.clear()

    def add_reading(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        rssi_dbm: float,
        timestamp: Optional[float] = None,
    ) -> BeaconReading:
        dist = rssi_to_distance(
            rssi_dbm,
            tx_power_dbm=self._tx_power,
            path_loss_exponent=self._ple,
        )
        reading = BeaconReading(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            rssi_dbm=rssi_dbm,
            timestamp=timestamp if timestamp is not None else time.time(),
            distance_m=dist,
        )
        with self._lock:
            if self._readings:
                last = self._readings[-1]
                if _haversine_m(
                    (last.latitude, last.longitude),
                    (latitude, longitude),
                ) < self._min_sample_distance_m:
                    self._readings[-1] = reading
                    return reading
            self._readings.append(reading)
            if len(self._readings) > self._max_samples:
                self._readings = self._readings[-self._max_samples :]
        return reading

    def compute_fix(self) -> Optional[BeaconFix]:
        with self._lock:
            readings = list(self._readings)
        if not readings:
            return None
        if len(readings) == 1:
            r = readings[0]
            return BeaconFix(
                latitude=r.latitude,
                longitude=r.longitude,
                confidence=0.15,
                residual_m=float(r.distance_m or 0.0),
                sample_count=1,
                method="single",
            )
        if len(readings) == 2:
            return self._centroid_fix(readings, method="centroid")
        fix = self._trilaterate(readings)
        if fix is not None:
            return fix
        return self._centroid_fix(readings, method="centroid")

    # --------------------------- math --------------------------------------

    def _centroid_fix(self, readings: List[BeaconReading], method: str) -> BeaconFix:
        total_w = 0.0
        cx = 0.0
        cy = 0.0
        for r in readings:
            w = 1.0 / max(1.0, (r.distance_m or 1.0))
            cx += r.latitude * w
            cy += r.longitude * w
            total_w += w
        lat = cx / max(1e-9, total_w)
        lon = cy / max(1e-9, total_w)
        residuals = []
        for r in readings:
            actual = _haversine_m((r.latitude, r.longitude), (lat, lon))
            if r.distance_m is not None:
                residuals.append(abs(actual - r.distance_m))
        mean_res = sum(residuals) / len(residuals) if residuals else 0.0
        conf = max(0.1, min(0.7, 1.0 / (1.0 + mean_res / 50.0)))
        return BeaconFix(
            latitude=lat,
            longitude=lon,
            confidence=conf,
            residual_m=mean_res,
            sample_count=len(readings),
            method=method,
        )

    def _trilaterate(self, readings: List[BeaconReading]) -> Optional[BeaconFix]:
        origin = (readings[0].latitude, readings[0].longitude)
        pts: List[Tuple[float, float, float]] = []
        for r in readings:
            x, y = _latlon_to_local_xy((r.latitude, r.longitude), origin)
            pts.append((x, y, float(r.distance_m or 0.0)))

        x0, y0, r0 = pts[0]
        # Build linear system: 2(x_i - x0) * X + 2(y_i - y0) * Y = (x_i^2 + y_i^2 - x0^2 - y0^2) - (r_i^2 - r0^2)
        ATA = [[0.0, 0.0], [0.0, 0.0]]
        ATb = [0.0, 0.0]
        for (xi, yi, ri) in pts[1:]:
            a1 = 2 * (xi - x0)
            a2 = 2 * (yi - y0)
            b = (xi * xi + yi * yi - x0 * x0 - y0 * y0) - (ri * ri - r0 * r0)
            ATA[0][0] += a1 * a1
            ATA[0][1] += a1 * a2
            ATA[1][0] += a2 * a1
            ATA[1][1] += a2 * a2
            ATb[0] += a1 * b
            ATb[1] += a2 * b

        det = ATA[0][0] * ATA[1][1] - ATA[0][1] * ATA[1][0]
        if abs(det) < 1e-6:
            return None
        inv = [
            [ATA[1][1] / det, -ATA[0][1] / det],
            [-ATA[1][0] / det, ATA[0][0] / det],
        ]
        X = inv[0][0] * ATb[0] + inv[0][1] * ATb[1]
        Y = inv[1][0] * ATb[0] + inv[1][1] * ATb[1]
        lat, lon = _local_xy_to_latlon((X, Y), origin)

        residuals: List[float] = []
        for r in readings:
            actual = _haversine_m((r.latitude, r.longitude), (lat, lon))
            if r.distance_m is not None:
                residuals.append(abs(actual - r.distance_m))
        mean_res = sum(residuals) / len(residuals) if residuals else 0.0
        conf = max(0.2, min(0.95, 1.0 / (1.0 + mean_res / 30.0)))
        return BeaconFix(
            latitude=lat,
            longitude=lon,
            confidence=conf,
            residual_m=mean_res,
            sample_count=len(readings),
            method="trilateration",
        )
