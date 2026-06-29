"""Tests for the RSSI beacon locator."""

import math

import pytest

from applications.survival.beacon_locator import (
    BeaconLocator,
    rssi_to_distance,
)


def _meters_between(a, b) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(h))


def test_rssi_to_distance_monotonic() -> None:
    d1 = rssi_to_distance(-50.0)
    d2 = rssi_to_distance(-70.0)
    d3 = rssi_to_distance(-90.0)
    assert d1 < d2 < d3


def test_rssi_to_distance_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        rssi_to_distance(-70.0, path_loss_exponent=0.0)
    with pytest.raises(ValueError):
        rssi_to_distance(-70.0, reference_distance_m=0.0)


def test_beacon_locator_no_samples_returns_none() -> None:
    bl = BeaconLocator()
    assert bl.compute_fix() is None


def test_beacon_locator_single_sample_low_confidence() -> None:
    bl = BeaconLocator()
    bl.add_reading(47.5, -122.3, 20.0, -65.0)
    fix = bl.compute_fix()
    assert fix is not None
    assert fix.method == "single"
    assert fix.sample_count == 1
    assert fix.confidence < 0.5


def test_beacon_locator_trilaterates_correctly() -> None:
    # Place beacon at true_pos; synthesize RSSI from true distances.
    true_pos = (47.5005, -122.3005)
    bl = BeaconLocator(tx_power_dbm=-40.0, path_loss_exponent=2.0)
    sample_points = [
        (47.5000, -122.3000, 20.0),
        (47.5012, -122.3002, 20.0),
        (47.5002, -122.3015, 20.0),
        (47.5010, -122.3012, 20.0),
    ]
    for (lat, lon, alt) in sample_points:
        d_true = _meters_between((lat, lon), true_pos)
        d_true = max(1.0, d_true)
        # invert log model
        rssi = -40.0 - 10.0 * 2.0 * math.log10(d_true)
        bl.add_reading(lat, lon, alt, rssi)
    fix = bl.compute_fix()
    assert fix is not None
    assert fix.method == "trilateration"
    err_m = _meters_between((fix.latitude, fix.longitude), true_pos)
    assert err_m < 20.0
    assert fix.confidence > 0.4


def test_beacon_locator_respects_min_sample_distance() -> None:
    bl = BeaconLocator(min_sample_distance_m=50.0)
    bl.add_reading(47.5000, -122.3000, 20.0, -55.0)
    bl.add_reading(47.5000, -122.3000, 20.0, -56.0)  # same spot
    assert bl.sample_count == 1


def test_beacon_locator_two_samples_uses_centroid() -> None:
    bl = BeaconLocator()
    bl.add_reading(47.5000, -122.3000, 20.0, -55.0)
    bl.add_reading(47.5010, -122.3010, 20.0, -75.0)
    fix = bl.compute_fix()
    assert fix is not None
    assert fix.method == "centroid"
    assert fix.sample_count == 2
