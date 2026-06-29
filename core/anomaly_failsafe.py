"""Bridge: route ``AnomalyDetector`` events into ``FailsafeManager``.

The anomaly detector knows *what is wrong* (motor vibration spike, GPS
HDOP blowing out, battery temperature climbing). The failsafe manager
knows *what the drone should do about it* (RTH, immediate land, hover).
This module connects the two so a critical/emergency anomaly auto-
triggers the matching failsafe — instead of just being logged.

A simple, explicit mapping table keeps the routing inspectable. The
``trigger_history`` lets callers audit which anomalies escalated.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ai.anomaly.anomaly_detector import (
    AnomalyDetector,
    AnomalyType,
    AnomalySeverity,
)
from core.failsafe import FailsafeAction, FailsafeManager, FailsafeType

if TYPE_CHECKING:
    from ai.anomaly.anomaly_detector import Anomaly


_ANOMALY_TO_FAILSAFE: Dict[AnomalyType, FailsafeType] = {
    AnomalyType.MOTOR_VIBRATION:     FailsafeType.MOTOR_FAILURE,
    AnomalyType.MOTOR_TEMPERATURE:   FailsafeType.MOTOR_FAILURE,
    AnomalyType.MOTOR_CURRENT:       FailsafeType.MOTOR_FAILURE,
    AnomalyType.BATTERY_DEGRADATION: FailsafeType.LOW_BATTERY,
    AnomalyType.BATTERY_TEMPERATURE: FailsafeType.CRITICAL_BATTERY,
    AnomalyType.IMU_DRIFT:           FailsafeType.IMU_FAILURE,
    AnomalyType.GPS_ANOMALY:         FailsafeType.GPS_LOSS,
}


@dataclass
class AnomalyTrigger:
    anomaly_type: str
    severity: str
    failsafe_type: str
    failsafe_action: str
    component: str
    triggered_at: float
    reason: str
    data: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "failsafe_type": self.failsafe_type,
            "failsafe_action": self.failsafe_action,
            "component": self.component,
            "triggered_at": self.triggered_at,
            "reason": self.reason,
            "data": dict(self.data),
        }


class AnomalyFailsafeBridge:
    """Forwards critical/emergency anomalies to the failsafe manager.

    Parameters
    ----------
    detector:
        Anomaly detector to subscribe to.
    failsafe:
        Failsafe manager whose ``_trigger_failsafe`` is invoked on a hit.
    min_severity:
        Anomalies below this severity are ignored. Default ``CRITICAL``.
    custom_mapping:
        Override (or extend) the default anomaly_type → failsafe_type table.
    on_trigger:
        Optional callback fired after each forwarded trigger; receives an
        ``AnomalyTrigger`` snapshot.
    """

    def __init__(
        self,
        detector: AnomalyDetector,
        failsafe: FailsafeManager,
        min_severity: AnomalySeverity = AnomalySeverity.CRITICAL,
        custom_mapping: Optional[Dict[AnomalyType, FailsafeType]] = None,
        on_trigger: Optional[Callable[[AnomalyTrigger], None]] = None,
    ) -> None:
        self._detector = detector
        self._failsafe = failsafe
        self._min_severity = min_severity
        self._mapping: Dict[AnomalyType, FailsafeType] = dict(_ANOMALY_TO_FAILSAFE)
        if custom_mapping:
            self._mapping.update(custom_mapping)
        self._on_trigger = on_trigger
        self._history: List[AnomalyTrigger] = []
        self._lock = threading.Lock()
        self._attached = False

    def attach(self) -> None:
        if self._attached:
            return
        self._detector.add_callback(self._on_anomaly)
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        self._detector.remove_callback(self._on_anomaly)
        self._attached = False

    @property
    def history(self) -> List[AnomalyTrigger]:
        with self._lock:
            return list(self._history)

    def map(self, anomaly_type: AnomalyType) -> Optional[FailsafeType]:
        return self._mapping.get(anomaly_type)

    @staticmethod
    def _severity_rank(s: AnomalySeverity) -> int:
        order = {
            AnomalySeverity.INFO:      0,
            AnomalySeverity.WARNING:   1,
            AnomalySeverity.CRITICAL:  2,
            AnomalySeverity.EMERGENCY: 3,
        }
        return order.get(s, -1)

    def _on_anomaly(self, anomaly: "Anomaly") -> None:
        if self._severity_rank(anomaly.severity) < self._severity_rank(self._min_severity):
            return
        failsafe_type = self._mapping.get(anomaly.anomaly_type)
        if failsafe_type is None:
            return

        reason = (
            f"{anomaly.anomaly_type.value} ({anomaly.severity.value}) "
            f"on {anomaly.component or 'unknown'}: "
            f"value={anomaly.value:.3g} dev={anomaly.deviation:.2f}σ"
        )
        data: Dict[str, object] = {
            "anomaly_id": anomaly.anomaly_id,
            "severity": anomaly.severity.value,
            "deviation": anomaly.deviation,
            "value": anomaly.value,
            "confidence": anomaly.confidence,
        }
        try:
            action = self._failsafe._trigger_failsafe(failsafe_type, reason, data)
        except Exception:
            return
        action_value = action.value if isinstance(action, FailsafeAction) else str(action)
        trigger = AnomalyTrigger(
            anomaly_type=anomaly.anomaly_type.value,
            severity=anomaly.severity.value,
            failsafe_type=failsafe_type.name,
            failsafe_action=action_value,
            component=anomaly.component or "",
            triggered_at=time.time(),
            reason=reason,
            data=data,
        )
        with self._lock:
            self._history.append(trigger)
        if self._on_trigger is not None:
            try:
                self._on_trigger(trigger)
            except Exception:
                pass
