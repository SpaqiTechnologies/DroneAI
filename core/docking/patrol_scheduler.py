"""Scheduled patrols for a docked drone.

Each ``PatrolJob`` is a recurring mission: name, period (seconds), an
arbitrary ``payload`` (passed to ``DockingStation.deploy``), and an
auto-recall duration. The scheduler runs in a background thread and
asks the dock to deploy when each job is due — provided the dock is
``READY``.

This is the cron-equivalent for drone-in-a-box. Real deployments would
wire each fired job into an ``AutonomyRuntime`` or ``MissionManager``
via the dock's ``set_deploy_hook``.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.docking.docking_station import DockingStation


class PatrolJobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COOLDOWN = "cooldown"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PatrolJob:
    """A single recurring patrol entry."""
    job_id: str
    name: str
    period_s: float
    payload: Dict[str, Any] = field(default_factory=dict)
    flight_duration_s: float = 60.0     # how long the patrol flies before recall
    next_due_at: float = 0.0
    last_started_at: Optional[float] = None
    last_finished_at: Optional[float] = None
    last_status: PatrolJobStatus = PatrolJobStatus.PENDING
    runs: int = 0
    failures: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "period_s": self.period_s,
            "payload": dict(self.payload),
            "flight_duration_s": self.flight_duration_s,
            "next_due_at": self.next_due_at,
            "last_started_at": self.last_started_at,
            "last_finished_at": self.last_finished_at,
            "last_status": self.last_status.value,
            "runs": self.runs,
            "failures": self.failures,
            "enabled": self.enabled,
        }


class PatrolScheduler:
    """Runs ``PatrolJob`` entries against a ``DockingStation``."""

    def __init__(
        self,
        dock: "DockingStation",
        tick_hz: float = 4.0,
    ) -> None:
        if tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        self._dock = dock
        self._jobs: Dict[str, PatrolJob] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tick_interval = 1.0 / float(tick_hz)
        self._current_job_id: Optional[str] = None

    def add_job(
        self,
        name: str,
        period_s: float,
        payload: Optional[Dict[str, Any]] = None,
        flight_duration_s: float = 60.0,
        start_immediately: bool = False,
    ) -> PatrolJob:
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        job = PatrolJob(
            job_id=uuid.uuid4().hex[:12],
            name=name,
            period_s=float(period_s),
            payload=dict(payload or {}),
            flight_duration_s=float(flight_duration_s),
            next_due_at=time.time() if start_immediately else time.time() + period_s,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def remove_job(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def enable_job(self, job_id: str, enabled: bool) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.enabled = bool(enabled)
            return True

    def jobs(self) -> List[PatrolJob]:
        with self._lock:
            return list(self._jobs.values())

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="patrol-sched",
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        from core.docking.docking_station import DockState
        while not self._stop.is_set():
            now = time.time()
            with self._lock:
                current_id = self._current_job_id
                current = self._jobs.get(current_id) if current_id else None

            # If a patrol is in flight, recall when its flight duration expires.
            if current is not None:
                started = current.last_started_at or now
                if now - started >= current.flight_duration_s:
                    self._dock.recall(reason=f"patrol '{current.name}' duration up")
                    with self._lock:
                        current.last_finished_at = now
                        current.last_status = PatrolJobStatus.COOLDOWN
                        current.runs += 1
                        self._current_job_id = None

            else:
                # Find the next due, enabled job, only if the dock is ready.
                if self._dock.state == DockState.READY:
                    due = self._select_due_job(now)
                    if due is not None:
                        ok, msg = self._dock.deploy(
                            patrol_id=due.job_id,
                            patrol_name=due.name,
                            **due.payload,
                        )
                        with self._lock:
                            if ok:
                                due.last_started_at = now
                                due.last_status = PatrolJobStatus.RUNNING
                                due.next_due_at = now + due.period_s
                                self._current_job_id = due.job_id
                            else:
                                due.failures += 1
                                due.last_status = PatrolJobStatus.FAILED
                                due.next_due_at = now + max(5.0, due.period_s * 0.1)

            time.sleep(self._tick_interval)

    def _select_due_job(self, now: float) -> Optional[PatrolJob]:
        with self._lock:
            due = [
                j for j in self._jobs.values()
                if j.enabled and j.next_due_at <= now
            ]
            if not due:
                return None
            # Highest priority = earliest next_due_at.
            due.sort(key=lambda j: j.next_due_at)
            return due[0]
