"""
Base application interface for drone operations.

Provides common functionality for all specialized
drone applications.
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class ApplicationState(Enum):
    """State of an application."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ApplicationProgress:
    """Progress information for an application."""
    percent: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    status_message: str = ""
    eta_seconds: Optional[float] = None
    data_collected: int = 0
    errors: int = 0


class BaseApplication(ABC):
    """
    Abstract base class for drone applications.

    Provides common infrastructure for:
    - Lifecycle management (init, start, pause, stop)
    - Progress tracking
    - Event callbacks
    - Data collection
    """

    def __init__(self, name: str):
        """
        Initialize application.

        Args:
            name: Application name
        """
        self.name = name
        self._state = ApplicationState.IDLE
        self._progress = ApplicationProgress()
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._collected_data: List[Any] = []
        self._errors: List[str] = []

        # Callbacks
        self._on_state_changed: List[Callable] = []
        self._on_progress_updated: List[Callable] = []
        self._on_data_collected: List[Callable] = []
        self._on_error: List[Callable] = []
        self._on_completed: List[Callable] = []

    @property
    def state(self) -> ApplicationState:
        """Get current application state."""
        return self._state

    @property
    def progress(self) -> ApplicationProgress:
        """Get current progress."""
        return self._progress

    @property
    def is_running(self) -> bool:
        """Check if application is running."""
        return self._state in [
            ApplicationState.RUNNING,
            ApplicationState.COMPLETING,
        ]

    @property
    def is_finished(self) -> bool:
        """Check if application has finished."""
        return self._state in [
            ApplicationState.COMPLETED,
            ApplicationState.FAILED,
            ApplicationState.ABORTED,
        ]

    @property
    def duration(self) -> Optional[float]:
        """Get application duration in seconds."""
        if self._start_time is None:
            return None
        end = self._end_time or time.time()
        return end - self._start_time

    def initialize(self) -> bool:
        """
        Initialize the application.

        Returns:
            True if initialization successful
        """
        if self._state != ApplicationState.IDLE:
            logger.warning(f"Cannot init from state {self._state}")
            return False

        self._set_state(ApplicationState.INITIALIZING)

        try:
            success = self._do_initialize()
            if success:
                self._set_state(ApplicationState.READY)
            else:
                self._set_state(ApplicationState.FAILED)
            return success

        except Exception as e:
            self._add_error(f"Init failed: {e}")
            self._set_state(ApplicationState.FAILED)
            return False

    def start(self) -> bool:
        """
        Start the application.

        Returns:
            True if started successfully
        """
        if self._state != ApplicationState.READY:
            logger.warning(f"Cannot start from state {self._state}")
            return False

        self._start_time = time.time()
        self._set_state(ApplicationState.RUNNING)

        try:
            return self._do_start()
        except Exception as e:
            self._add_error(f"Start failed: {e}")
            self._set_state(ApplicationState.FAILED)
            return False

    def pause(self) -> bool:
        """Pause the application."""
        if self._state != ApplicationState.RUNNING:
            return False

        self._set_state(ApplicationState.PAUSED)
        return self._do_pause()

    def resume(self) -> bool:
        """Resume paused application."""
        if self._state != ApplicationState.PAUSED:
            return False

        self._set_state(ApplicationState.RUNNING)
        return self._do_resume()

    def abort(self) -> None:
        """Abort the application."""
        if self.is_finished:
            return

        self._end_time = time.time()
        self._set_state(ApplicationState.ABORTED)
        self._do_abort()

    def update(self, dt: float) -> None:
        """
        Update application (called each frame).

        Args:
            dt: Delta time in seconds
        """
        if self._state != ApplicationState.RUNNING:
            return

        try:
            self._do_update(dt)

            # Check completion
            if self._check_completion():
                self._complete()

        except Exception as e:
            self._add_error(f"Update error: {e}")

    def _complete(self) -> None:
        """Mark application as completed."""
        self._set_state(ApplicationState.COMPLETING)
        self._do_complete()
        self._end_time = time.time()
        self._set_state(ApplicationState.COMPLETED)
        self._notify_completed()

    # Abstract methods for subclasses

    @abstractmethod
    def _do_initialize(self) -> bool:
        """Perform initialization. Override in subclass."""
        pass

    @abstractmethod
    def _do_start(self) -> bool:
        """Perform start. Override in subclass."""
        pass

    @abstractmethod
    def _do_update(self, dt: float) -> None:
        """Perform update. Override in subclass."""
        pass

    @abstractmethod
    def _check_completion(self) -> bool:
        """Check if application is complete."""
        pass

    def _do_pause(self) -> bool:
        """Override for pause logic."""
        return True

    def _do_resume(self) -> bool:
        """Override for resume logic."""
        return True

    def _do_abort(self) -> None:
        """Override for abort cleanup."""
        pass

    def _do_complete(self) -> None:
        """Override for completion processing."""
        pass

    # Progress helpers

    def _update_progress(
        self,
        percent: Optional[float] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update progress information."""
        if percent is not None:
            self._progress.percent = percent
        if current_step is not None:
            self._progress.current_step = current_step
        if total_steps is not None:
            self._progress.total_steps = total_steps
        if message is not None:
            self._progress.status_message = message

        # Calculate ETA
        if self._start_time and self._progress.percent > 0:
            elapsed = time.time() - self._start_time
            total_est = elapsed / (self._progress.percent / 100)
            self._progress.eta_seconds = total_est - elapsed

        self._notify_progress()

    def _add_data(self, data: Any) -> None:
        """Add collected data."""
        self._collected_data.append(data)
        self._progress.data_collected = len(self._collected_data)
        self._notify_data_collected(data)

    def _add_error(self, error: str) -> None:
        """Record an error."""
        self._errors.append(error)
        self._progress.errors = len(self._errors)
        logger.error(f"[{self.name}] {error}")
        self._notify_error(error)

    # State management

    def _set_state(self, new_state: ApplicationState) -> None:
        """Set application state."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            logger.info(f"[{self.name}] {old_state} -> {new_state}")
            self._notify_state_changed(old_state, new_state)

    # Event registration

    def on_state_changed(self, callback: Callable) -> None:
        """Register state change callback."""
        self._on_state_changed.append(callback)

    def on_progress_updated(self, callback: Callable) -> None:
        """Register progress callback."""
        self._on_progress_updated.append(callback)

    def on_data_collected(self, callback: Callable) -> None:
        """Register data collection callback."""
        self._on_data_collected.append(callback)

    def on_error(self, callback: Callable) -> None:
        """Register error callback."""
        self._on_error.append(callback)

    def on_completed(self, callback: Callable) -> None:
        """Register completion callback."""
        self._on_completed.append(callback)

    # Notifications

    def _notify_state_changed(
        self,
        old: ApplicationState,
        new: ApplicationState,
    ) -> None:
        """Notify state change callbacks."""
        for cb in self._on_state_changed:
            try:
                cb(old, new)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_progress(self) -> None:
        """Notify progress callbacks."""
        for cb in self._on_progress_updated:
            try:
                cb(self._progress)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_data_collected(self, data: Any) -> None:
        """Notify data collection callbacks."""
        for cb in self._on_data_collected:
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_error(self, error: str) -> None:
        """Notify error callbacks."""
        for cb in self._on_error:
            try:
                cb(error)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _notify_completed(self) -> None:
        """Notify completion callbacks."""
        for cb in self._on_completed:
            try:
                cb()
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_results(self) -> Dict:
        """Get application results."""
        return {
            'name': self.name,
            'state': self._state.value,
            'duration': self.duration,
            'progress': self._progress.percent,
            'data_count': len(self._collected_data),
            'error_count': len(self._errors),
            'errors': self._errors,
        }
