"""Graceful SIGTERM / SIGINT handling for long-running job loops."""
from __future__ import annotations

import signal
import threading
from types import FrameType
from typing import Any, Optional


class ShutdownCoordinator:
    """Process-wide stop flag set by SIGTERM/SIGINT.

    The runner checks :meth:`should_stop` between jobs so in-flight work can
    finish or be left ``RUNNING`` for resume reclaim without starting new jobs.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._reason: Optional[str] = None
        self._installed = False
        self._prev_term: Any = None
        self._prev_int: Any = None

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def request_stop(self, reason: str = "shutdown") -> None:
        self._reason = str(reason)
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def install(self) -> None:
        if self._installed:
            return

        def _handler(signum: int, frame: Optional[FrameType]) -> None:
            try:
                name = signal.Signals(signum).name
            except (ValueError, AttributeError):
                name = str(signum)
            self.request_stop(f"signal:{name}")

        self._prev_term = signal.getsignal(signal.SIGTERM)
        self._prev_int = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._prev_term is not None:
            signal.signal(signal.SIGTERM, self._prev_term)
        if self._prev_int is not None:
            signal.signal(signal.SIGINT, self._prev_int)
        self._installed = False
