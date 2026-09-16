"""Base agent class.

Every agent in the system implements this ABC, receiving structured
input and returning structured output. The base provides shared
timing / trace utilities so that agents do not repeat that plumbing.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from src.models.schemas import AgentTraceEntry


class BaseAgent(ABC):
    name: str = "BaseAgent"

    def __init__(self) -> None:
        self._last_trace: AgentTraceEntry | None = None

    @abstractmethod
    def run(self, *args: Any) -> Any:
        ...

    def run_with_trace(self, *args: Any, input_summary: str = "") -> tuple[Any, AgentTraceEntry]:
        "Run self.run(*args) and record an AgentTraceEntry."
        t0 = time.time()
        output = self.run(*args)
        elapsed = (time.time() - t0) * 1000
        trace = AgentTraceEntry(
            agent=self.name,
            action="run",
            input_summary=input_summary,
            output_summary=str(output)[:300],
            elapse_ms=round(elapsed, 1),
        )
        self._last_trace = trace
        return output, trace