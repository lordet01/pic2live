from __future__ import annotations

import time


class FpsTimer:
    def __init__(self, avg_window: int = 60) -> None:
        self.avg_window = max(1, avg_window)
        self.stamps = []

    def tick(self) -> float:
        now = time.perf_counter()
        self.stamps.append(now)
        if len(self.stamps) > self.avg_window:
            self.stamps.pop(0)
        if len(self.stamps) >= 2:
            dt = self.stamps[-1] - self.stamps[0]
            if dt > 0:
                return (len(self.stamps) - 1) / dt
        return 0.0


