from __future__ import annotations

import math
import threading
import time
from collections import deque


class ApiRateLimiter:
  """In-process sliding-window limit keyed by client IP."""

  def __init__(
      self,
      per_minute: int,
      per_day: int,
      global_per_minute: int | None = None,
      global_per_day: int | None = None,
      ) -> None:
    self._per_minute = per_minute
    self._per_day = per_day
    self._global_per_minute = global_per_minute
    self._global_per_day = global_per_day
    self._requests: dict[str, deque[float]] = {} 
    self._global_requests: deque[float] = deque()
    self._lock = threading.Lock()
    self._checks = 0

  def retry_after(self, client: str) -> int | None:
    now = time.monotonic()
    with self._lock:
      requests = self._requests.setdefault(client, deque())
      while requests and requests[0] <= now - 86_400:
        requests.popleft()
      while self._global_requests and self._global_requests[0] <= now - 86_400:
        self._global_requests.popleft()

      scopes = [(requests, self._per_minute, self._per_day)]
      if self._global_per_minute is not None and self._global_per_day is not None:
        if self._global_per_minute is not None and self._global_per_day is not None:
          scopes.append((self._global_requests, self._global_per_minute, self._global_per_day))

        for events, per_minute, per_day in scopes:
          if len(events) >= per_day:
            return max(1, math.ceil(events[0] + 86_400 - now))

          recent = [timestamp for timestamp in events if timestamp > now - 60]
          if len(recent) >= per_minute:
            return max(1, math.ceil(recent[0] + 60 - now))

        requests.append(now)
        if len(scopes) == 2:
          self._global_requests.append(now)
        self._checks += 1
        if self._checks % 256 == 0:
          self._requests = {key: values for key, values in self._requests.items() if values}
        return None