"""Headroom proxy.log parser and live monitor.

Adapts the logic from https://github.com/RonnieTheTester/headroom-meter
for server-side use. Tails ~/.headroom/logs/proxy.log, parses PERF /
frame-compression / TOIN lines, and exposes aggregated metrics via
subscriptions.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Iterable

DEFAULT_LOG = Path.home() / ".headroom" / "logs" / "proxy.log"

PERF_RE = re.compile(
    r"tok_before=(?P<before>\d+).*?"
    r"tok_after=(?P<after>\d+).*?"
    r"tok_saved=(?P<saved>\d+).*?"
    r"cache_read=(?P<cache_read>\d+).*?"
    r"cache_write=(?P<cache_write>\d+).*?"
    r"cache_hit_pct=(?P<cache_hit>\d+).*?"
    r"opt_ms=(?P<opt_ms>\d+)"
)
FRAME_RE = re.compile(
    r"frame compressed (?P<before>\d+)\D+(?P<after>\d+) bytes "
    r"\((?P<saved>\d+) tokens saved"
)
TOIN_RE = re.compile(
    r"TOIN: (?P<patterns>\d+) patterns, (?P<compressions>\d+) compressions, "
    r"(?P<retrievals>\d+) retrievals, (?P<rate>[0-9.]+)% retrieval rate"
)
TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return 100.0 * part / whole


@dataclass(frozen=True)
class PerfSample:
    before: int
    after: int
    saved: int
    cache_read: int
    cache_write: int
    cache_hit: int
    opt_ms: int
    line: str
    at: datetime

    @property
    def pct(self) -> float:
        return _percent(self.saved, self.before)


@dataclass(frozen=True)
class FrameSample:
    before: int
    after: int
    saved: int
    at: datetime

    @property
    def byte_pct(self) -> float:
        return _percent(self.before - self.after, self.before)


@dataclass(frozen=True)
class ToinSample:
    patterns: int
    compressions: int
    retrievals: int
    rate: float


@dataclass
class Meter:
    perf_requests: int = 0
    tok_before: int = 0
    tok_after: int = 0
    tok_saved: int = 0
    cache_weighted: int = 0
    cache_weight: int = 0
    opt_ms_total: int = 0
    frame_count: int = 0
    frame_before: int = 0
    frame_after: int = 0
    frame_tokens_saved: int = 0
    last_perf: PerfSample | None = None
    last_frame: FrameSample | None = None
    last_toin: ToinSample | None = None
    last_event_at: datetime | None = None
    recent_pct: deque[float] = field(default_factory=lambda: deque(maxlen=72))
    recent_saved: deque[int] = field(default_factory=lambda: deque(maxlen=72))
    recent_frame_pct: deque[float] = field(default_factory=lambda: deque(maxlen=72))
    transforms: list[str] = field(default_factory=list)

    def add_line(self, line: str) -> bool:
        changed = False
        timestamp = _parse_timestamp(line) or datetime.now()

        if " PERF " in line and "total_ms=" in line:
            match = PERF_RE.search(line)
            if match:
                sample = PerfSample(
                    before=int(match.group("before")),
                    after=int(match.group("after")),
                    saved=int(match.group("saved")),
                    cache_read=int(match.group("cache_read")),
                    cache_write=int(match.group("cache_write")),
                    cache_hit=int(match.group("cache_hit")),
                    opt_ms=int(match.group("opt_ms")),
                    line=line.rstrip(),
                    at=timestamp,
                )
                self.perf_requests += 1
                self.tok_before += sample.before
                self.tok_after += sample.after
                self.tok_saved += sample.saved
                self.cache_weighted += sample.cache_hit * max(sample.before, 1)
                self.cache_weight += max(sample.before, 1)
                self.opt_ms_total += sample.opt_ms
                self.last_perf = sample
                self.last_event_at = timestamp
                self.recent_pct.append(sample.pct)
                self.recent_saved.append(sample.saved)
                self.transforms = _extract_transforms(line)
                changed = True

        if "frame compressed" in line:
            match = FRAME_RE.search(line)
            if match:
                sample = FrameSample(
                    before=int(match.group("before")),
                    after=int(match.group("after")),
                    saved=int(match.group("saved")),
                    at=timestamp,
                )
                self.frame_count += 1
                self.frame_before += sample.before
                self.frame_after += sample.after
                self.frame_tokens_saved += sample.saved
                self.last_frame = sample
                self.last_event_at = timestamp
                self.recent_frame_pct.append(sample.byte_pct)
                changed = True

        if "TOIN:" in line:
            match = TOIN_RE.search(line)
            if match:
                self.last_toin = ToinSample(
                    patterns=int(match.group("patterns")),
                    compressions=int(match.group("compressions")),
                    retrievals=int(match.group("retrievals")),
                    rate=float(match.group("rate")),
                )
                self.last_event_at = timestamp
                changed = True

        return changed

    @property
    def token_savings_pct(self) -> float:
        return _percent(self.tok_saved, self.tok_before)

    @property
    def byte_savings_pct(self) -> float:
        return _percent(self.frame_before - self.frame_after, self.frame_before)

    @property
    def avg_cache_hit(self) -> float:
        if self.cache_weight <= 0:
            return 0.0
        return self.cache_weighted / self.cache_weight

    @property
    def avg_opt_ms(self) -> float:
        if self.perf_requests <= 0:
            return 0.0
        return self.opt_ms_total / self.perf_requests

    @property
    def avg_saved(self) -> float:
        if self.perf_requests <= 0:
            return 0.0
        return self.tok_saved / self.perf_requests

    def snapshot(self) -> dict:
        last_pct = self.last_perf.pct if self.last_perf else 0.0
        last_saved = self.last_perf.saved if self.last_perf else 0
        return {
            "status": "watching",
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
            "speedometer": {
                "now_pct": round(last_pct, 2),
                "scale_max": 25.0,
            },
            "odometer": {
                "tok_saved": self.tok_saved,
            },
            "token_regen": {
                "pct": round(self.token_savings_pct, 2),
                "scale_max": 25.0,
            },
            "recent_pulses": list(self.recent_pct),
            "frame_compression": {
                "byte_pct": round(self.byte_savings_pct, 2),
                "last_frame_pct": round(self.last_frame.byte_pct, 2) if self.last_frame else 0.0,
                "recent": list(self.recent_frame_pct),
            },
            "saved_tokens": {
                "total": self.tok_saved,
                "last_request": last_saved,
            },
            "requests": {
                "completed": self.perf_requests,
                "avg_saved": round(self.avg_saved, 1),
            },
            "cache_battery": {
                "pct": round(self.avg_cache_hit, 1),
            },
            "optimize_time": {
                "avg_ms": round(self.avg_opt_ms, 1),
            },
            "transforms": self.transforms,
            "toin": {
                "patterns": self.last_toin.patterns if self.last_toin else 0,
                "compressions": self.last_toin.compressions if self.last_toin else 0,
                "retrievals": self.last_toin.retrievals if self.last_toin else 0,
                "rate": self.last_toin.rate if self.last_toin else 0.0,
            },
        }


def _parse_timestamp(line: str) -> datetime | None:
    match = TS_RE.search(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None


def _extract_transforms(line: str) -> list[str]:
    """Best-effort extraction of transform names from a PERF line."""
    transforms: list[str] = []
    # Common Headroom transform names observed in logs.
    for name in ("text", "smart_crusher", "kompress", "log", "mixed", "tool_schema_compaction"):
        if name in line and name not in transforms:
            transforms.append(name)
    return transforms


def tail_log(path: Path, stop_event: threading.Event, poll_interval: float = 0.25) -> Iterable[str]:
    """Yield new lines from a log file, handling rotation/recreation."""
    current_path = Path(path)
    while not stop_event.is_set():
        try:
            if not current_path.exists():
                time.sleep(poll_interval)
                continue

            with current_path.open("r", encoding="utf-8", errors="replace") as f:
                # Start at end of file to only emit new lines.
                f.seek(0, 2)
                inode = current_path.stat().st_ino
                while not stop_event.is_set():
                    line = f.readline()
                    if line:
                        yield line
                        continue

                    # No new data; check for rotation.
                    try:
                        new_inode = current_path.stat().st_ino
                    except FileNotFoundError:
                        new_inode = None

                    if new_inode != inode:
                        # File rotated; reopen.
                        break

                    time.sleep(poll_interval)
        except Exception:
            # Be resilient: wait and retry.
            time.sleep(poll_interval)


class HeadroomMonitor:
    """Thread-safe monitor that tails a Headroom proxy.log and broadcasts
    snapshots to subscribers.
    """

    def __init__(self, log_path: Path | str = DEFAULT_LOG) -> None:
        self.log_path = Path(log_path)
        self.meter = Meter()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._subscribers: list[Queue[dict]] = []

    @property
    def status(self) -> str:
        if self._thread is None or not self._thread.is_alive():
            return "idle"
        if not self.log_path.exists():
            return "waiting_for_log"
        return "watching"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest_snapshot(self) -> dict:
        with self._lock:
            snap = self.meter.snapshot()
        snap["status"] = self.status
        snap["log_path"] = str(self.log_path)
        return snap

    def subscribe(self) -> Queue[dict]:
        q: Queue[dict] = Queue(maxsize=64)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue[dict]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, snap: dict) -> None:
        with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(snap)
                except Exception:
                    # Drop slow subscribers lazily.
                    try:
                        self._subscribers.remove(q)
                    except ValueError:
                        pass

    def _run(self) -> None:
        for line in tail_log(self.log_path, self._stop_event):
            if self._stop_event.is_set():
                break
            with self._lock:
                changed = self.meter.add_line(line)
            if changed:
                self._broadcast(self.latest_snapshot())
