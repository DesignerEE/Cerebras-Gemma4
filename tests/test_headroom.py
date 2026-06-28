"""Tests for the Headroom proxy.log parser and monitor."""

from datetime import datetime
from pathlib import Path
from threading import Event

import pytest

from headroom import Meter, HeadroomMonitor, tail_log


SAMPLE_PERF_LINE = (
    "2026-06-25 07:13:45,123 PERF req_id=abc model=gemma-4-31b "
    "tok_before=60229 tok_after=57678 tok_saved=2551 "
    "cache_read=120 cache_write=5 cache_hit_pct=98 opt_ms=45 total_ms=120 "
    "transforms=text,smart_crusher"
)

SAMPLE_FRAME_LINE = (
    "2026-06-25 07:13:46,456 frame compressed 15432 bytes to 14512 bytes "
    "(120 tokens saved)"
)

SAMPLE_TOIN_LINE = (
    "2026-06-25 07:13:47,789 TOIN: 181 patterns, 283 compressions, "
    "0 retrievals, 0.0% retrieval rate"
)


def test_meter_parses_perf_line():
    meter = Meter()
    assert meter.add_line(SAMPLE_PERF_LINE) is True
    assert meter.perf_requests == 1
    assert meter.tok_saved == 2551
    assert meter.last_perf is not None
    assert meter.last_perf.pct == pytest.approx(2551 / 60229 * 100)
    assert meter.avg_cache_hit == 98.0
    assert meter.avg_opt_ms == 45.0


def test_meter_ignores_duplicate_perfs_without_total_ms():
    meter = Meter()
    line_without_total = SAMPLE_PERF_LINE.replace(" total_ms=120", "")
    assert meter.add_line(line_without_total) is False
    assert meter.perf_requests == 0


def test_meter_parses_frame_line():
    meter = Meter()
    assert meter.add_line(SAMPLE_FRAME_LINE) is True
    assert meter.frame_count == 1
    assert meter.byte_savings_pct == pytest.approx((15432 - 14512) / 15432 * 100)


def test_meter_parses_toin_line():
    meter = Meter()
    assert meter.add_line(SAMPLE_TOIN_LINE) is True
    assert meter.last_toin is not None
    assert meter.last_toin.patterns == 181
    assert meter.last_toin.rate == 0.0


def test_meter_snapshot_structure():
    meter = Meter()
    meter.add_line(SAMPLE_PERF_LINE)
    snap = meter.snapshot()
    assert snap["status"] == "watching"
    assert snap["odometer"]["tok_saved"] == 2551
    assert snap["requests"]["completed"] == 1
    assert snap["cache_battery"]["pct"] == 98.0
    assert "speedometer" in snap
    assert "recent_pulses" in snap


def test_tail_log_picks_up_new_lines(tmp_path: Path):
    log = tmp_path / "proxy.log"
    log.write_text("")
    stop = Event()

    def write_line():
        with log.open("a") as f:
            f.write(SAMPLE_PERF_LINE + "\n")

    # Start consumer in a thread.
    lines = []

    def consume():
        for line in tail_log(log, stop):
            lines.append(line)
            if len(lines) >= 1:
                break

    import threading
    t = threading.Thread(target=consume, daemon=True)
    t.start()

    import time
    time.sleep(0.2)
    write_line()
    t.join(timeout=2.0)
    stop.set()

    assert len(lines) == 1
    assert "PERF" in lines[0]


def test_monitor_subscribe_and_snapshot(tmp_path: Path):
    log = tmp_path / "proxy.log"
    log.write_text(SAMPLE_PERF_LINE + "\n")
    monitor = HeadroomMonitor(log)
    q = monitor.subscribe()
    monitor.start()

    import time
    time.sleep(0.5)

    # Monitor starts at end, so existing line is ignored. Append a new line.
    with log.open("a") as f:
        f.write(SAMPLE_PERF_LINE.replace("req_id=abc", "req_id=def") + "\n")

    try:
        snap = q.get(timeout=2.0)
    finally:
        monitor.stop()

    assert snap["requests"]["completed"] >= 1
    assert snap["odometer"]["tok_saved"] >= 2551
