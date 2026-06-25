"""Tests for the benchmark memory indexing/query engine."""

import json
from pathlib import Path

import pytest

from benchmark_memory import BenchmarkMemory


@pytest.fixture
def sample_results(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "race_1.json").write_text(json.dumps({
        "metric": "tok/s",
        "model": "gpt-oss-120b",
        "timestamp": "2026-06-20T12:00:00",
        "sweep": [
            {"config": {"name": "c8_t100", "concurrency": 8, "max_completion_tokens": 100},
             "tok_per_sec": 1000, "req_per_sec": 10, "avg_latency": 0.5, "num_requests": 5},
            {"config": {"name": "c16_t200", "concurrency": 16, "max_completion_tokens": 200},
             "tok_per_sec": 2000, "req_per_sec": 8, "avg_latency": 0.6, "num_requests": 5},
        ],
        "race": {"config": {"name": "c16_t200", "concurrency": 16, "max_completion_tokens": 200},
                 "tok_per_sec": 1800, "req_per_sec": 7, "avg_latency": 0.65, "num_requests": 50},
    }))
    (results_dir / "race_2.json").write_text(json.dumps({
        "metric": "tok/s",
        "model": "gpt-oss-120b",
        "timestamp": "2026-06-21T12:00:00",
        "sweep": [
            {"config": {"name": "c8_t100", "concurrency": 8, "max_completion_tokens": 100},
             "tok_per_sec": 1100, "req_per_sec": 11, "avg_latency": 0.45, "num_requests": 5},
        ],
        "best": {"config": {"name": "c8_t100", "concurrency": 8, "max_completion_tokens": 100},
                 "tok_per_sec": 1200, "req_per_sec": 12, "avg_latency": 0.4, "num_requests": 50},
    }))
    return results_dir


def test_loads_records(sample_results: Path):
    mem = BenchmarkMemory(sample_results)
    assert mem.count() == 5  # 2 sweep + 1 race + 1 sweep + 1 best


def test_top_by_tok_per_sec(sample_results: Path):
    mem = BenchmarkMemory(sample_results)
    top = mem.top("tok/s", k=2)
    assert top[0]["tok_per_sec"] == 2000
    assert top[1]["tok_per_sec"] == 1800


def test_insights(sample_results: Path):
    mem = BenchmarkMemory(sample_results)
    ins = mem.insights()
    assert ins["status"] == "ok"
    assert ins["total_records"] == 5
    assert ins["top_tok_per_sec"]["tok_per_sec"] == 2000


def test_search_by_config_name(sample_results: Path):
    mem = BenchmarkMemory(sample_results)
    res = mem.search("c8_t100")
    assert res["status"] == "ok"
    assert len(res["results"]) == 3


def test_compare(sample_results: Path):
    mem = BenchmarkMemory(sample_results)
    comp = mem.compare("c8_t100", "c16_t200")
    assert comp["status"] == "ok"
    assert comp["a"]["runs"] == 3
    assert comp["b"]["runs"] == 2
    assert comp["a"]["avg_tok_per_sec"] == pytest.approx(1100.0)
    assert comp["b"]["avg_tok_per_sec"] == pytest.approx(1900.0)
