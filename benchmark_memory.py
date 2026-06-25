"""Benchmark Memory — lightweight local indexing and query engine for
results/ JSON files.

Implements context-mode-style algorithms (aggregate, rank, filter, compare)
over past Cerebras benchmark runs so the dashboard can answer questions like
"what was the best tok/s config?" without calling external services.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"


def _norm_config(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize config shape across legacy and advanced result files.

    Legacy files nest config under `config`; advanced files put the fields
    directly on the sweep/best/race object.
    """
    cfg = obj.get("config") or obj
    concurrency = cfg.get("concurrency")
    max_tokens = cfg.get("max_completion_tokens") or cfg.get("max_tokens")
    name = cfg.get("name") or (
        f"c{concurrency}_t{max_tokens}" if concurrency is not None and max_tokens is not None else "unknown"
    )
    return {"name": name, "concurrency": concurrency, "max_tokens": max_tokens}


def _safe_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


class BenchmarkMemory:
    """In-memory index of all benchmark result files under results/."""

    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        self.results_dir = results_dir
        self.records: list[dict[str, Any]] = []
        self._load()

    @classmethod
    def from_file(cls, path: Path, data: dict[str, Any] | None = None) -> "BenchmarkMemory":
        """Create a memory index from a single file (for test uploads)."""
        mem = object.__new__(cls)
        mem.results_dir = path.parent
        mem.records = []
        if data is None:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        mem._load_data(data, path.name)
        return mem

    def _load(self) -> None:
        if not self.results_dir.exists():
            return
        for path in sorted(self.results_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            self._load_data(data, path.name)

    def _load_data(self, data: dict[str, Any], filename: str) -> None:
        metric = data.get("metric", "tok/s")
        timestamp = data.get("timestamp") or Path(filename).stem
        model = data.get("model", "unknown")

        # Sweep points
        for s in data.get("sweep", []):
            self.records.append({
                "file": filename,
                "timestamp": timestamp,
                "model": model,
                "metric": metric,
                "phase": "sweep",
                **_norm_config(s),
                "tok_per_sec": _safe_float(s.get("tok_per_sec")),
                "req_per_sec": _safe_float(s.get("req_per_sec")),
                "avg_latency": _safe_float(s.get("avg_latency")),
                "num_requests": s.get("num_requests", 0),
            })

        # Sustained race / best config
        for key in ("race", "best"):
            r = data.get(key)
            if not r:
                continue
            self.records.append({
                "file": filename,
                "timestamp": timestamp,
                "model": model,
                "metric": metric,
                "phase": "sustained" if key == "race" else "best",
                **_norm_config(r),
                "tok_per_sec": _safe_float(r.get("tok_per_sec")),
                "req_per_sec": _safe_float(r.get("req_per_sec")),
                "avg_latency": _safe_float(r.get("avg_latency")),
                "num_requests": r.get("num_requests", 0),
            })

    def count(self) -> int:
        return len(self.records)

    @staticmethod
    def _is_realistic(r: dict[str, Any]) -> bool:
        """Exclude pure cache-hit outliers where batch_time ≈ 0 inflates tok/s."""
        if r.get("avg_latency", 0) == 0 and r.get("tok_per_sec", 0) > 1_000_000:
            return False
        if r.get("tok_per_sec", 0) > 100_000_000:
            return False
        return True

    def top(self, metric: str = "tok/s", phase: str | None = None, k: int = 5) -> list[dict[str, Any]]:
        """Return top-k records by the chosen metric, optionally filtered by phase."""
        recs = self.records
        if phase:
            recs = [r for r in recs if r["phase"] == phase]
        key = "tok_per_sec" if metric == "tok/s" else "req_per_sec"
        realistic = [r for r in recs if self._is_realistic(r)]
        realistic.sort(key=lambda r: r[key], reverse=True)
        return realistic[:k]

    def insights(self) -> dict[str, Any]:
        """Aggregate insights across all loaded records."""
        if not self.records:
            return {"status": "empty", "message": "No benchmark results found in results/"}

        # Filter out pathological outliers for summary stats.
        clean = [r for r in self.records if self._is_realistic(r) and r["tok_per_sec"] > 0]

        top_tok = self.top("tok/s", k=1)
        top_req = self.top("req/s", k=1)
        top_sustained = self.top("tok/s", phase="sustained", k=1)

        return {
            "status": "ok",
            "total_records": len(self.records),
            "files_loaded": sorted({r["file"] for r in self.records}),
            "models": sorted({r["model"] for r in self.records}),
            "summary": {
                "mean_tok_per_sec": round(statistics.mean(r["tok_per_sec"] for r in clean), 2) if clean else 0,
                "median_tok_per_sec": round(statistics.median(r["tok_per_sec"] for r in clean), 2) if clean else 0,
                "configs_tested": len({r["name"] for r in self.records if r["name"] != "unknown"}),
            },
            "top_tok_per_sec": self._format_record(top_tok[0]) if top_tok else None,
            "top_req_per_sec": self._format_record(top_req[0]) if top_req else None,
            "top_sustained_tok_per_sec": self._format_record(top_sustained[0]) if top_sustained else None,
        }

    def search(self, query: str) -> dict[str, Any]:
        """Simple keyword search over config names and metrics."""
        q = query.lower().strip()
        if not q:
            return {"status": "ok", "query": q, "results": []}

        scores: dict[int, float] = {}
        terms = q.split()
        for idx, r in enumerate(self.records):
            score = 0.0
            text = f"{r['name']} {r['phase']} {r['metric']} {r['model']}".lower()
            for term in terms:
                if term in text:
                    score += 1.0
                # Boost exact config name match.
                if term == r["name"].lower():
                    score += 2.0
            if score > 0:
                scores[idx] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "status": "ok",
            "query": q,
            "results": [self._format_record(self.records[i]) for i, _ in ranked],
        }

    def compare(self, name_a: str, name_b: str) -> dict[str, Any]:
        """Compare two configs by averaging their records."""
        a_records = [r for r in self.records if r["name"].lower() == name_a.lower()]
        b_records = [r for r in self.records if r["name"].lower() == name_b.lower()]

        def _avg(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not recs:
                return None
            return {
                "name": recs[0]["name"],
                "concurrency": recs[0]["concurrency"],
                "max_tokens": recs[0]["max_tokens"],
                "runs": len(recs),
                "avg_tok_per_sec": round(statistics.mean(r["tok_per_sec"] for r in recs), 2),
                "avg_req_per_sec": round(statistics.mean(r["req_per_sec"] for r in recs), 2),
                "avg_latency": round(statistics.mean(r["avg_latency"] for r in recs), 4),
            }

        return {
            "status": "ok",
            "name_a": name_a,
            "name_b": name_b,
            "a": _avg(a_records),
            "b": _avg(b_records),
        }

    @staticmethod
    def _format_record(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "file": r["file"],
            "timestamp": r["timestamp"],
            "model": r["model"],
            "phase": r["phase"],
            "config": {
                "name": r["name"],
                "concurrency": r["concurrency"],
                "max_tokens": r["max_tokens"],
            },
            "tok_per_sec": round(r["tok_per_sec"], 2),
            "req_per_sec": round(r["req_per_sec"], 2),
            "avg_latency": round(r["avg_latency"], 4),
        }
