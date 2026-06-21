#!/usr/bin/env python3
"""Compatibility shim — re-exports from cfire.

This file was the original 475-line race client. After the Phase 2
extraction into `cfire/`, the implementation lives there. This thin
wrapper keeps the old import path working so news_agents.py,
web_demo.py, cerebras_race_advanced.py, and tests/test_news_pipeline.py
keep their `from cerebras_race_client import CerebrasRaceClient,
CompletionResult` lines unchanged.

Phase 4 will delete this file once every consumer migrates to direct
`from cfire import AsyncCfire, ChatResponse` imports.
"""

from __future__ import annotations

from cfire._compat import CompletionResult, CerebrasRaceClient

# Re-export the public surface any consumer might still reach for.
# Direct module-level references like `cerebras_race_client.Metrics`
# are intentionally NOT supported — the new Metrics lives in
# cfire.metrics and uses a callback observer pattern. Any consumer
# needing it should `from cfire.metrics import Metrics`.

__all__ = ["CompletionResult", "CerebrasRaceClient"]
