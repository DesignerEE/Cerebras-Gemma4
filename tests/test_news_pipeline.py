"""End-to-end tests for the Commander-Scout news pipeline.

These are the highest-leverage tests in the repo: they exercise feed parsing,
article extraction, scout summarization, commander synthesis, citation flow,
and the SSE-style event stream in a single run — all with mocked I/O so no
network or Cerebras API key is required.

Mocks target the module-level helpers `news_agents.fetch_feed` and
`news_agents.fetch_page` (the right seam — `NewsAgentTeam` accepts any
client that quacks like `CerebrasRaceClient.complete`).
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from cerebras_race_client import CompletionResult
from news_agents import NewsAgentTeam


# --- Shared fixtures ----------------------------------------------------

CANNED_RSS: dict = {
    "entries": [
        {"title": "OpenAI launches GPT-5",
         "link": "https://example.com/gpt5",
         "published": "2026-06-20",
         "summary": "OpenAI announced GPT-5 with major reasoning gains."},
        {"title": "Industry reacts to GPT-5",
         "link": "https://example.com/react",
         "published": "2026-06-20",
         "summary": "Competitors and analysts weigh in."},
    ],
}

CANNED_PAGES: dict[str, str] = {
    "https://example.com/gpt5":
        "OpenAI today launched GPT-5. It scores 95% on MMLU. "
        "Price is $5/Mtok in, $15/Mtok out.",
    "https://example.com/react":
        "Anthropic, Google, and Meta all responded. "
        "Anthropic emphasized safety. Google noted speed.",
}


class _FakeClient:
    """Drop-in for CerebrasRaceClient. Deterministic, no network."""
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass

    async def complete(self, prompt, max_completion_tokens=1000, **kw):
        p = prompt.lower()
        if "you are the commander" in p or "synthesize" in p:
            text = ("Based on scout reports, OpenAI launched GPT-5 with 95% MMLU. "
                    "Sources disagree on pricing impact. See https://example.com/gpt5.")
        else:
            text = "Scout summary: GPT-5 launched; strong benchmarks; mixed industry reaction."
        return CompletionResult(
            text=text, completion_tokens=30,
            prompt_tokens=100, total_tokens=130, latency=0.05,
        )


async def _fake_fetch_feed(url, timeout=15.0):
    return CANNED_RSS


async def _fake_fetch_page(url, timeout=12.0):
    return CANNED_PAGES.get(url, "")


# --- Tests ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_e2e_produces_cited_synthesis():
    """Single test that exercises the full news research pipeline.

    Verifies:
      - Events fire in the documented order
      - Scout returns ≥1 source with non-empty summary
      - Commander produces non-empty synthesis
      - Synthesis contains at least one URL citation
    """
    queue: asyncio.Queue = asyncio.Queue()
    events: list[dict] = []

    async def capture():
        while True:
            ev = await queue.get()
            if ev is None:
                break
            events.append(ev)

    consumer = asyncio.create_task(capture())

    with patch("news_agents.fetch_feed", side_effect=_fake_fetch_feed), \
         patch("news_agents.fetch_page", side_effect=_fake_fetch_page):
        team = NewsAgentTeam(_FakeClient(), angles=["latest breaking news"])
        report = await team.search("OpenAI GPT-5 launch", progress_queue=queue)

    await queue.put(None)
    await consumer

    # Event contract: order must match what the dashboard expects to render.
    event_types = [e["type"] for e in events]
    assert event_types == [
        "team_start", "scout_start", "scout_done",
        "commander_start", "commander_done", "team_done",
    ], f"unexpected event order: {event_types}"

    # Scout shape.
    assert len(report.scouts) == 1
    scout = report.scouts[0]
    assert len(scout.sources) >= 1, "scout produced no sources"
    assert scout.summary, "scout summary is empty"
    assert scout.error is None, f"scout reported error: {scout.error}"

    # Commander shape.
    assert report.synthesis, "commander synthesis is empty"
    assert "https://" in report.synthesis, "synthesis missing URL citation"


@pytest.mark.asyncio
async def test_scout_dedupes_articles_across_feeds():
    """Regression: same article URL appearing in multiple feeds should be
    fetched and counted once per scout. Before the fix, this scenario
    produced N copies of the same source and N duplicate HTTP fetches.
    """
    fetched_pages: list[str] = []

    async def tracking_fake_page(url, timeout=12.0):
        fetched_pages.append(url)
        return CANNED_PAGES.get(url, "")

    with patch("news_agents.fetch_feed", side_effect=_fake_fetch_feed), \
         patch("news_agents.fetch_page", side_effect=tracking_fake_page):
        # The "latest breaking news" angle has 4 feeds. With CANNED_RSS
        # returning the same 2 entries for every feed, the scout sees
        # 4×2 = 8 raw entries but only 2 unique URLs.
        team = NewsAgentTeam(_FakeClient(), angles=["latest breaking news"])
        report = await team.search("test query")

    scout = report.scouts[0]
    urls = [s.url for s in scout.sources]

    # Each URL appears at most once in the final source list.
    assert len(urls) == len(set(urls)), f"duplicate URLs in scout sources: {urls}"

    # fetch_page was called once per unique URL, not once per (feed, entry).
    assert len(fetched_pages) == len(set(fetched_pages)), \
        f"fetch_page called for duplicate URLs: {fetched_pages}"

    # And the count is bounded by max_articles (default 4) — here 2 unique URLs.
    assert len(urls) <= 4
    assert len(urls) >= 1
