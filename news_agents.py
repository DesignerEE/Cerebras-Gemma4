#!/usr/bin/env python3
"""
Commander-Scout News Search Agents — RSS Edition

Scouts fan out across RSS feeds grouped by angle,
then the commander synthesizes a unified, cited report via LLM.

Uses:
  - feedparser for RSS feeds
  - httpx + BeautifulSoup for article extraction
  - CerebrasRaceClient for summarization / synthesis
"""

from __future__ import annotations

import asyncio
import re
import time

from pydantic import BaseModel, Field, computed_field

import feedparser
import httpx
from bs4 import BeautifulSoup

from cerebras_race_client import CerebrasRaceClient


# RSS feeds grouped by scout angle. Mix of mainstream tech/business/news sources.
RSS_FEEDS = {
    "latest breaking news": [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://feeds.npr.org/1001/rss.xml",
        "https://www.theguardian.com/uk/rss",
    ],
    "technical details analysis": [
        "https://news.ycombinator.com/rss",
        "https://techcrunch.com/feed/",
        "https://www.engadget.com/rss.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.theverge.com/rss/index.xml",
    ],
    "market and industry reaction": [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.ft.com/?format=rss",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    ],
}


class Source(BaseModel):
    title: str
    url: str
    snippet: str
    angle: str = ""
    published: str = ""


class ScoutReport(BaseModel):
    angle: str
    query: str
    sources: list[Source] = Field(default_factory=list)
    summary: str = ""
    elapsed: float = 0.0
    error: str | None = None


class CommanderReport(BaseModel):
    query: str
    scouts: list[ScoutReport]
    synthesis: str = ""
    elapsed: float = 0.0

    @computed_field
    @property
    def total_elapsed(self) -> float:
        """Wall-clock elapsed time including scout fan-out."""
        return self.elapsed + sum(s.elapsed for s in self.scouts)


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


async def fetch_page(url: str, timeout: float = 12.0) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                },
            )
            resp.raise_for_status()
            return clean_text(resp.text)
    except Exception as e:
        return f"[fetch error: {type(e).__name__}: {e}]"


async def fetch_feed(url: str, timeout: float = 15.0) -> dict:
    """Fetch and parse an RSS feed. Returns feedparser dict."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                },
            )
            resp.raise_for_status()
            return feedparser.parse(resp.text)
    except Exception as e:
        return {"bozo": True, "bozo_exception": e, "entries": []}


def score_relevance(entry: dict, query: str, angle: str) -> float:
    """Score how relevant an RSS entry is to the query + angle."""
    text = " ".join([
        entry.get("title", ""),
        entry.get("summary", ""),
        entry.get("description", ""),
    ]).lower()

    query_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    angle_words = [w for w in re.findall(r"\w+", angle.lower()) if len(w) > 2]

    score = 0.0
    for w in query_words:
        if w in text:
            score += 1.0
    for w in angle_words:
        if w in text:
            score += 0.3

    # Boost recent entries if published date exists
    if entry.get("published_parsed"):
        score += 0.1

    return score


class NewsScout:
    """Single scout with one angle and its associated RSS feeds."""

    def __init__(
        self,
        angle: str,
        client: CerebrasRaceClient,
        feeds: list[str] | None = None,
        max_articles: int = 4,
        max_page_chars: int = 6000,
    ):
        self.angle = angle
        self.client = client
        self.feeds = feeds or RSS_FEEDS.get(angle, [])
        self.max_articles = max_articles
        self.max_page_chars = max_page_chars

    async def run(self, query: str, progress_queue: asyncio.Queue | None = None) -> ScoutReport:
        start = time.perf_counter()
        report = ScoutReport(angle=self.angle, query=query)

        try:
            await self._emit(progress_queue, "scout_start", {"angle": self.angle})

            if not self.feeds:
                report.error = "no feeds configured"
                await self._emit(progress_queue, "scout_done", {"angle": self.angle, "error": report.error})
                return report

            # Fetch all feeds for this angle in parallel
            feed_data = await asyncio.gather(*[fetch_feed(url) for url in self.feeds])

            # Collect and score entries (dedupe by link across feeds)
            scored_entries = []
            seen_links: set[str] = set()
            for feed in feed_data:
                for entry in feed.get("entries", []):
                    link = entry.get("link", "")
                    if link:
                        if link in seen_links:
                            continue
                        seen_links.add(link)
                    score = score_relevance(entry, query, self.angle)
                    scored_entries.append((score, entry))

            # Sort by relevance and take top N (even if score is low; LLM will filter)
            scored_entries.sort(key=lambda x: x[0], reverse=True)
            top_entries = [e for _, e in scored_entries[:self.max_articles]]

            if not top_entries:
                report.error = "no relevant articles found"
                await self._emit(progress_queue, "scout_done", {"angle": self.angle, "error": report.error})
                return report

            # Fetch article pages
            urls = [e.get("link", "") for e in top_entries]
            pages = await asyncio.gather(*[fetch_page(url) for url in urls])

            for entry, page_text in zip(top_entries, pages):
                source = Source(
                    title=entry.get("title", "Untitled"),
                    url=entry.get("link", ""),
                    snippet=entry.get("summary", "")[:500],
                    angle=self.angle,
                    published=entry.get("published", ""),
                )
                report.sources.append(source)

            # Summarize findings with LLM
            combined = "\n\n---\n\n".join(
                f"TITLE: {s.title}\nURL: {s.url}\nPUBLISHED: {s.published}\nSNIPPET: {s.snippet}\nBODY: {pages[i][:self.max_page_chars]}"
                for i, s in enumerate(report.sources)
            )

            prompt = (
                f"You are a news scout focusing on: {self.angle}\n"
                f"Query: {query}\n\n"
                f"Extract the 3-5 most important facts, claims, or developments relevant to this angle. "
                f"Be concise, factual, and cite sources by URL.\n\n{combined}"
            )
            result = await self.client.complete(
                prompt,
                max_completion_tokens=600,
                temperature=0.2,
                reasoning_effort="low",
            )
            report.summary = result.text

            await self._emit(progress_queue, "scout_done", {
                "angle": self.angle,
                "sources": len(report.sources),
                "summary": report.summary[:200],
            })

        except Exception as e:
            report.error = str(e)
            await self._emit(progress_queue, "scout_error", {"angle": self.angle, "error": report.error})

        report.elapsed = time.perf_counter() - start
        return report

    async def _emit(self, queue: asyncio.Queue | None, event_type: str, data: dict):
        if queue:
            await queue.put({"type": event_type, **data})


class NewsCommander:
    """Synthesizes scout reports into one coherent answer."""

    def __init__(self, client: CerebrasRaceClient):
        self.client = client

    async def _emit(self, queue: asyncio.Queue | None, event_type: str, data: dict):
        if queue:
            await queue.put({"type": event_type, **data})

    async def synthesize(
        self,
        query: str,
        scouts: list[ScoutReport],
        progress_queue: asyncio.Queue | None = None,
    ) -> CommanderReport:
        start = time.perf_counter()
        report = CommanderReport(query=query, scouts=scouts)

        await self._emit(progress_queue, "commander_start", {"query": query})

        briefing = []
        for s in scouts:
            briefing.append(f"## Scout angle: {s.angle}\n{s.summary}\n")
            briefing.append("Sources:")
            for src in s.sources:
                briefing.append(f"- {src.title}: {src.url}")
            briefing.append("")

        prompt = (
            f"You are the commander. Synthesize the following scout reports into a coherent, "
            f"well-structured answer to the user's query. Include key facts, disagreements between sources, "
            f"and a concise conclusion. Cite sources by URL.\n\n"
            f"Query: {query}\n\n"
            f"{chr(10).join(briefing)}"
        )

        result = await self.client.complete(
            prompt,
            max_completion_tokens=1200,
            temperature=0.3,
            reasoning_effort="low",
        )
        report.synthesis = result.text
        report.elapsed = time.perf_counter() - start

        await self._emit(progress_queue, "commander_done", {
            "query": query,
            "synthesis": report.synthesis[:300],
            "elapsed": report.elapsed,
        })
        return report


class NewsAgentTeam:
    """Full commander-scout team for a query."""

    DEFAULT_ANGLES = list(RSS_FEEDS.keys())

    def __init__(self, client: CerebrasRaceClient, angles: list[str] | None = None):
        self.client = client
        self.angles = angles or self.DEFAULT_ANGLES

    async def search(
        self,
        query: str,
        progress_queue: asyncio.Queue | None = None,
    ) -> CommanderReport:
        await self._emit(progress_queue, "team_start", {"query": query, "angles": self.angles})

        scouts = [NewsScout(angle, self.client) for angle in self.angles]
        scout_reports = await asyncio.gather(*[s.run(query, progress_queue) for s in scouts])

        commander = NewsCommander(self.client)
        report = await commander.synthesize(query, scout_reports, progress_queue)

        await self._emit(progress_queue, "team_done", {
            "query": query,
            "elapsed": report.elapsed + sum(s.elapsed for s in scout_reports),
        })
        return report

    async def _emit(self, queue: asyncio.Queue | None, event_type: str, data: dict):
        if queue:
            await queue.put({"type": event_type, **data})
