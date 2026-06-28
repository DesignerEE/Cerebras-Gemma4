"""Tests for the VisionOps image-aware agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from cfire import AsyncCfire, ChatResponse
from cfire.models import Choice, Message, Usage
from vision_ops import VisionOpsAgent, Diagnosis, Action


class _FakeBackend:
    """Backend that returns canned vision responses without network."""

    base_url = "mock://"

    def __init__(self, response_text: str):
        self._response_text = response_text

    async def complete(self, request):
        return ChatResponse(
            id="test",
            model="gemma-4-31b",
            choices=[Choice(message=Message(role="assistant", content=self._response_text))],
            usage=Usage(prompt_tokens=100, completion_tokens=80, total_tokens=180),
            latency=0.05,
        )

    async def stream(self, request):
        yield {}

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_analyze_returns_structured_diagnosis():
    canned = json.dumps({
        "summary": "CPU spike on api-gateway pod",
        "severity": "warning",
        "root_cause": "Traffic surge exceeded replica capacity",
        "actions": [
            {"name": "Scale api-gateway", "description": "Add replicas", "command": "kubectl scale deploy api-gateway --replicas=5", "safe_to_run": False},
        ],
    })
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend(canned), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes", mime_type="image/png")

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.summary == "CPU spike on api-gateway pod"
    assert diagnosis.severity == "warning"
    assert len(diagnosis.actions) == 1
    assert diagnosis.actions[0].name == "Scale api-gateway"


@pytest.mark.asyncio
async def test_analyze_extracts_json_from_markdown():
    canned = "```json\n" + json.dumps({
        "summary": "Memory pressure detected",
        "severity": "critical",
        "actions": [],
    }) + "\n```"
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend(canned), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert diagnosis.summary == "Memory pressure detected"
    assert diagnosis.severity == "critical"


@pytest.mark.asyncio
async def test_analyze_extracts_json_from_xml_tags():
    canned = '<json>\n' + json.dumps({
        "summary": "Disk usage high",
        "severity": "warning",
        "actions": [],
    }) + '\n</json>'
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend(canned), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert diagnosis.summary == "Disk usage high"
    assert diagnosis.severity == "warning"


@pytest.mark.asyncio
async def test_analyze_extracts_json_from_malformed_gemma_output():
    """Gemma-4-31b occasionally emits junk before/after the JSON object."""
    canned = '{"/></json> {\n  "summary": "Conceptual illustration, not a screenshot",\n  "severity": "info",\n  "root_cause": "N/A",\n  "actions": []\n}'
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend(canned), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert "Conceptual illustration" in diagnosis.summary
    assert diagnosis.severity == "info"


@pytest.mark.asyncio
async def test_analyze_gracefully_handles_bad_json():
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend("not json"), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.summary  # summary is populated with salvaged text or fallback text
    assert diagnosis.severity in ("info", "warning", "critical")


@pytest.mark.asyncio
async def test_analyze_handles_partial_empty_json():
    """The model sometimes returns a valid but empty JSON object."""
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend("{}"), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert isinstance(diagnosis, Diagnosis)
    assert "did not provide" in diagnosis.summary or "unstructured" in diagnosis.summary.lower()


@pytest.mark.asyncio
async def test_analyze_salvages_gemma_garbage_output():
    """Regression for Gemma-4-31b emitting brace/punctuation noise."""
    garbage = '{\"): { ":" } } { ","+ "  :"" ,"@{ "  :"" ,"@{ "  :"" ,"@{ "  :""'
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend(garbage), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.severity in ("info", "warning", "critical")
    assert "unstructured" in diagnosis.summary.lower() or "malformed" in diagnosis.summary.lower()


@pytest.mark.asyncio
async def test_analyze_parses_delimited_format():
    canned = """SUMMARY: Conceptual illustration of the Sutton Cycle and latent space
SEVERITY: info
ROOT_CAUSE: The image is digital artwork explaining reinforcement-learning concepts, not a technical monitoring screen.
ACTIONS:
- Explain Sutton Cycle | Provide a brief explanation of the Sutton Cycle in RL | echo "Sutton Cycle: model-based RL loop" | safe
END"""
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend(canned), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert diagnosis.summary == "Conceptual illustration of the Sutton Cycle and latent space"
    assert diagnosis.severity == "info"
    assert "reinforcement-learning" in diagnosis.root_cause.lower()
    assert len(diagnosis.actions) == 1
    assert diagnosis.actions[0].name == "Explain Sutton Cycle"


@pytest.mark.asyncio
async def test_analyze_parses_partial_malformed_json():
    """Gemma sometimes emits almost-JSON with corrupted keys and truncation."""
    canned = '{\">=summary": "Conceptual illustration of the Sutton Cycle and multidimensional latent space, not an infrastructure screenshot.",\n  "severity": "info",\n  "root_cause": "The provided image is digital a'
    agent = VisionOpsAgent(client=AsyncCfire(backend=_FakeBackend(canned), enable_cache=False))
    diagnosis = await agent.analyze(b"fake-image-bytes")

    assert isinstance(diagnosis, Diagnosis)
    assert "Sutton Cycle" in diagnosis.summary
    assert diagnosis.severity == "info"


def test_diagnosis_model_validation():
    d = Diagnosis(
        summary="test",
        actions=[Action(name="restart", description="restart service", command="/bin/restart")],
    )
    assert d.actions[0].safe_to_run is False
