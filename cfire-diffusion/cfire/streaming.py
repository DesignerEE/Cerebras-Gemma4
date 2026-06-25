"""Real SSE streaming for cfire.

Fix #4: legacy complete_stream (cerebras_race_client.py:424-439) set
stream=True on the payload but then called resp.json() and yielded the
whole text at once. This module does it properly — one StreamChunk per
SSE `data:` block.

SSE format from Cerebras /chat/completions with stream=true:

    data: {"choices":[{"delta":{"content":"Hello"}}]}

    data: {"choices":[{"delta":{"content":", world"}}]}

    data: {"choices":[{"finish_reason":"stop"}],"usage":{...}}

    data: [DONE]

Blocks are separated by blank lines (\\n\\n). Lines start with `data: `.
The final block may carry usage for accounting; we forward it as a
StreamChunk with delta="" and usage populated.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from .models import StreamChunk, Usage


def parse_chunk_obj(obj: dict[str, Any]) -> StreamChunk | None:
    """Convert one parsed SSE JSON object into a StreamChunk.

    Returns None if the object has no useful payload (e.g. only role deltas
    at the start of a stream).
    """
    choices = obj.get("choices") or []
    delta_text = ""
    finish_reason: str | None = None
    if choices:
        first = choices[0]
        finish_reason = first.get("finish_reason")
        msg = first.get("delta") or first.get("message") or {}
        delta_text = msg.get("content") or ""

    usage_obj = obj.get("usage")
    usage = Usage.from_api(usage_obj) if usage_obj else None

    if not delta_text and finish_reason is None and usage is None:
        return None

    return StreamChunk(delta=delta_text, finish_reason=finish_reason, usage=usage)


async def parse_sse_stream(resp: httpx.Response) -> AsyncIterator[StreamChunk]:
    """Yield StreamChunk per `data:` block in an SSE response.

    Stops cleanly at `data: [DONE]`. Handles blocks split across HTTP chunks
    via an internal buffer.
    """
    buffer = ""
    async for text in resp.aiter_text():
        buffer += text
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            chunk = _parse_block(block)
            if chunk is _DONE_SENTINEL:
                return
            if chunk is not None:
                yield chunk
    # Flush any trailing block
    if buffer.strip():
        chunk = _parse_block(buffer)
        if chunk is _DONE_SENTINEL:
            return
        if chunk is not None:
            yield chunk


_DONE_SENTINEL = object()


def _parse_block(block: str) -> StreamChunk | None | object:
    """Parse one SSE block (already split from the stream).

    A block may contain multiple lines (e.g. `event: ping\\n` followed by
    `data: {...}`). The SSE spec lets senders mix comment (`:`), `event:`,
    `id:`, and `data:` lines freely within a single block. We scan for the
    first `data:` line and parse it; everything else is ignored.

    Returns:
      StreamChunk    -> a usable delta
      None           -> ignore (no data: line, or unparseable)
      _DONE_SENTINEL -> terminal marker, caller stops iterating
    """
    data_line: str | None = None
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_line = line[5:].strip()
            break  # first data: line wins
        # event:, id:, retry:, etc. — ignore
    if data_line is None:
        return None
    if data_line == "[DONE]":
        return _DONE_SENTINEL
    try:
        obj = json.loads(data_line)
    except json.JSONDecodeError:
        return None
    return parse_chunk_obj(obj)


__all__ = ["parse_sse_stream", "parse_chunk_obj"]
