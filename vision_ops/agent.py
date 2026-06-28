"""VisionOps agent: analyze infrastructure screenshots and suggest actions.

The agent is intentionally lightweight for the hackathon:
  - one image in
  - one structured diagnosis + action list out
  - optional streaming log events for the dashboard
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, AsyncIterator, Callable

from pydantic import BaseModel, Field

from cfire import AsyncCfire, ChatRequest
from cfire.models import Message


class Action(BaseModel):
    """A suggested remediation action."""

    name: str
    description: str
    command: str = ""
    safe_to_run: bool = False


class Diagnosis(BaseModel):
    """Structured result from analyzing a screenshot."""

    summary: str
    severity: str = "info"  # info | warning | critical
    root_cause: str = ""
    actions: list[Action] = Field(default_factory=list)


DIAGNOSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
        "root_cause": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "command": {"type": "string"},
                    "safe_to_run": {"type": "boolean"},
                },
                "required": ["name", "description", "command", "safe_to_run"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "severity", "root_cause", "actions"],
    "additionalProperties": False,
}


DEFAULT_SYSTEM_PROMPT = """You are VisionOps, an expert visual analyst. Look at the provided image, understand its context, and describe what it shows.

When the image contains infrastructure screenshots (dashboards, alerts, logs, metrics, architecture diagrams), act as an SRE and provide a diagnosis with severity, root cause, and suggested actions.

When the image contains something else (artwork, diagrams, slides, UI mockups, photographs, etc.), describe the content accurately and set severity to "info". Do not dismiss it as merely "not an infrastructure screenshot"; explain what is actually depicted.

Respond ONLY with a single JSON object matching the provided schema. Do not add markdown, code fences, XML tags, or any other formatting."""


class VisionOpsAgent:
    """Image-aware SRE agent powered by Gemma 4 31B on Cerebras Inference."""

    def __init__(
        self,
        client: AsyncCfire | None = None,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.client = client
        self.model = model or os.environ.get("VISIONOPS_MODEL", "gemma-4-31b")
        self.system_prompt = system_prompt
        self.event_callback = event_callback or (lambda _: None)

    async def _get_client(self) -> AsyncCfire:
        if self.client is None:
            self.client = AsyncCfire(model=self.model)
            await self.client.open()
        return self.client

    def _emit(self, event: dict[str, Any]) -> None:
        self.event_callback(event)

    @staticmethod
    def _image_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    async def analyze(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        context: str = "",
    ) -> Diagnosis:
        """Analyze a screenshot and return a structured diagnosis."""
        self._emit({"type": "log", "who": "Vision", "text": "Encoding screenshot..."})
        data_url = self._image_data_url(image_bytes, mime_type)

        content_parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        prompt = (
            "Analyze this image and return a JSON diagnosis.\n"
            "\n"
            "Guidelines:\n"
            "- Be concise but descriptive. Explain the actual content of the image.\n"
            "- If the image is an infrastructure dashboard, alert, or log: identify the unhealthy service/metric, root cause, and remediation actions.\n"
            "- If the image is a diagram, flowchart, or architecture: describe the components and relationships briefly.\n"
            "- If the image is artwork, a slide, a UI mockup, or a photo: describe the subject, style, and purpose; severity must be 'info'.\n"
            "- Only mark actions safe_to_run if they are read-only or clearly reversible.\n"
            "- If no actions are needed, return an empty actions list."
        )
        if context:
            prompt += f"\nAdditional context: {context}"
        content_parts.append({"type": "text", "text": prompt})

        self._emit({"type": "log", "who": "Vision", "text": f"Sending {len(image_bytes) // 1024} KB image to {self.model}..."})

        client = await self._get_client()
        request = ChatRequest(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content_parts},
            ],
            max_completion_tokens=800,
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "diagnosis", "schema": DIAGNOSIS_JSON_SCHEMA},
            },
        )

        response = await client.complete(request)
        raw = response.text or ""

        self._emit({"type": "log", "who": "Vision", "text": f"Response received ({response.usage.completion_tokens} tokens, {response.latency*1000:.0f} ms)"})

        data = self._parse_response(raw)

        diagnosis = Diagnosis.model_validate(data)
        self._emit({"type": "diagnosis", "diagnosis": diagnosis.model_dump()})
        return diagnosis

    async def analyze_stream(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        context: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """Analyze a screenshot and yield log/diagnosis events."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def callback(event: dict[str, Any]) -> None:
            queue.put_nowait(event)

        original_callback = self.event_callback
        self.event_callback = callback
        try:
            task = asyncio.create_task(self.analyze(image_bytes, mime_type, context))
            done = False
            while not done:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.2)
                    yield event
                except asyncio.TimeoutError:
                    if task.done():
                        done = True
                        try:
                            diagnosis = await task
                            yield {"type": "diagnosis", "diagnosis": diagnosis.model_dump()}
                        except Exception as e:
                            yield {"type": "log", "who": "Vision", "text": f"Analysis failed: {e}", "level": "error"}
        finally:
            self.event_callback = original_callback

    @classmethod
    def _parse_response(cls, raw: str) -> dict[str, Any]:
        """Parse the model response into a diagnosis dict.

        The preferred format is the SUMMARY/SEVERITY/ROOT_CAUSE/ACTIONS/END block.
        Falls back to JSON extraction and finally to free-text salvage.
        """
        text = raw.strip()

        # 1. Preferred delimited format
        parsed = cls._parse_delimited(text)
        if parsed is not None:
            return parsed

        # 2. JSON fallback
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return cls._normalize_diagnosis(data)
        except json.JSONDecodeError:
            code = cls._extract_json(text)
            try:
                data = json.loads(code)
                if isinstance(data, dict):
                    return cls._normalize_diagnosis(data)
            except json.JSONDecodeError:
                pass

        # 3. Free-text salvage
        return cls._salvage_diagnosis(text)

    @staticmethod
    def _normalize_diagnosis(data: dict[str, Any]) -> dict[str, Any]:
        """Ensure a parsed dict has the expected Diagnosis shape."""
        normalized: dict[str, Any] = {
            "summary": data.get("summary") or "The model did not provide a summary for this image.",
            "severity": data.get("severity") or "info",
            "root_cause": data.get("root_cause") or "N/A",
            "actions": [],
        }
        for action in data.get("actions") or []:
            if isinstance(action, dict):
                normalized["actions"].append({
                    "name": action.get("name", "Action"),
                    "description": action.get("description", ""),
                    "command": action.get("command", ""),
                    "safe_to_run": bool(action.get("safe_to_run", False)),
                })
        return normalized

    @staticmethod
    def _parse_delimited(text: str) -> dict[str, Any] | None:
        """Parse SUMMARY/SEVERITY/ROOT_CAUSE/ACTIONS/END blocks."""
        import re

        # Loose matching: allow lowercase headers and tolerate leading punctuation
        def header_pattern(name: str) -> str:
            return rf'^[\s{{}}"]*{name}[\s"]*[:=][\s"]*'

        def strict_header_pattern(name: str) -> str:
            return rf'^\s*{name}\s*[:=]\s*'

        lines = text.splitlines()
        if not lines:
            return None

        # Check that it looks like our delimited format (no leading braces)
        if not re.match(strict_header_pattern(r'SUMMARY'), lines[0], re.IGNORECASE):
            return None

        result: dict[str, Any] = {
            "summary": "",
            "severity": "info",
            "root_cause": "N/A",
            "actions": [],
        }

        current_field: str | None = None
        current_value: list[str] = []

        def flush() -> None:
            nonlocal current_field, current_value
            if current_field is None:
                return
            joined = " ".join(current_value).strip()
            if current_field == "summary":
                result["summary"] = joined
            elif current_field == "severity":
                sev = joined.lower()
                result["severity"] = sev if sev in ("info", "warning", "critical") else "info"
            elif current_field == "root_cause":
                result["root_cause"] = joined
            current_field = None
            current_value = []

        in_actions = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line == "END":
                flush()
                break

            if re.match(header_pattern(r'SUMMARY'), line, re.IGNORECASE):
                flush()
                current_field = "summary"
                current_value = [re.sub(header_pattern(r'SUMMARY'), '', line, flags=re.IGNORECASE)]
            elif re.match(header_pattern(r'SEVERITY'), line, re.IGNORECASE):
                flush()
                current_field = "severity"
                current_value = [re.sub(header_pattern(r'SEVERITY'), '', line, flags=re.IGNORECASE)]
            elif re.match(header_pattern(r'ROOT[_\s]?CAUSE'), line, re.IGNORECASE):
                flush()
                current_field = "root_cause"
                current_value = [re.sub(header_pattern(r'ROOT[_\s]?CAUSE'), '', line, flags=re.IGNORECASE)]
            elif re.match(r'^[\s{}"]*ACTIONS[\s"]*[:=]?', line, re.IGNORECASE):
                flush()
                in_actions = True
            elif in_actions and line.startswith("-"):
                # action line: - name | description | command | safe_to_run
                parts = [p.strip() for p in line[1:].split("|")]
                if len(parts) >= 1 and parts[0]:
                    result["actions"].append({
                        "name": parts[0],
                        "description": parts[1] if len(parts) > 1 else "",
                        "command": parts[2] if len(parts) > 2 else "",
                        "safe_to_run": parts[-1].lower() in ("true", "yes", "safe") if parts else False,
                    })
            elif current_field is not None:
                current_value.append(line)

        flush()

        if not result["summary"]:
            return None
        return result

    @staticmethod
    def _salvage_diagnosis(raw: str) -> dict[str, Any]:
        """Best-effort extraction of a diagnosis from unstructured model output.

        Tries to pull out the longest coherent sentence for the summary, looks
        for severity keywords, and keeps the original snippet as root_cause so
        the user can see what the model emitted.
        """
        import re

        text = raw.strip()
        # Remove obvious JSON-like debris so we can read any English prose.
        cleaned = re.sub(r'[{}\[\]"@+]+', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Severity heuristic
        lower = text.lower()
        if any(w in lower for w in ("critical", "down", "outage", "failure", "error")):
            severity = "critical"
        elif any(w in lower for w in ("warning", "degraded", "slow", "high", "alert")):
            severity = "warning"
        else:
            severity = "info"

        # Use the longest sentence-like chunk as the summary, but only if it
        # contains real words (letters). Pure punctuation/junk falls back.
        sentences = [s.strip() for s in re.split(r'[.!?\n]', cleaned) if len(s.strip()) > 5]
        readable = [s for s in sentences if re.search(r'[a-zA-Z]{3,}', s)]
        if readable:
            summary = max(readable, key=len)
            # Strip leftover JSON key names like "summary :" or "root cause :"
            summary = re.sub(r'^(summary|root[_\s]cause|description)\s*[:=]\s*', '', summary, flags=re.IGNORECASE).strip()
        elif sentences:
            summary = "The model returned malformed output for this image."
        else:
            summary = "The model returned unstructured output for this image."

        return {
            "summary": summary[:250],
            "severity": severity,
            "root_cause": f"Unstructured model output: {text[:200]}",
            "actions": [],
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        """Best-effort JSON extraction from markdown, XML tags, or prose.

        Tries, in order:
        1. Explicit <json>...</json> tags.
        2. Markdown ```json fences.
        3. Markdown ``` fences.
        4. The first balanced/valid JSON object found by scanning braces.
        """
        stripped = text.strip()

        # 1. <json>...</json> tags
        if "<json>" in stripped and "</json>" in stripped:
            inner = stripped.split("<json>", 1)[1].split("</json>", 1)[0].strip()
            # The model sometimes emits malformed openers like `{" /></json>`;
            # try the inner content first, fall back to the full text below.
            try:
                json.loads(inner)
                return inner
            except json.JSONDecodeError:
                pass

        # 2. Markdown code fences
        if "```json" in stripped:
            return stripped.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in stripped:
            return stripped.split("```", 1)[1].split("```", 1)[0].strip()

        # 3. Find the first valid JSON object by scanning brace positions.
        #    This handles outputs like `{"/>
        #    </json> { "summary": "..." }` where the first brace is junk.
        start = stripped.find("{")
        while start != -1:
            brace_depth = 0
            in_string = False
            escape_next = False
            for i, ch in enumerate(stripped[start:], start=start):
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not in_string:
                    in_string = True
                elif ch == '"' and in_string:
                    in_string = False
                elif not in_string:
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                        if brace_depth == 0:
                            candidate = stripped[start : i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                break
            start = stripped.find("{", start + 1)

        # 4. Last resort: first '{' to last '}'
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1]
        return stripped
