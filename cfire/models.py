"""Pydantic v2 models for cfire.

Replaces the @dataclass CompletionResult (cerebras_race_client.py:65-74)
with an OpenAI-compatible ChatResponse that also carries the
Cerebras-unique time_info object.

Backward-compat:
  - ChatResponse.text is a computed property -> choices[0].message.content
  - ChatResponse keeps latency / cached / compressed as client-side flags
  - CompletionResult in cfire._compat subclasses ChatResponse so the old
    field-name access (.completion_tokens etc.) still works for legacy code

Forward-compat:
  - time_info: TimeInfo | None   (only present on CerebrasBackend)
  - usage.reasoning_tokens       (OpenAI reasoning-model extension)
  - ChatRequest carries all Cerebras-specific params (service_tier,
    prompt_cache_key, predicted_output, etc.) so users can set them
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


# --- Request side --------------------------------------------------------

Role = Literal["system", "developer", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single chat message. `developer` replaces `system` for gpt-oss-120b."""
    model_config = ConfigDict(extra="allow")
    role: Role
    content: str


class PredictedOutput(BaseModel):
    """Cerebras-unique: pre-supplied content for regeneration-heavy workloads.

    Supplying the known prefix of the expected output lets the server skip
    re-encoding it, slashing TTFT for code-edit / fill-in-the-middle tasks.
    """
    content: str | list[dict[str, Any]] | None = None
    type: Literal["content"] = "content"


class ResponseFormat(BaseModel):
    """OpenAI-compatible response_format for structured outputs."""
    model_config = ConfigDict(extra="allow")
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    """What the user sends. Maps 1:1 to the Cerebras /chat/completions body."""
    model_config = ConfigDict(extra="allow")
    model: str = "gpt-oss-120b"
    messages: list[Message]
    max_completion_tokens: int = 1000
    temperature: float = 0.3
    top_p: float = 1.0
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    clear_thinking: bool | None = None
    service_tier: Literal["flex", "default", "auto", "priority"] = "default"
    prompt_cache_key: str | None = None
    predicted_output: PredictedOutput | None = None
    response_format: ResponseFormat | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = None


# --- Response side -------------------------------------------------------

class Usage(BaseModel):
    """Token usage. reasoning_tokens is the OpenAI reasoning-model extension."""
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> "Usage":
        """Build from the raw `usage` object in a Cerebras response.

        Handles the nested completion_tokens_details.reasoning_tokens path.
        """
        if not raw:
            return cls()
        details = raw.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=raw.get("prompt_tokens", 0),
            completion_tokens=raw.get("completion_tokens", 0),
            total_tokens=raw.get("total_tokens", raw.get("completion_tokens", 0)
                                 + raw.get("prompt_tokens", 0)),
            reasoning_tokens=details.get("reasoning_tokens", 0),
        )


class TimeInfo(BaseModel):
    """Cerebras-unique per-response latency breakdown.

    Other backends (local, CDN) won't populate this — leave as None there.
    """
    model_config = ConfigDict(extra="allow")
    queue_time: float = 0.0
    prompt_time: float = 0.0
    completion_time: float = 0.0
    total_time: float = 0.0

    @classmethod
    def from_api(cls, raw: dict[str, Any] | None) -> "TimeInfo | None":
        if not raw:
            return None
        return cls(
            queue_time=float(raw.get("queue_time", 0.0)),
            prompt_time=float(raw.get("prompt_time", 0.0)),
            completion_time=float(raw.get("completion_time", 0.0)),
            total_time=float(raw.get("total_time", 0.0)),
        )


class Choice(BaseModel):
    """One choice from the response. Cerebras returns exactly one by default."""
    model_config = ConfigDict(extra="allow")
    index: int = 0
    message: Message
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    """Full response. Carries both server fields and client-side metadata."""
    model_config = ConfigDict(extra="allow")
    id: str = ""
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    time_info: TimeInfo | None = None
    # Client-side metadata (not from server):
    latency: float = 0.0
    cached: bool = False
    compressed: bool = False

    @property
    def text(self) -> str:
        """Convenience: choices[0].message.content, or empty string."""
        if not self.choices:
            return ""
        return self.choices[0].message.content

    @computed_field  # type: ignore[misc]
    @property
    def completion_tokens(self) -> int:
        """Back-compat with legacy CompletionResult field name."""
        return self.usage.completion_tokens

    @property
    def prompt_tokens(self) -> int:
        """Back-compat with legacy CompletionResult field name."""
        return self.usage.prompt_tokens

    @property
    def total_tokens(self) -> int:
        """Back-compat with legacy CompletionResult field name."""
        return self.usage.total_tokens

    @property
    def reasoning_tokens(self) -> int:
        """Back-compat with legacy CompletionResult field name."""
        return self.usage.reasoning_tokens


# --- Streaming -----------------------------------------------------------

class StreamChunk(BaseModel):
    """One SSE delta from a streaming response."""
    model_config = ConfigDict(extra="allow")
    delta: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None


__all__ = [
    "Role",
    "Message",
    "PredictedOutput",
    "ResponseFormat",
    "ChatRequest",
    "Usage",
    "TimeInfo",
    "Choice",
    "ChatResponse",
    "StreamChunk",
]
