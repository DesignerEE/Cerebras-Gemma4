# cfire — DiffusionGemma variant

**The Cerebras + DiffusionGemma routing fork of [`cfire`](../README.md).**

This is the **0.2.0** variant of `cfire` that adds a `DiffusionGemmaBackend` and routes code-shaped requests to a local [DiffusionGemma](https://github.com/NVIDIA/DiffusionGemma) server (default `nvidia/diffusiongemma-26B-A4B-it-NVFP4`), while general chat continues to hit Cerebras `gemma-4-31b` for first-class `time_info` telemetry.

## Why this variant exists

DiffusionGemma is a text-to-text diffusion model — it generates text via iterative denoising rather than autoregressive token prediction. Empirically it performs well on **code-completion and structured-edit workloads** (where the target distribution is narrow and the denoising process can exploit it), while Cerebras Gemma 4 31B wins on open-ended generation and reasoning.

This fork lets one `AsyncCfire` client transparently dispatch:

- **Code / regeneration-heavy requests** → local DiffusionGemma (no queue, free, fast on narrow distributions)
- **General chat and reasoning** → Cerebras `gemma-4-31b` (~1,500 TPS, full `time_info` telemetry)

The router decides based on `RoutingPolicy` patterns.

## What's new vs parent `cfire` 0.1.0

| Change | Reason |
|---|---|
| New `DiffusionGemmaBackend` class | OpenAI-compatible client for the local DiffusionGemma server |
| `CerebrasBackend` refactored → `OpenAICompatibleBackend` base | Lets `CerebrasBackend` and `DiffusionGemmaBackend` share the HTTP path via `_default_base_url()` / `_default_model()` hooks |
| Router routes code requests to DiffusionGemma | The actual feature; covered by `test_e2e.py` |
| `Message.content` is `str` only | DiffusionGemma endpoint does not accept multimodal content lists (no `image_url` parts) |
| DiffusionGemma responses omit `time_info` | `time_info` is Cerebras-specific; `_parse_response` handles its absence |

## Quick start

Install from this directory (note: the package name is still `cfire` — see [Caveats](#caveats)):

```bash
cd cfire-diffusion
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Plain Cerebras call (same API as parent `cfire`):

```python
from cfire import AsyncCfire, ChatRequest, Message

async with AsyncCfire() as client:
    resp = await client.complete(ChatRequest(
        model="gemma-4-31b",
        messages=[Message(role="user", content="Explain async I/O.")],
    ))
    print(resp.choices[0].message.content)
    print(resp.time_info)  # queue/prompt/completion/total — Cerebras only
```

## Routing code to DiffusionGemma

This is the variant's reason for existing:

```python
from cfire import AsyncCfire, ChatRequest, Message, RoutingPolicy
from cfire.backends import CerebrasBackend, DiffusionGemmaBackend
from cfire.router import Router

client = AsyncCfire(backend=Router(
    primary=CerebrasBackend(),
    fallbacks=[
        DiffusionGemmaBackend(
            base_url="https://api.cerebras.ai/v1",
            model="nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        ),
    ],
    policy=RoutingPolicy(
        prefer_local_for=[r"^```", r"^(def|class|function|import)\b", r"translate.*to\s+\w+$"],
    ),
))
```

Requests matching `prefer_local_for` go to DiffusionGemma first; everything else hits Cerebras. If DiffusionGemma is unreachable, the router falls through to Cerebras transparently.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CFIRE_DIFFUSIONGEMMA_BASE_URL` | `https://api.cerebras.ai/v1` | DiffusionGemma OpenAI-compatible server URL |
| `CFIRE_DIFFUSIONGEMMA_MODEL` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | DiffusionGemma model ID |
| `CFIRE_DIFFUSIONGEMMA_API_KEY` | `""` | Optional API key if your DiffusionGemma server requires one |
| `CFIRE_CEREBRAS_BASE_URL` | `https://api.cerebras.ai/v1` | Cerebras endpoint |
| `CFIRE_API_KEY` or `CEREBRAS_API_KEY` | — | Cerebras API key (required for primary backend) |

Plus all the parent `cfire` env vars — see [parent README](../README.md#environment-variables).

## When to use which backend

| Workload | Backend | Why |
|---|---|---|
| Code completion, structured edits, regeneration-heavy loops | DiffusionGemma | Diffusion models shine on narrow target distributions; no queue, free |
| Open-ended chat, reasoning, long-form generation | Cerebras `gemma-4-31b` | ~1,500 TPS, mature telemetry, broader capability |
| Multimodal (image + text) | **Not supported here** | `Message.content` is `str` only in this variant — use parent `cfire` instead |
| Anything needing `time_info` | Cerebras | DiffusionGemma responses omit `time_info` |

## Tests

```bash
cd cfire-diffusion
pytest cfire/tests -q
```

Must run from this directory because of import-path overlap with the parent `cfire/` package. Includes `test_e2e.py` which verifies the router dispatches code-shaped requests to DiffusionGemma.

## Caveats

- **Package name collision.** Both this variant and the parent declare `name = "cfire"` in `pyproject.toml`. They cannot coexist in the same environment — uninstall one before installing the other, or use separate venvs.
- **Self-hosted DiffusionGemma.** The default DiffusionGemma URL is the Cerebras API. To use your own server (e.g. a local vLLM instance), point `CFIRE_DIFFUSIONGEMMA_BASE_URL` at it.
- **No multimodal.** `Message.content` is `str` only. Use the parent `cfire/` package if you need image + text messages (as `vision_ops` does).

## Related

- [Parent `cfire` README](../README.md) — main project, multimodal support, full feature set
- [Cerebras Speed Guide](../CEREBRAS_SPEED_GUIDE.md) — applies to the Cerebras primary backend here
- [Project architecture](../README.md#architecture) — how this variant fits in

## Status

**Alpha, 0.2.0.** Forked from `cfire` 0.1.0 to add DiffusionGemma routing. The parent `cfire` package is the actively-recommended default for most users; this variant is for the specific code-routing use case.
