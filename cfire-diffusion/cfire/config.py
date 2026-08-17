"""Configuration for cfire.

Reads environment with NO embedded fallback secrets — this is the fix for
the hardcoded-API-key smell at cerebras_race_client.py:53-56 where a real
key was committed to the repo in three files.

Env vars:
  Required for CerebrasBackend:
    CEREBRAS_API_KEY            (no default — raises ConfigError if missing)

  Optional:
    CFIRE_CEREBRAS_BASE_URL     default https://api.cerebras.ai/v1
    CFIRE_LOCAL_BASE_URL        default http://127.0.0.1:8123
    CFIRE_CDN_BASE_URL          no default (CDNBackend disabled if unset)
    CFIRE_REDIS_URL             no default (RedisCache disabled if unset)
    CFIRE_MODEL                 default gemma-4-31b
    CFIRE_CONCURRENCY           default 16 (benchmark-measured sweet spot)
    CFIRE_REQ_PER_MIN           default 1000 (Developer tier)
    CFIRE_TOK_PER_MIN           default 1_000_000 (Developer tier)
    CFIRE_COMPRESS_THRESHOLD    default 4096 bytes
    CFIRE_CACHE_MAXSIZE         default 1024 entries
    CFIRE_CACHE_TTL             default 3600 seconds
    CFIRE_CB_THRESHOLD          default 5 consecutive failures
    CFIRE_CB_COOLDOWN           default 30.0 seconds
"""

from __future__ import annotations

import os

from .exceptions import ConfigError

# --- Endpoint defaults --------------------------------------------------

CEREBRAS_BASE_URL = os.environ.get("CFIRE_CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
LOCAL_BASE_URL = os.environ.get("CFIRE_LOCAL_BASE_URL", "http://127.0.0.1:8123")
CDN_BASE_URL: str | None = os.environ.get("CFIRE_CDN_BASE_URL")
REDIS_URL: str | None = os.environ.get("CFIRE_REDIS_URL")

# --- API key -------------------------------------------------------------

def get_api_key(env_var: str = "CEREBRAS_API_KEY") -> str:
    """Read the Cerebras API key from env. Raises ConfigError if missing.

    No embedded fallback. Mirrors how cerebras_cloud_sdk treats the key.
    """
    v = os.environ.get(env_var)
    if not v:
        raise ConfigError(
            f"{env_var} not set. Get a key at https://cloud.cerebras.ai "
            f"and export {env_var}=csk-..."
        )
    return v

# --- DiffusionGemma defaults --------------------------------------------

DIFFUSIONGEMMA_BASE_URL = os.environ.get(
    "CFIRE_DIFFUSIONGEMMA_BASE_URL", "https://api.cerebras.ai/v1"
)
DIFFUSIONGEMMA_MODEL = os.environ.get(
    "CFIRE_DIFFUSIONGEMMA_MODEL", "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
)
DIFFUSIONGEMMA_API_KEY = os.environ.get("CFIRE_DIFFUSIONGEMMA_API_KEY", "")

# --- Defaults ------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("CFIRE_MODEL", "gemma-4-31b")
DEFAULT_CONCURRENCY = int(os.environ.get("CFIRE_CONCURRENCY", "16"))
DEFAULT_REQ_PER_MIN = float(os.environ.get("CFIRE_REQ_PER_MIN", "1000"))
DEFAULT_TOK_PER_MIN = float(os.environ.get("CFIRE_TOK_PER_MIN", "1000000"))
COMPRESS_THRESHOLD_BYTES = int(os.environ.get("CFIRE_COMPRESS_THRESHOLD", "4096"))
CACHE_MAXSIZE = int(os.environ.get("CFIRE_CACHE_MAXSIZE", "1024"))
CACHE_TTL_SEC = float(os.environ.get("CFIRE_CACHE_TTL", "3600"))
CIRCUIT_THRESHOLD = int(os.environ.get("CFIRE_CB_THRESHOLD", "5"))
CIRCUIT_COOLDOWN = float(os.environ.get("CFIRE_CB_COOLDOWN", "30.0"))


__all__ = [
    "CEREBRAS_BASE_URL",
    "LOCAL_BASE_URL",
    "CDN_BASE_URL",
    "REDIS_URL",
    "DIFFUSIONGEMMA_BASE_URL",
    "DIFFUSIONGEMMA_MODEL",
    "DIFFUSIONGEMMA_API_KEY",
    "get_api_key",
    "DEFAULT_MODEL",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_REQ_PER_MIN",
    "DEFAULT_TOK_PER_MIN",
    "COMPRESS_THRESHOLD_BYTES",
    "CACHE_MAXSIZE",
    "CACHE_TTL_SEC",
    "CIRCUIT_THRESHOLD",
    "CIRCUIT_COOLDOWN",
]
