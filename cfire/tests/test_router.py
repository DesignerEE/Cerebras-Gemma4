"""Tests for cfire.router.

Phase 3 targets:
- Router stays on primary when it succeeds
- Router falls over to fallback on RateLimitError
- Router falls over to fallback on TimeoutError
- Router respects prefer_local_for regex match
- Router honors max_retries_per_backend before giving up
- Router re-raises the most recent error when all backends exhausted
"""

import pytest


@pytest.mark.xfail(reason="Phase 3 implementation", strict=True)
def test_phase3_placeholder():
    """Removed once Phase 3 lands real tests."""
    assert False
