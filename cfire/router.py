"""Small local router for cfire.

Policy-driven, not heuristic-heavy. No ML routing — deterministic rules.
Satisfies the "small local router for fallback" constraint.

Decision order in Router.complete(request):
  1. If request matches any prefer_local_for pattern → start with fallbacks
  2. Else start with primary (default: Cerebras)
  3. On failover_on exception, advance to next backend
  4. Honor max_retries_per_backend before giving up on a single backend
  5. If every backend exhausted, re-raise the most recent error

Phase 3 will implement:
- RoutingPolicy  — pydantic model with failover_on, prefer_local_for,
                   max_retries_per_backend, cerebras_first
- Router         — holds (primary, fallbacks, policy); exposes async
                   complete() / stream() that conform to the Backend
                   Protocol so Router can itself be wrapped.
"""

# Phase 3 implementation pending.
