"""Batch API wrapper for cfire.

Cerebras exposes POST /v1/batch accepting up to 200MB payloads where each
line is a standard chat completion request. Useful for offline workloads
(bulk evaluation, dataset annotation) where streaming + per-request HTTP
overhead would dominate.

Phase 3 will implement:
- submit_batch(requests: Iterable[ChatRequest]) -> BatchJob
- await_batch(job: BatchJob, poll_interval: float = 5.0) -> list[ChatResponse]
- list_batches() / cancel_batch(job_id)
"""

# Phase 3 implementation pending.
