# API Simplification & Optimization Plan

This document describes a concrete plan to simplify and optimize the OmoAI API, make
execution event‑loop safe, and migrate summary outputs to bullets‑only. It also covers
operational improvements for reliability, observability, and developer experience.

## Goals

- Simplify and harden the API surface.
- Make execution event‑loop safe and cancellable.
- Enforce bullets‑only summaries (remove “points” everywhere).
- Reduce side effects; improve operability, metrics, and DX.

## Scope

- Public endpoints, services, models, and docs.
- Breaking change: “points” is removed from all responses and code paths.

## High‑Level Plan

1) Lock summary schema to bullets‑only
- Replace “points” with “bullets” across models, services, controllers, and any writers.
- Normalize upstream outputs: convert any “points” to “bullets” and strip “points” before
  returning from the API.
- Update OpenAPI and examples to bullets‑only.

2) Centralize response shaping in services
- Services become the single source of truth for `PipelineResponse` assembly.
- Remove duplicate shaping and logging from controllers; controllers call service and return.

3) Make blocking work event‑loop safe
- Offload all blocking subprocess and file IO via `asyncio.to_thread()` or a process pool.
- Add per‑step subprocess timeouts and cancellation handling (terminate children on cancel).
- Propagate structured errors for consistent client handling.

4) Implement temp directory lifecycle & cleanup
- Use request‑scoped temp folders under `config.api.temp_dir`.
- Cleanup in `finally` when `cleanup_temp_files` is true; keep predictable folder names for
  troubleshooting when disabled.

5) Remove server‑side artifact side effects
- Stop writing artifacts in response to query flags like `formats`.
- Use content negotiation for `text/plain` (`Accept: text/plain`) to return inline content
  without server‑side writes.
- Optionally, add explicit artifact endpoints later if persistence is needed.

6) Version routes and add request IDs
- Move handlers under `/v1/...`.
- Add request ID middleware; include it in responses and structured logs.

7) Standardize error envelope and metrics
- Error schema: `{code, message, details?, trace_id}` with consistent HTTP status codes.
- Add Prometheus metrics (request count/latency/status) and basic stage timings.

8) Update tests and documentation
- Replace all test expectations using `points` with `bullets`.
- Update fixtures/assets, serializers, and OpenAPI examples.
- Revise docs and changelog (explicit breaking‑change callout).

9) Plan rollout and migration guide
- Upgrade notes: “points” removed; clients must read `summary.bullets`.
- Provide before/after payloads and `curl` examples for common flows.

## Bullets‑Only Migration Details

API contract:
- `PipelineResponse.summary` contains exactly: `{title: str, abstract: str, bullets: List[str]}`.
- “points” is removed entirely from models, payloads, writers, and docs.

Ingestion/normalization:
- Inputs are strict: API expects `bullets`. Upstream must emit `bullets`.

File writers (SRT/VTT/MD):
- Markdown generation reads `summary.bullets` only; remove any mapping from `points`.

Controllers/services:
- Replace all reads/writes of `points` with `bullets`.
- Delete compatibility branches that map `points` ↔ `bullets`.

## Simplified Public API (proposed)

- `POST /v1/pipeline`
  - Body: multipart `{audio_file}`; optional query to include extras
    (e.g., `include=segments,quality_metrics,diffs`).
  - Response 200 JSON:

```json
{
  "summary": {
    "title": "...",
    "abstract": "...",
    "bullets": ["...", "..."]
  },
  "transcript_punct": "...",
  "segments": [ ... ],
  "quality_metrics": { ... },
  "diffs": { ... }
}
```

- Text output: set `Accept: text/plain` (or `?format=text`) to receive composed text; no
  server‑side writes.

Optional async mode (future):
- `POST /v1/pipeline?async=true` → `202 Accepted` + `{job_id, status_url}`.
- `GET /v1/jobs/{id}` returns status/result when ready.

## Event‑Loop Safety, Timeouts, and Cleanup

- Offload blocking steps in `_run_full_pipeline_script()` via `asyncio.to_thread()` or a
  process pool.
- Add `timeout` to all `subprocess.run` calls; on timeout/cancel, terminate child processes
  reliably and raise a structured error.
- Create a request‑scoped temp directory and delete it in `finally` when cleanup is enabled.

## Error Handling & Observability

- Error envelope: `{code, message, details?, trace_id}` for all error responses.
- Assign/propagate a request ID (trace_id) via middleware; inject into logs and responses.
- Metrics: request count/latency/status labels; optional stage timings
  (preprocess/asr/postprocess durations).

## Acceptance Criteria

- All endpoints return bullets‑only summaries; “points” never appears in API responses.
- Controllers are thin; services own response shaping exclusively.
- Blocking steps do not block the event loop; per‑step timeouts applied; cancellation cleans up.
- No artifact writes triggered by query flags; text output via content negotiation only.
- Temp files cleaned up when configured; request ID present in logs; metrics exposed.
- Tests updated and passing; docs and OpenAPI examples reflect bullets‑only schema.

## Proposed File Changes (overview)

- `src/omoai/api/models.py`
  - Document bullets‑only summary; optional dedicated `Summary` model.

- `src/omoai/api/services.py`
  - Normalize upstream outputs to bullets; strip points before returning.
  - Centralize `PipelineResponse` assembly; remove file‑write side effects.
  - Offload blocking calls; add subprocess timeouts; request‑scoped temp dirs and cleanup.

- `src/omoai/api/main_controller.py`
  - Thin orchestration; remove duplicate response shaping; consume bullets only.

- `src/omoai/api/app.py`
  - Version routes under `/v1`; request ID middleware; dev vs prod server flags.

- Tests (`tests/`)
  - Replace `points` assertions with `bullets`; update fixtures and text expectations.

- Docs (`docs/`)
  - OpenAPI examples, README snippets, changelog with breaking change notice.

## Rollout & Migration

- Breaking change announcement: bullets‑only; remove any client reliance on `points`.
- Provide before/after examples and a migration checklist for client teams.
- Tag a minor release with deprecations removed; follow up with the metrics and versioning
  improvements as a second pass if needed.

## Phases & Timeline (suggested)

Phase 1 (core, low risk): bullets‑only migration, centralize response shaping (controller now mostly pass‑through except text/plain), event‑loop safety (to_thread + subprocess timeouts), and temp cleanup (request‑scoped dir). Update tests/docs.

Phase 2 (operational): version routes, request IDs, error envelope, metrics.

Phase 3 (optional): async jobs pattern and artifact endpoints if persistence is required.

---

Implementation progress (current):
- Updated core parsing to emit `bullets` instead of `points` (`postprocess_core_utils.py`).
- Normalized summaries in services to bullets‑only and gated metrics/diffs by params.
  - Defaults updated: quality_metrics and diffs are omitted unless explicitly requested via query params.
- Offloaded blocking subprocess calls with `asyncio.to_thread()` and added per‑step timeouts.
- Introduced request‑scoped temp directories with cleanup based on config.
- Simplified controller: returns service result for JSON; still supports `formats=text` for compatibility.
- Updated tests to use `bullets` everywhere and reflect pass‑through behavior.
- Introduced versioned routes under `/v1` (pipeline, health, metrics). Root `/` still redirects to `/schema`.
- Added Request ID middleware and standardized error envelope `{code, message, trace_id, details}`.
- Added lightweight metrics middleware and `/v1/metrics` endpoint with basic counters.
- Added async job pattern:
  - `POST /v1/pipeline?async=true` enqueues job and returns `202` + `{job_id, status_url}`.
  - `GET /v1/jobs/{id}` returns `{status, result?, error?}`.
  - In-memory job manager runs `run_full_pipeline()` in the background; results are stored for retrieval.

Upcoming (if needed): explicit artifact endpoints that serve files only when configured to persist.
