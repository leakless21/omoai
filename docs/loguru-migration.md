# Migration Plan: Move OMOAI Logging to Loguru

This document describes a clean, config‑driven migration from Python's stdlib
`logging` to Loguru across the OMOAI project. The plan keeps a single logging
system under `src/omoai/logging_system`, keeps behavior configurable via
`config.yaml`, and preserves existing helper APIs while simplifying setup.

## Objectives

- Single logging system (no duplicate modules) under `logging_system`.
- Configurable via `config.yaml` (top‑level `logging:` block) with env overrides.
- Human‑friendly console logs + JSONL file logs with rotation/retention/compression.
- Capture stdlib/third‑party logs (uvicorn, litestar, transformers, torch, etc.).
- Preserve helper APIs (performance/error logging, decorators) with Loguru
  underneath; minimize churn in call sites.

## Current State (Summary)

- Stdlib logging is used throughout; `logging_system` sets handlers/formatters.
- API previously had its own logging config (now removed); app calls
  `setup_logging()` and passes `logging_config=None` to Litestar.
- `config.yaml` now supports a top‑level `logging:` block (see below).
- `@logs/...` alias support is implemented and normalized to `./logs/...`.

## Design Decisions

- Use `loguru.logger` as the core logger; configure sinks once per process.
- Intercept stdlib logging with an `InterceptHandler` attached to the root
  logger so all third‑party logs flow into Loguru (no duplicate lines).
- Console sink is colored and readable; file sink is JSON (`serialize: true`),
  UTC timestamps by default.
- Enable async logging (`enqueue: true`) for resiliency under load.
- Use `logger.bind(...)` to add structured context (e.g., `request_id`,
  `component`, `operation`).
- Implement the migration inside `logging_system` (no new module); keep public
  helper function names stable for minimal code changes.

## Config Schema (config.yaml → Loguru)

Top‑level `logging:` block in `config.yaml` controls Loguru setup. These fields
already exist in `src/omoai/config/schemas.py` as `LoggingSettings` and will map
to Loguru sink options as follows:

- `level`: base level (DEBUG/INFO/WARNING/ERROR). Overridden by `debug_mode` or
  `quiet_mode`.
- `enable_console`: add/remove stdout sink.
- `enable_file`: add/remove file JSON sink.
- `log_file`: path to log file (supports `@logs/...` alias → `./logs/...`).
- `max_file_size`, `backup_count`: used as default rotation/retention if the
  advanced fields below are not set.
- `rotation`: Loguru rotation policy (e.g., `"10 MB"`, `"00:00"`, or callable).
- `retention`: retention policy (e.g., `"14 days"`).
- `compression`: compression for rotated files (e.g., `"gz"`).
- `enqueue`: enable async logging.
- `debug_mode`: force DEBUG level.
- `quiet_mode`: force ERROR level.

Example config:

```yaml
logging:
  level: INFO
  enable_console: true
  enable_file: true
  log_file: "@logs/api_server.jsonl"
  max_file_size: 10485760   # 10 MB
  backup_count: 5
  # Advanced (Loguru-native)
  rotation: "10 MB"
  retention: "14 days"
  compression: "gz"
  enqueue: true
  debug_mode: false
  quiet_mode: false
```

Environment overrides (kept for compatibility): `OMOAI_LOG_*` continue to work
and are merged over `config.yaml`.

## Migration Steps

1) Dependency
- Add `loguru` to `pyproject.toml` dependencies.

2) Implement Loguru setup inside `logging_system`
- File: `src/omoai/logging_system/logger.py`
  - Update `setup_logging()` to:
    - Read merged `LoggingSettings` from `config.yaml` and env.
    - Remove existing stdlib handlers on root logger.
    - Attach a single `InterceptHandler` to route stdlib `logging` to Loguru.
    - Configure Loguru sinks:
      - Console: human‑friendly format (colored, concise).
      - File: JSON lines with `serialize=True`, `rotation`, `retention`,
        `compression`, `enqueue`, UTC timestamps.
    - Optionally enable `logging.captureWarnings(True)`.
  - Keep public APIs stable and reimplement with Loguru:
    - `get_logger(name)` → return `logger.bind(name=name)`.
    - `log_performance(...)` → use `logger.bind(...).log(level, msg)`.
    - `log_error(...)` → use `logger.opt(exception=True).error(msg)` and bind
      `error_type`, `error_code`, etc.
    - Decorators (`performance_context`, `with_request_context`) → bind context
      into logger; ensure context is included in outputs.

3) Third‑party capture and noise control
- Ensure `uvicorn`, `litestar`, `transformers`, `torch`, `pydantic`,
  `urllib3` logs are captured via stdlib interception.
- Set noise levels: WARNING by default; DEBUG when `debug_mode` is true.

4) API and scripts entrypoints
- API (`src/omoai/api/app.py`) already calls `setup_logging()` and passes
  `logging_config=None` to Litestar; no further changes needed.
- Scripts (e.g., `scripts/asr.py`):
  - Remove `logging.basicConfig(...)`.
  - Add early initialization:

```python
from omoai.logging_system.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)
logger.info("script started")
```

5) Context and correlation
- Provide `get_component_logger(component, **ctx)` → `logger.bind(component=..., **ctx)`.
- For requests, add middleware/decorators to bind `request_id`, `path`,
  `client_ip`; these extras should appear in JSON lines and optionally in the
  console format.

6) Security and redaction
- Add a redaction filter for common secret patterns (e.g., bearer tokens, API
  keys). Make it opt‑in via config/env to avoid false positives.

7) Tests
- Add tests under `tests/logging/`:
  - Console sink: capture and assert level/format (smoke test).
  - File JSON sink: write to temp file; parse lines and validate keys (time,
    level, message, extras).
  - Rotation/retention/compression: generate logs to trigger rotation and check
    rotated/compressed files exist.
  - Stdlib interception: `logging.getLogger("uvicorn").info(...)` appears in
    JSON file.
  - Context binding: `logger.bind(request_id=...)` appears in output.
  - Alias expansion: `@logs/test.jsonl` becomes `logs/test.jsonl`.

8) Documentation
- Keep this document in `docs/loguru-migration.md`.
- Add `docs/logging.md` with usage examples, sink descriptions, and production
  tips once migration is complete.

9) Cleanup
- Remove legacy stdlib‑specific formatters/handlers from `logging_system` once
  all entrypoints are migrated and tests pass.

## Best Practices (Guidelines Adopted)

- Log in UTC. Use ISO‑8601 timestamps in JSON logs.
- Console output is concise and human‑friendly; file output is structured JSONL.
- Use async logging (`enqueue: true`) in production or when under load.
- Bind context (`request_id`, `component`, `operation`) instead of embedding in
  message strings.
- Avoid logging secrets. Enable redaction when in doubt.
- Prevent duplicate logs by centralizing handlers and intercepting stdlib once.

## Rollback Strategy

- Add an environment feature flag, e.g., `OMOAI_USE_LOGLIB=stdlib`, to make
  `setup_logging()` short‑circuit to the prior stdlib configuration if needed.
- Document how to toggle the flag for quick recovery during incidents.

## Timeline (Suggested)

- Day 1: Add dependency; implement `setup_logging()` and interception; smoke
  tests for console/file sinks.
- Day 2: Migrate helper APIs to Loguru; add request context middleware; tune
  third‑party logger levels.
- Day 3: Update scripts; add full test coverage; author `docs/logging.md`.
- Day 4: Burn‑in, remove deprecated code, finalize configuration defaults.

---

After implementing the above, all logging (application and dependencies) will
flow through Loguru, be controlled by `config.yaml`, and write cleanly to the
console and rotated JSONL files in `./logs/`.

