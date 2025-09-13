CI/CD Overview

- GitHub Actions workflow `.github/workflows/ci.yml` runs on push/PR.
- Uses Python 3.11 on Ubuntu and installs minimal deps for tests (excludes vLLM).
- Installs CPU wheels for torch/torchaudio and `ffmpeg` for `pydub`.
- Runs tests via `scripts/test.sh`.

Running Locally

- Install minimal deps (no vLLM): `./scripts/ci-install.sh`.
- Run tests: `./scripts/test.sh`.
- To run full tests without exclusions: `OMOAI_CI_RUN_FULL=1 ./scripts/test.sh`.

Notes

- The test `tests/test_asr_config_fix.py::test_asr_script_with_valid_config` requires a local ChunkFormer checkpoint.
  - The default CI run excludes it.
  - To enable it in CI, dispatch the workflow with input `run_full: true` and ensure models are available in `models/chunkformer/chunkformer-large-vie/`.

