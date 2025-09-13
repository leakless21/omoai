Place large or real audio fixtures here for opt-in tests.

Recommended filename: `testaudio.mp3`

Tests that can use it:
- API integration: prefers `tests/assets/testaudio.mp3` when posting to `/pipeline`.
- Benchmark (marked `@slow`): skipped unless this file exists or `OMOAI_TEST_MP3` is set.

Alternative: set env var `OMOAI_TEST_MP3` to an absolute path.

Note: Real end-to-end heavy processing is stubbed by default for speed and determinism. The
audio file is used to exercise request handling and payload size paths without incurring
model inference time.

