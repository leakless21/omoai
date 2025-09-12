"""
In-memory postprocessing removed.

This module previously implemented vLLM-based in-memory postprocessing.
To prevent any accidental in-process vLLM usage, the implementation was removed
and replaced by a clear ImportError encouraging the script-based pipeline.

Use:
  - scripts/post.py (script)
  - src/omoai/api/scripts/postprocess_wrapper.run_postprocess_script
  - src/omoai/api/services._postprocess_script
to run post-processing via subprocess/script wrappers.
"""
raise ImportError(
    "In-memory postprocessing has been removed. "
    "Use the script-based pipeline (scripts/post.py) via "
    "src.omoai.api.scripts.postprocess_wrapper.run_postprocess_script "
    "or src.omoai.api.services._postprocess_script."
)