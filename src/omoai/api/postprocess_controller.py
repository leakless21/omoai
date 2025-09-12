import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from litestar import Controller, post
from litestar.datastructures import State

from omoai.api.models import PostprocessRequest, PostprocessResponse, OutputFormatParams


from omoai.api.services import postprocess_service


class PostprocessController(Controller):
    path = "/postprocess"

    @post()
    async def postprocess(
        self,
        data: PostprocessRequest,
        state: State,
        output_params: Optional[OutputFormatParams] = None
    ) -> PostprocessResponse:
        """Process post-processing request with optional output formatting.

        Query Parameters (optional):
        - include: What to include (transcript_raw, transcript_punct, segments)
        - ts: Timestamp format (none, s, ms, clock)
        - summary: Summary type (bullets, abstract, both, none)
        - summary_bullets_max: Maximum number of bullet points
        - summary_lang: Summary language
        - include_quality_metrics: Include quality metrics in response
        - include_diffs: Include human-readable diffs in response
        """
        # Process with optional quality metrics and diffs
        # Convert OutputFormatParams to dict if provided
        output_params_dict = None
        if output_params:
            output_params_dict = {
                "include_quality_metrics": output_params.include_quality_metrics,
                "include_diffs": output_params.include_diffs,
                "include": output_params.include,
                "ts": output_params.ts,
                "summary": output_params.summary,
                "summary_bullets_max": output_params.summary_bullets_max,
                "summary_lang": output_params.summary_lang,
                "return_summary_raw": output_params.return_summary_raw,
            }
        
        result = await postprocess_service(data, output_params_dict)
        
        return result
