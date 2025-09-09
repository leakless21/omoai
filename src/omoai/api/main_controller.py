from typing import Annotated, Optional, List, Literal
from litestar import Controller, post, get
from litestar.params import Body
from litestar.enums import RequestEncodingType
from litestar.response import Redirect, Response
import logging
from pathlib import Path
from datetime import datetime
from omoai.api.models import PipelineRequest, PipelineResponse, OutputFormatParams
from omoai.api.services_enhanced import run_full_pipeline as run_full_pipeline_enhanced
from omoai.config import get_config
from omoai.output.writer import write_outputs

# Set up logger
logger = logging.getLogger(__name__)


class MainController(Controller):
    """
    MainController handles the main API endpoint for running the entire audio processing pipeline.
    """

    @get("/")
    async def root(self) -> Redirect:
        """
        Root endpoint that redirects to the OpenAPI schema documentation.
        
        This provides a convenient way for users to access the API documentation
        when they visit the root URL of the application.
        """
        return Redirect(path="/schema")

    @post("/pipeline")
    async def pipeline(
        self,
        data: Annotated[PipelineRequest, Body(media_type=RequestEncodingType.MULTI_PART)],
        # Query parameters for output formatting
        formats: Optional[List[Literal["json", "text", "srt", "vtt", "md"]]] = None,
        include: Optional[List[Literal["transcript_raw", "transcript_punct", "segments"]]] = None,
        ts: Optional[Literal["none", "s", "ms", "clock"]] = None,
        summary: Optional[Literal["bullets", "abstract", "both", "none"]] = None,
        summary_bullets_max: Optional[int] = None,
        summary_lang: Optional[str] = None,
        # Quality metrics and diff options
        include_quality_metrics: Optional[bool] = None,
        include_diffs: Optional[bool] = None,
    ) -> PipelineResponse | Response[str]:
        """
        Endpoint to run the entire audio processing pipeline.

        This endpoint executes the full workflow:
        1. Preprocess: Take the uploaded audio file and preprocess it.
        2. ASR: Pass the preprocessed file path to the ASR logic to get the raw transcript and segments.
        3. Post-process: Pass the ASR output to the post-processing logic to get the final punctuated transcript and summary.

        Default Response (no query parameters):
        - transcript_punct: Punctuated transcript text
        - summary.bullets: Bullet point summary
        - summary.abstract: Abstract summary
        - segments: Empty array (excluded by default)

        Query Parameters (optional):
        - formats: List of output formats (json, text, srt, vtt, md)
        - include: What to include (transcript_raw, transcript_punct, segments)
        - ts: Timestamp format (none, s, ms, clock)
        - summary: Summary type (both=default, bullets, abstract, none)
        - summary_bullets_max: Maximum number of bullet points
        - summary_lang: Summary language

        Examples:
        - Default: POST /pipeline (returns bullets + abstract + punctuated transcript)
        - With segments: POST /pipeline?include=segments
        - Bullets only: POST /pipeline?summary=bullets
        - Full response: POST /pipeline?include=transcript_raw,segments
        """
        # Create OutputFormatParams object from query parameters
        output_params = OutputFormatParams(
            formats=formats,
            include=include,
            ts=ts,
            summary=summary,
            summary_bullets_max=summary_bullets_max,
            summary_lang=summary_lang,
            include_quality_metrics=include_quality_metrics,
            include_diffs=include_diffs
        )
        
        # Debug logging to validate query parameter parsing
        logger.info(f"Received query parameters - summary: {summary}, summary_bullets_max: {summary_bullets_max}, include: {include}")
        logger.info(f"Created output_params: {output_params}")
        if output_params:
            logger.info(f"output_params.summary: {output_params.summary}")
            logger.info(f"output_params.summary_bullets_max: {output_params.summary_bullets_max}")
            logger.info(f"output_params.include: {output_params.include}")
        
        # Run pipeline using the enhanced, high-performance service
        params = output_params.dict(exclude_none=True) if output_params else None
        result = await run_full_pipeline_enhanced(data, params)

        # Persist outputs to disk for API requests when configured in config.output.save_on_api
        try:
            cfg = get_config()
            if getattr(cfg.output, "save_on_api", False):
                api_out = getattr(cfg.output, "api_output_dir", None)
                base_output_dir = Path(api_out) if api_out else cfg.paths.out_dir

                # Create a unique timestamped subdirectory per API request
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                unique_output_dir = Path(base_output_dir) / timestamp
                try:
                    unique_output_dir.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    # Extremely unlikely due to microsecond precision, but handle just in case
                    counter = 1
                    while unique_output_dir.exists():
                        unique_output_dir = Path(base_output_dir) / f"{timestamp}_{counter}"
                        counter += 1
                    unique_output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    written = write_outputs(
                        unique_output_dir,
                        result.segments,
                        getattr(result, "transcript_raw", ""),
                        getattr(result, "transcript_punct", ""),
                        result.summary or {},
                        getattr(result, "metadata", {}),
                        cfg.output,
                    )
                    logger.info(f"Saved API outputs to {unique_output_dir}: {written}")
                except Exception as e:
                    logger.exception(f"Failed to write API outputs to {unique_output_dir}: {e}")
        except Exception:
            # Best-effort only — do not fail the API because persistence failed
            logger.debug("Skipping API output persistence due to configuration/read error", exc_info=True)

        # Default: respond with JSON but exclude raw transcript unless explicitly requested
        no_query_params = (
            formats is None
            and include is None
            and ts is None
            and summary is None
            and summary_bullets_max is None
            and summary_lang is None
        )

        if no_query_params:
            # Return structured JSON response by default
            # Include punctuated transcript and full summary (bullets + abstract, no segments)
            summary_default = {}
            if result.summary and isinstance(result.summary, dict):
                # Include both bullets and abstract by default
                if "bullets" in result.summary:
                    summary_default["bullets"] = result.summary["bullets"]
                if "abstract" in result.summary:
                    summary_default["abstract"] = result.summary["abstract"]
            
            return PipelineResponse(
                summary=summary_default,
                segments=[],  # Exclude segments by default
                transcript_punct=result.transcript_punct
            )

        # For backward compatibility, handle text/plain response when specifically requested
        if formats == ["text"] or (formats and "text" in formats and len(formats) == 1):
            # Compose plain text with transcript and summary (bullets + abstract)
            parts: List[str] = []
            if result.transcript_punct:
                parts.append(result.transcript_punct)
            # Append summary if available
            if result.summary:
                bullets = result.summary.get("bullets") if isinstance(result.summary, dict) else None
                abstract = result.summary.get("abstract") if isinstance(result.summary, dict) else None
                lines: List[str] = []
                if bullets:
                    lines.append("\n\n# Summary Points\n")
                    for b in bullets:
                        lines.append(f"- {b}")
                if abstract:
                    # Add header separating abstract if bullets also present
                    if bullets:
                        lines.append("\n# Abstract\n")
                    else:
                        lines.append("\n\n# Abstract\n")
                    lines.append(str(abstract))
                if lines:
                    parts.append("\n".join(lines))
            return Response("\n".join(parts), media_type="text/plain; charset=utf-8")

        # Otherwise, return structured JSON response based on include parameters
        # Check what components are explicitly requested
        include_raw = include and "transcript_raw" in include
        include_segments = include and "segments" in include
        include_punct = include and "transcript_punct" in include
        
        # Determine what to include in summary based on summary parameter
        summary_to_include = {}
        if result.summary and isinstance(result.summary, dict):
            if summary is None or summary == "both":  # Default to both bullets and abstract
                summary_to_include = result.summary
            elif summary == "bullets":
                if "bullets" in result.summary:
                    summary_to_include = {"bullets": result.summary["bullets"]}
            elif summary == "abstract":
                if "abstract" in result.summary:
                    summary_to_include = {"abstract": result.summary["abstract"]}
            elif summary == "none":
                summary_to_include = {}
        
        # Build response based on explicit include parameters
        response_data = {
            "summary": summary_to_include,
            "segments": result.segments if include_segments else [],
            "transcript_punct": result.transcript_punct if (include_punct or include is None) else None
        }
        
        # Add raw transcript only if explicitly requested
        if include_raw:
            response_data["transcript_raw"] = getattr(result, "transcript_raw", "")
        
        return PipelineResponse(**{k: v for k, v in response_data.items() if v is not None})