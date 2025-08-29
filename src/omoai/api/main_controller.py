from typing import Annotated, Optional, List, Literal
from litestar import Controller, post, get
from litestar.params import Body
from litestar.enums import RequestEncodingType
from litestar.response import Redirect, Response
import logging
from omoai.api.models import PipelineRequest, PipelineResponse, OutputFormatParams
from omoai.api.services_enhanced import run_full_pipeline as run_full_pipeline_enhanced

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
    ) -> PipelineResponse | Response[str]:
        """
        Endpoint to run the entire audio processing pipeline.

        This endpoint executes the full workflow:
        1. Preprocess: Take the uploaded audio file and preprocess it.
        2. ASR: Pass the preprocessed file path to the ASR logic to get the raw transcript and segments.
        3. Post-process: Pass the ASR output to the post-processing logic to get the final punctuated transcript and summary.

        Query Parameters (optional):
        - formats: List of output formats (json, text, srt, vtt, md)
        - include: What to include (transcript_raw, transcript_punct, segments)
        - ts: Timestamp format (none, s, ms, clock)
        - summary: Summary type (bullets, abstract, both, none)
        - summary_bullets_max: Maximum number of bullet points
        - summary_lang: Summary language

        Example: GET /pipeline?include=segments&ts=clock&summary=bullets
        """
        # Create OutputFormatParams object from query parameters
        output_params = OutputFormatParams(
            formats=formats,
            include=include,
            ts=ts,
            summary=summary,
            summary_bullets_max=summary_bullets_max,
            summary_lang=summary_lang
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
            # Return structured JSON response by default without raw transcript
            # Only include punctuated transcript and summary by default
            return PipelineResponse(
                summary=result.summary,
                segments=result.segments,
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

        # Otherwise, return structured JSON response but exclude raw transcript unless explicitly requested
        # Check if transcript_raw is explicitly requested in include parameter
        include_raw = include and "transcript_raw" in include
        
        if include_raw:
            # If raw transcript is explicitly requested, return full result
            return result
        else:
            # Exclude raw transcript from response
            return PipelineResponse(
                summary=result.summary,
                segments=result.segments,
                transcript_punct=result.transcript_punct
            )