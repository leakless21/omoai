import logging
from typing import Annotated, Literal

from litestar import Controller, get, post
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import Redirect, Response

from omoai.api.models import OutputFormatParams, PipelineRequest, PipelineResponse
from omoai.config.schemas import get_config
from omoai.api.services import run_full_pipeline

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
        data: Annotated[
            PipelineRequest, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        # Query parameters for output formatting
        formats: list[Literal["json", "text", "srt", "vtt", "md"]] | None = None,
        include: list[Literal["transcript_raw", "transcript_punct", "segments"]]
        | None = None,
        ts: Literal["none", "s", "ms", "clock"] | None = None,
        summary: Literal["bullets", "abstract", "both", "none"] | None = None,
        summary_bullets_max: int | None = None,
        summary_lang: str | None = None,
        return_summary_raw: bool | None = None,
        # Quality metrics and diff options
        include_quality_metrics: bool | None = None,
        include_diffs: bool | None = None,
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
            include_diffs=include_diffs,
            return_summary_raw=return_summary_raw,
        )

        # Apply defaults from configuration when not provided via query
        try:
            cfg = get_config()
            out_cfg = getattr(cfg, "output", None)
            sum_cfg = getattr(out_cfg, "summary", None)
            if out_cfg and sum_cfg:
                if output_params.summary is None and getattr(sum_cfg, "mode", None):
                    output_params.summary = sum_cfg.mode  # type: ignore[assignment]
                if (
                    output_params.summary_bullets_max is None
                    and getattr(sum_cfg, "bullets_max", None) is not None
                ):
                    output_params.summary_bullets_max = int(sum_cfg.bullets_max)  # type: ignore[assignment]
                if (
                    output_params.summary_lang is None
                    and getattr(sum_cfg, "language", None)
                ):
                    output_params.summary_lang = str(sum_cfg.language)  # type: ignore[assignment]
        except Exception:
            pass

        # Debug logging to validate query parameter parsing
        logger.info(
            f"Received query parameters - summary: {summary}, summary_bullets_max: {summary_bullets_max}, include: {include}"
        )
        logger.info(f"Created output_params: {output_params}")
        if output_params:
            logger.info(f"output_params.summary: {output_params.summary}")
            logger.info(
                f"output_params.summary_bullets_max: {output_params.summary_bullets_max}"
            )
            logger.info(f"output_params.include: {output_params.include}")

        # Run pipeline using the enhanced, high-performance service
        # Pass the OutputFormatParams object directly so services receive a typed object
        params = output_params if output_params else None

        logger.info("Starting pipeline execution with enhanced service")

        try:
            result = await run_full_pipeline(data, params)
            logger.info("Pipeline execution completed successfully")
        except Exception as e:
            logger.error(f"Pipeline execution failed with error: {e!s}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

        logger.info("Pipeline execution completed, preparing response")
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result attributes: {dir(result)}")

        # The summarization returns a structured dictionary with keys: title, summary, points
        summary_data = getattr(result, "summary", {}) or {}
        logger.info(f"Summary data type: {type(summary_data)}")
        logger.info(f"Summary data content: {summary_data}")
        logger.info(
            f"Summary data keys: {list(summary_data.keys()) if isinstance(summary_data, dict) else 'N/A'}"
        )
        if not isinstance(summary_data, dict):
            logger.error("Pipeline returned unexpected summary type; expected dict.")
            raise ValueError(
                "Pipeline returned unexpected summary type; expected dict with keys 'title','summary','points'."
            )
        # Extract commonly-used pieces
        title = summary_data.get("title", "")
        abstract_text = summary_data.get("summary", "")
        points = summary_data.get("points", [])
        logger.info(f"Extracted summary title: {title[:100] if title else 'None'}")
        logger.info(
            f"Extracted summary abstract: {abstract_text[:100] if abstract_text else 'None'}"
        )

        logger.info("Starting response preparation")
        # Default: respond with JSON but exclude raw transcript unless explicitly requested
        no_query_params = (
            formats is None
            and include is None
            and ts is None
            and summary is None
            and summary_bullets_max is None
            and summary_lang is None
        )

        logger.info("Determining response format based on query parameters")
        logger.info(f"no_query_params: {no_query_params}")
        logger.info(f"formats: {formats}")
        logger.info(f"include: {include}")

        try:
            if no_query_params:
                logger.info("Returning default JSON response")
                # Return JSON response with structured summary dict and punctuated transcript
                response_obj = PipelineResponse(
                    summary=summary_data,
                    segments=[],  # Exclude segments by default
                    transcript_punct=result.transcript_punct,
                    transcript_raw=getattr(result, "transcript_raw", None),
                    # Pass through optional fields when requested
                    quality_metrics=(
                        getattr(result, "quality_metrics", None)
                        if include_quality_metrics
                        else None
                    ),
                    diffs=(
                        getattr(result, "diffs", None)
                        if include_diffs
                        else None
                    ),
                    summary_raw_text=(
                        getattr(result, "summary_raw_text", None)
                        if return_summary_raw
                        else None
                    ),
                )
                logger.info(
                    f"PipelineResponse created successfully: {type(response_obj)}"
                )
                return response_obj

            # For backward compatibility, handle text/plain response when specifically requested
            if formats == ["text"] or (
                formats and "text" in formats and len(formats) == 1
            ):
                logger.info("Returning text/plain response")
                # If raw summary requested and available, return only the raw LLM output
                if return_summary_raw and getattr(result, "summary_raw_text", None):
                    raw_text = result.summary_raw_text or ""
                    logger.info(f"Raw text response length: {len(raw_text)} characters")
                    return Response(raw_text, media_type="text/plain; charset=utf-8")
                # Otherwise compose plain text with transcript and structured summary (title, abstract, points)
                parts: list[str] = []
                if result.transcript_punct:
                    parts.append(result.transcript_punct)
                if title:
                    parts.append(f"\n\n# {title}\n")
                if abstract_text:
                    parts.append(str(abstract_text))
                if points:
                    parts.append("\n\n# Points\n")
                    parts.extend([f"- {p}" for p in points])
                text_response = "\n".join(parts)
                logger.info(f"Text response length: {len(text_response)} characters")
                return Response(text_response, media_type="text/plain; charset=utf-8")

            # Otherwise, return structured JSON response based on include parameters
            logger.info("Returning structured JSON response")
            # Check what components are explicitly requested
            include_raw = include and "transcript_raw" in include
            include_segments = include and "segments" in include
            include_punct = include and "transcript_punct" in include

            logger.info(f"include_raw: {include_raw}")
            logger.info(f"include_segments: {include_segments}")
            logger.info(f"include_punct: {include_punct}")

            # Build response based on explicit include parameters
            response_data = {
                "summary": summary_data,
                "segments": result.segments if include_segments else [],
                "transcript_punct": result.transcript_punct
                if (include_punct or include is None)
                else None,
            }

            # Add raw transcript only if explicitly requested
            if include_raw:
                response_data["transcript_raw"] = getattr(result, "transcript_raw", "")

            # Forward optional fields when requested
            if include_quality_metrics:
                response_data["quality_metrics"] = getattr(
                    result, "quality_metrics", None
                )
            if include_diffs:
                response_data["diffs"] = getattr(result, "diffs", None)
            if return_summary_raw:
                response_data["summary_raw_text"] = getattr(
                    result, "summary_raw_text", None
                )

            logger.info("Creating PipelineResponse with response_data")
            response_obj = PipelineResponse(**response_data)
            logger.info(f"PipelineResponse created successfully: {type(response_obj)}")
            return response_obj

        except Exception as e:
            logger.error(f"Error creating response: {e!s}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
