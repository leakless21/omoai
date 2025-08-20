from typing import Annotated
from litestar import Controller, post
from litestar.params import Body
from litestar.enums import RequestEncodingType
from src.omoai.api.models import PipelineRequest, PipelineResponse
from src.omoai.api.services import run_full_pipeline


class MainController(Controller):
    """
    MainController handles the main API endpoint for running the entire audio processing pipeline.
    """

    @post("/pipeline")
    async def pipeline(
        self, data: Annotated[PipelineRequest, Body(media_type=RequestEncodingType.MULTI_PART)]
    ) -> PipelineResponse:
        """
        Endpoint to run the entire audio processing pipeline.

        This endpoint executes the full workflow:
        1. Preprocess: Take the uploaded audio file and preprocess it.
        2. ASR: Pass the preprocessed file path to the ASR logic to get the raw transcript and segments.
        3. Post-process: Pass the ASR output to the post-processing logic to get the final punctuated transcript and summary.
        """
        return await run_full_pipeline(data)