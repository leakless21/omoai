from typing import Annotated
from litestar import Controller, post
from litestar.params import Body
from litestar.enums import RequestEncodingType
from omoai.api.models import PreprocessRequest, PreprocessResponse
from omoai.api.services import preprocess_audio_service


class PreprocessController(Controller):
    path = "/preprocess"

    @post()
    async def preprocess(
        self, data: Annotated[PreprocessRequest, Body(media_type=RequestEncodingType.MULTI_PART)]
    ) -> PreprocessResponse:
        """Preprocess an audio file."""
        return await preprocess_audio_service(data)