from litestar import Controller, post
from litestar.datastructures import State

from omoai.api.models import ASRRequest, ASRResponse

from omoai.api.services import asr_service


class ASRController(Controller):
    path = "/asr"

    @post("/")
    async def asr(self, data: ASRRequest, state: State) -> ASRResponse:
        """Process ASR request using the refactored service function."""
        return await asr_service(data)