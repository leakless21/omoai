import uuid
from litestar import Controller, post, get
from litestar.datastructures import UploadFile
from litestar.status_codes import HTTP_200_OK

from app.domain.audio.schemas import TranscriptionJob, HealthStatus


class TranscriptionController(Controller):
    path = "/v1"

    @post("/transcribe")
    async def transcribe_audio(self, data: UploadFile) -> TranscriptionJob:
        # In a real implementation, you would process the audio file here.
        # For this minimal version, we return a mock job ID and status.
        mock_job_id = str(uuid.uuid4())
        return TranscriptionJob(job_id=mock_job_id, status="pending")

    @get("/health", status_code=HTTP_200_OK)
    async def health_check(self) -> HealthStatus:
        return HealthStatus(status="ok")