from pydantic import BaseModel


class TranscriptionJob(BaseModel):
    job_id: str
    status: str


class HealthStatus(BaseModel):
    status: str