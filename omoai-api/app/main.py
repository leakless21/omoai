from litestar import Litestar

from app.domain.audio.controllers import TranscriptionController

app = Litestar(route_handlers=[TranscriptionController])