from fastapi import FastAPI

from app.helpers.logging import VERSION, logger
from app.models.chat_completion import ChatCompletionRequest
from app.think import think

# First log
logger.info(
    "deepthink-api v%s",
    VERSION,
)

# FastAPI
api = FastAPI(
    contact={
        "url": "https://github.com/clemlesne/deepthink-api",
    },
    description="DeepThink exposed as a standard LLM API with thinking. Swappable model backend.",
    license_info={
        "name": "Apache-2.0",
        "url": "https://github.com/clemlesne/deepthink-api/blob/master/LICENSE",
    },
    title="deepthink-api",
    version=VERSION,
)


@api.post("/v1/chat/completions")
async def v1_chat_completions(req: ChatCompletionRequest):
    return await think(req)
