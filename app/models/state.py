from enum import StrEnum

from pydantic import BaseModel

from app.models.chat_completion import ChatCompletionRequest, Usage


class ObjectiveStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"


class ObjectiveState(BaseModel):
    answer: str | None = None
    description: str
    knowledge: str = ""
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    steps: list[str] = []


class ThinkState(BaseModel):
    objectives: list[ObjectiveState] = []
    req: ChatCompletionRequest
    usage: Usage = Usage()

    @property
    def user_question(self) -> str:
        """
        Extract the user messages from the request.
        """
        return " ".join(
            [message.content for message in self.req.messages if message.role == "user"]
        )
