from enum import StrEnum

from litellm.types.completion import ChatCompletionAssistantMessageParam
from pydantic import BaseModel, Field

from app.helpers.llm import MessagesList
from app.models.chat_completion import ChatCompletionRequest, Usage


class ObjectiveStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"


class StepState(BaseModel):
    short_name: str = Field(
        description="A summary of the thinking process. This should be a short name, in the form of a single word or a short phrase."
    )
    thinking: str = Field(
        description="The reasoning behind the step. This should be a detailed explanation of the thought process."
    )


class ObjectiveState(BaseModel):
    answer: str | None = None
    description: str
    knowledge: str = ""
    short_name: str
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    steps: list[StepState] = []

    @property
    def history(self) -> MessagesList:
        return [
            ChatCompletionAssistantMessageParam(
                content=step.thinking,
                role="assistant",
            )
            for step in self.steps
        ]


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
