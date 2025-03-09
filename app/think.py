import asyncio

from aiojobs import Scheduler
from pydantic import BaseModel, Field
from structlog.contextvars import bound_contextvars

from app.helpers.llm import non_empty_completion, read_url_tool, validated_completion
from app.helpers.logging import logger
from app.models.chat_completion import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from app.models.state import ObjectiveState, ObjectiveStatus, StepState, ThinkState

MAX_OBJECTIVES = 5
MIN_STEPS = 3
MAX_STEPS = 10


async def think(req: ChatCompletionRequest) -> ChatCompletionResponse:
    state = ThinkState(
        req=req,
    )
    logger.debug("Initialized think state")

    # Add first objective
    state.objectives.append(
        ObjectiveState(
            description="Identify the ins and outs of the question. What? Why? How? When? Where? Who?",
            knowledge=f"User question is: {state.user_question}",
            short_name="Identify question",
        )
    )

    async with Scheduler() as scheduler:
        # Schedule the first objective
        await scheduler.spawn(
            _run_objective(
                think=state,
                objective=state.objectives[0],
            )
        )

        # Run the scheduler until all objectives are completed or failed
        while any(
            objective.status in (ObjectiveStatus.IN_PROGRESS, ObjectiveStatus.PENDING)
            for objective in state.objectives
        ):
            # Wait for all tasks to complete
            while scheduler.active_count > 0:
                await asyncio.sleep(0.1)
                continue

            # Update state
            new_objectives: list[ObjectiveState] = []
            for objective in await _detect_new_objectives(state):
                # Stop if max objectives reached
                if len(state.objectives) >= MAX_OBJECTIVES:
                    logger.debug("Max objectives reached")
                    break
                new_objectives.append(objective)
                state.objectives.append(objective)

            # Abort iteration if no new objectives
            if not new_objectives:
                # If maximum objectives reached, stop the loop
                if len(state.objectives) >= MAX_OBJECTIVES:
                    logger.debug("Max objectives reached")
                    break
                # Wait for objectives to complete
                continue

            logger.debug(
                "New objectives: %s",
                [objective.short_name for objective in new_objectives],
            )

            # Schedule new objectives
            for objective in new_objectives:
                await scheduler.spawn(
                    _run_objective(
                        think=state,
                        objective=objective,
                    )
                )

    # Pretty pring objectives and steps with a tree, for debugging
    logger.debug("Initial question: %s", state.user_question)
    for objective in state.objectives:
        logger.debug(
            " | Objective: %s (%s)",
            objective.short_name,
            objective.status,
        )
        for step in objective.steps:
            logger.debug(
                " |-- Step: %s",
                step.short_name,
            )

    # Get answer
    answer = await _answer_user(state)
    logger.debug("Answer: %s", answer)

    # Print usage
    logger.debug(
        "Usage: prompt=%i, completion=%i",
        state.usage.prompt_tokens,
        state.usage.completion_tokens,
    )

    # Create response
    return ChatCompletionResponse(
        model=state.req.model,
        usage=Usage(
            prompt_tokens=state.usage.prompt_tokens,
            completion_tokens=state.usage.completion_tokens,
        ),
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    content=f"""
                        <thinking>
                        {"\n".join([f"- {objective.description}" for objective in state.objectives])}
                        </thinking>
                        {answer}
                    """,
                    role="assistant",
                ),
                finish_reason="stop",
            ),
        ],
    )


async def _answer_user(
    think: ThinkState,
) -> str:
    """
    Answer the user question.

    This function is called when the assistant can answer the question with a high level of confidence.
    """
    res = await non_empty_completion(
        model=think.req.model,
        temperature=think.req.temperature,
        top_p=think.req.top_p,
        usage=think.usage,
        system=f"""
            Assistant is a business analyst with 20 years of experience.

            # Objective
            Answer the following question. Answer must be sourced and quantified with a high level of detail.

            # Context
            You gathered knowledge from research and analysis. This knowledge is trusted and reliable.

            # Rules
            - Don't make assumptions
            - Only use the knowledge you gathered to answer

            # Question
            {think.user_question}

            # Knowledge
            {"\n".join([f"{objective.description}: {objective.knowledge}" for objective in think.objectives if objective.status == ObjectiveStatus.COMPLETED])}
        """,
    )

    return res


async def _detect_new_objectives(
    think: ThinkState,
) -> list[ObjectiveState]:
    """
    Detect new objectives from the current state.

    Objectives are a list of questions that the assistant will answer iteratively, in order to answer the main question.

    Response is a list of tasks that the assistant will do to answer the question. If the assistant can answer the question, list will be empty.
    """

    class _Objective(BaseModel):
        description: str
        short_name: str

    class _Res(BaseModel):
        tasks: list[_Objective] = Field(
            max_length=3,  # Not too much new objectives to make sure the thinking is focused
        )

    res = await validated_completion(
        res_type=_Res,
        model=think.req.model,
        temperature=think.req.temperature,
        top_p=think.req.top_p,
        usage=think.usage,
        system=f"""
            Assistant is a business analyst with 20 years of experience.

            # Objective
            Determine if the following question can be answered with a high level of confidence. If not, what would be the ideal tasks to answer it? You must be able to add enough sources and quantification to answer the question.

            # Context
            You already worked on objectives to solve the problem.

            # Rules
            - Don't make assumptions
            - Only use the knowledge you gathered to answer

            # Question
            {think.user_question}

            # Ideas you worked on
            {"\n".join([f"{objective.description}: {objective.answer}" for objective in think.objectives if objective.answer])}
            {"\n".join([f"{objective.description}: {objective.status}" for objective in think.objectives if not objective.answer])}

            # Response options
            - A list of tasks, to help you answer the question
            - Empty array, if you are confident you can answer the question with the knowledge you gathered
        """,
    )

    return [
        ObjectiveState(
            description=task.description,
            short_name=task.short_name,
        )
        for task in res.tasks
    ]


async def _run_objective(
    think: ThinkState,
    objective: ObjectiveState,
) -> None:
    """
    Run the objective until it's completed or failed.

    The objective is a list of steps that will be executed one by one.
    """
    with bound_contextvars(
        objective=objective.short_name,
    ):
        logger.debug("Starting objective: %s", objective.description)
        objective.status = ObjectiveStatus.IN_PROGRESS

        while objective.status is ObjectiveStatus.IN_PROGRESS:
            # Ensure a minimum steps to force cognition
            if len(objective.steps) >= MIN_STEPS:
                # Check if completed
                should_stop = await _should_stop_objective(
                    think=think,
                    objective=objective,
                )
                if isinstance(should_stop, str):
                    logger.debug("Objective completed: %s", should_stop)
                    objective.status = ObjectiveStatus.COMPLETED
                    objective.answer = should_stop
                    break

            # Check if max steps reached
            if len(objective.steps) >= MAX_STEPS:
                objective.status = ObjectiveStatus.FAILED
                break

            # Create new step
            step = await _new_step(
                objective=objective,
                think=think,
            )
            objective.steps.append(step)
            logger.debug(
                "New step: %s (%i/%i)",
                step.short_name,
                len(objective.steps),
                MAX_STEPS,
            )

        logger.debug("Objective ended: %s", objective.status)


async def _should_stop_objective(
    think: ThinkState,
    objective: ObjectiveState,
) -> bool | str:
    """
    Check if the assistant can answer the objective.

    If the assistant can answer the objective, return the answer. Else, return False.
    """
    res = await non_empty_completion(
        model=think.req.model,
        temperature=think.req.temperature,
        top_p=think.req.top_p,
        usage=think.usage,
        system=f"""
            Assistant is a business analyst with 20 years of experience.

            # Objective
            Can you answer the following question with a high level of confidence?

            # Context
            You gathered knowledge from research and analysis. This knowledge is trusted and reliable.

            # Rules
            - Don't make assumptions
            - Only use the knowledge you gathered to answer

            # Question
            {objective.description}

            # Knowledge
            {objective.knowledge}

            # Response options
            - "Can't answer", if you can't answer the question
            - The full answer, if you can answer the question
        """,
    )

    # Return false if can't answer
    if "can't answer" in res.lower():
        return False

    # Return the answer
    return res


async def _new_step(
    objective: ObjectiveState,
    think: ThinkState,
) -> StepState:
    """
    Create a new step for the objective.

    Step is a question that will be asked to the assistant.
    """

    async def _knowledge_tool(
        knowledge: str,
    ) -> str:
        """
        Persist knowledge into the documentation.

        A knowledge is a sourced informmation that will be used to answer the question. Like a research, a data, or a fact.
        """
        objective.knowledge += f"\n{knowledge}"
        return "Knowledge persisted."

    return await validated_completion(
        existing_history=objective.history,
        model=think.req.model,
        res_type=StepState,
        temperature=think.req.temperature,
        tools=[
            _knowledge_tool,
            read_url_tool(
                model=think.req.model,
                temperature=think.req.temperature,
                top_p=think.req.top_p,
                usage=think.usage,
            ),
        ],
        top_p=think.req.top_p,
        usage=think.usage,
        system=f"""
            Assistant is a business analyst with 20 years of experience. Assistant is meticulous and perfectionist.

            # Objective
            Solve a problem with a high level of confidence. Think step by step and store relevent knowledge. Knowledge will be used in the end to answer the question. Always find a way to enhance your knowledge and ensure you get the maximim out of the research.

            # Context
            Effort is limited to {MAX_STEPS} steps, you've already completed {len(objective.steps)} one. Do your best to fulfill the objective within this limit.

            # Rules
            - Be concise, no yapping
            - Don't make assumptions
            - Only use the knowledge you gathered to answer

            # Question
            {objective.description}

            # Knowledge
            {objective.knowledge}
        """,
    )
