import os

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion import Choice as SyncChoice
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice

# Dummy message
MESSAGES = [
    ChatCompletionUserMessageParam(
        content="What is the status of the world?",
        role="user",
    )
]


@pytest.mark.asyncio
async def test_openai_sync():
    async with AsyncOpenAI(
        api_key="mock",
        base_url="http://localhost:8080/v1",
    ) as client:
        # Test API
        res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=MESSAGES,
        )

        # Validate response
        assert len(res.choices) == 1
        assert res.choices[0].finish_reason == "stop"
        assert isinstance(res.choices[0], SyncChoice)
        assert res.choices[0].message.content is not None

        # Debug
        print(res.choices[0].message.content)  # noqa: T201


@pytest.mark.asyncio
async def test_openai_stream():
    async with AsyncOpenAI(
        api_key="mock",
        base_url="http://localhost:8080/v1",
    ) as client:
        # Test API
        res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=MESSAGES,
            stream=True,
        )

        # Consume stream
        async for delta in res:
            # Validate response
            assert len(delta.choices) == 1
            assert isinstance(delta.choices[0], StreamChoice)
            if delta.choices[0].delta.content is None:
                assert delta.choices[0].finish_reason == "stop"
            else:
                assert delta.choices[0].finish_reason is None

            # Debug
            print(delta.choices[0].delta.content)  # noqa: T201
