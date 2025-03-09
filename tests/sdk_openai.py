import os

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam


@pytest.mark.asyncio
async def test_llm():
    # Dummy message
    messages = [
        ChatCompletionUserMessageParam(
            content="What is the status of the world?",
            role="user",
        )
    ]

    async with AsyncOpenAI(
        api_key="mock",
        base_url="http://localhost:8080/v1",
    ) as client:
        # Test API
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        # Print for debugging
        print(response)  # noqa: T201
