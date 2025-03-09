import os

import pytest
from litellm import acompletion


@pytest.mark.asyncio
async def test_llm():
    # Dummy message
    messages = [
        {
            "content": "What is the status of the world?",
            "role": "user",
        }
    ]

    # Test API
    response = await acompletion(
        api_key="mock",
        base_url="http://localhost:8080/v1",
        messages=messages,
        model="openai/gpt-4o-mini",
    )

    # Print for debugging
    print(response)  # noqa: T201
