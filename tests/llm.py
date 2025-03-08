import os

import pytest
from litellm import acompletion
from litellm._logging import _turn_on_debug

# Debug LiteLLM
_turn_on_debug()


@pytest.mark.asyncio
async def test_llm():
    # Perpare auth
    os.environ["OPENAI_API_KEY"] = "mock"

    # Dummy message
    messages = [
        {
            "content": "What is the status of the world?",
            "role": "user",
        }
    ]

    # Test API
    response = await acompletion(
        base_url="http://localhost:8080/v1",
        messages=messages,
        model="openai/gpt-4o-mini",
    )

    # Print for debugging
    print(response)  # noqa: T201
