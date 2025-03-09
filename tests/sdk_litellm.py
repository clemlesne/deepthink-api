import pytest
from litellm import CustomStreamWrapper, acompletion
from litellm.types.utils import Choices, ModelResponse, StreamingChoices

# Dummy message
MESSAGES = [
    {
        "content": "What is the status of the world?",
        "role": "user",
    }
]


@pytest.mark.asyncio
async def test_litellm_sync():
    # Test API
    res: ModelResponse = await acompletion(  # pyright: ignore[reportAssignmentType]
        api_key="mock",
        base_url="http://localhost:8080/v1",
        messages=MESSAGES,
        model="openai/gpt-4o-mini",
    )

    # Validate response
    assert len(res.choices) == 1
    assert res.choices[0].finish_reason == "stop"
    assert isinstance(res.choices[0], Choices)
    assert res.choices[0].message.content is not None

    # Debug
    print(res.choices[0].message.content)  # noqa: T201


@pytest.mark.asyncio
async def test_litellm_stream():
    # Test API
    res: CustomStreamWrapper = await acompletion(  # pyright: ignore[reportAssignmentType]
        api_key="mock",
        base_url="http://localhost:8080/v1",
        messages=MESSAGES,
        model="openai/gpt-4o-mini",
        stream=True,
    )

    # Consume stream
    async for delta in res:
        # Validate response
        assert len(delta.choices) == 1
        assert isinstance(delta.choices[0], StreamingChoices)
        if delta.choices[0].delta.content is None:
            assert delta.choices[0].finish_reason == "stop"
        else:
            assert delta.choices[0].finish_reason is None

        # Debug
        print(delta.choices[0].delta.content)  # noqa: T201
