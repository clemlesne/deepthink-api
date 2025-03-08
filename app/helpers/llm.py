import asyncio
from collections.abc import Awaitable, Callable
from json import loads

import litellm
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, CrawlResult
from diskcache import Cache as DiskCache
from litellm import ChatCompletionMessageToolCall, Choices, acompletion
from litellm.caching.caching import Cache
from litellm.files.main import ModelResponse
from litellm.types.caching import LiteLLMCacheType
from litellm.types.completion import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from litellm.types.utils import Message
from litellm.utils import function_to_dict, token_counter

from app.helpers.logging import logger
from app.models.chat_completion import (
    Usage,
)

# Enable Litellm cache
litellm.cache = Cache(
    disk_cache_dir=".litellm_cache",
    type=LiteLLMCacheType.DISK,
)

# Crawler configuration
CRAWL_CONFIG = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    verbose=False,
)
CRAWL_CACHE = DiskCache(".crawl4ai_cache")


class CompletionException(Exception):
    pass


async def read_url_tool(
    url: str,
) -> str:
    """
    Read a URL and return the content.
    """

    async def _exec() -> str | None:
        # Try cache
        cache_key = url
        cache = CRAWL_CACHE.get(cache_key)
        if cache:
            return cache  # pyright: ignore[reportReturnType]

        # Scrape
        async with AsyncWebCrawler() as crawler:
            result: CrawlResult = await crawler.arun(
                config=CRAWL_CONFIG,
                url=url,
            )  # pyright: ignore[reportAssignmentType]
        res = str(result.markdown)

        # Update cache
        CRAWL_CACHE.set(
            expire=60 * 60 * 24 * 7,  # 7 days
            key=cache_key,
            value=res,
        )

        return res

    # Execute
    markdown = await _exec()

    # Inform LLM if no content
    if not markdown:
        return "Could not read URL."

    # Return page content
    return markdown


async def abstract_completion(  # noqa: PLR0913
    messages: list[
        Message
        | ChatCompletionSystemMessageParam
        | ChatCompletionUserMessageParam
        | ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
    ],
    model: str,
    temperature: float,
    top_p: float,
    usage: Usage,
    json: bool = False,
    tools: list[Callable[..., Awaitable[str]]] = [],
) -> str:
    """
    Get a response from the LLM.

    Exception is raised if the response is truncated or empty.
    """
    new_messages = messages.copy()
    tools_dict = [
        {
            "type": "function",
            "function": function_to_dict(tool),
        }
        for tool in tools
    ]

    res: ModelResponse = await acompletion(
        caching=True,
        messages=new_messages,
        model=model,
        response_format={"type": "json_object"} if json else None,
        temperature=temperature,
        tools=tools_dict,
        top_p=top_p,
    )  # pyright: ignore[reportAssignmentType]
    choice: Choices = res.choices[0]  # pyright: ignore[reportAssignmentType]

    # Update usage
    # TODO: Consumption is not accurate at all with that method, use data from the response
    usage.completion_tokens += token_counter(
        text=str(choice.message),
        model=model,
    )  # Increment completion from the response string
    usage.prompt_tokens += token_counter(
        text="\n".join([str(message) for message in new_messages]),
        model=model,
    )  # Increment prompt from the request objects

    # Check for tools
    tool_calls = choice.message.tool_calls
    if tool_calls:
        # List available functions
        available_functions = {function.__name__: function for function in tools}

        # Extend conversation with assistant's reply
        new_messages.append(choice.message)

        # Execute tools
        for tool in await asyncio.gather(
            *[
                _execute_tool(
                    available_functions=available_functions,
                    tool_call=tool_call,
                )
                for tool_call in tool_calls
            ]
        ):
            # Skip if no tool
            if not tool:
                continue

            # Add to history
            new_messages.append(tool)

        # Re-run the completion with the new messages
        return await abstract_completion(
            messages=new_messages,
            model=model,
            temperature=temperature,
            tools=tools,
            top_p=top_p,
            usage=usage,
        )

    # Raise if response is truncated
    finish_reason = choice.finish_reason
    if finish_reason != "stop":
        raise CompletionException(
            f"Completion did not finish correctly: {finish_reason}"
        )

    # Raise if no content
    content = choice.message.content
    if not content:
        raise CompletionException("Completion message is empty")

    return content


async def _execute_tool(
    tool_call: ChatCompletionMessageToolCall,
    available_functions: dict[str, Callable[..., Awaitable[str]]],
) -> ChatCompletionToolMessageParam | None:
    # Skip if no function name
    function_name = tool_call.function.name
    if not function_name:
        return

    # Skip if function not available
    if function_name not in available_functions:
        return

    # Execute
    logger.debug("Executing tool: %s", function_name)
    function_to_call = available_functions[function_name]
    function_response = await function_to_call(**loads(tool_call.function.arguments))

    # Build tool model
    return ChatCompletionToolMessageParam(
        content=function_response,
        role="tool",
        tool_call_id=tool_call.id,
    )
