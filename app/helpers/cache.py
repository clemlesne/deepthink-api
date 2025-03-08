import asyncio
from collections import OrderedDict
from collections.abc import Awaitable
from functools import wraps


def lru_acache(maxsize: int = 128):
    """
    Caches an async function's return value each time it is called.

    If the maxsize is reached, the least recently used value is removed.
    """

    def decorator(func):
        cache: OrderedDict[tuple, Awaitable] = OrderedDict()

        @wraps(func)
        async def wrapper(*args, **kwargs) -> Awaitable:
            # Create a cache key from event loop, args and kwargs, using frozenset for kwargs to ensure hashability
            key = (
                id(asyncio.get_event_loop()),
                args,
                frozenset(kwargs.items()),
            )

            if key in cache:
                # Move the recently accessed key to the end (most recently used)
                cache.move_to_end(key)
                return cache[key]

            # Compute the value since it's not cached
            value = await func(*args, **kwargs)
            cache[key] = value
            cache.move_to_end(key)

            # Remove the least recently used key if the cache is full
            if len(cache) > maxsize:
                cache.popitem(last=False)

            return value

        return wrapper

    return decorator


def lru_cache(maxsize: int = 128):
    """
    Caches a sync function's return value each time it is called.

    If the maxsize is reached, the least recently used value is removed.
    """

    def decorator(func):
        cache: OrderedDict[tuple, Awaitable] = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs) -> Awaitable:
            # Create a cache key from args and kwargs, using frozenset for kwargs to ensure hashability
            key = (
                args,
                frozenset(kwargs.items()),
            )

            if key in cache:
                # Move the recently accessed key to the end (most recently used)
                cache.move_to_end(key)
                return cache[key]

            # Compute the value since it's not cached
            value = func(*args, **kwargs)
            cache[key] = value
            cache.move_to_end(key)

            # Remove the least recently used key if the cache is full
            if len(cache) > maxsize:
                cache.popitem(last=False)

            return value

        return wrapper

    return decorator
