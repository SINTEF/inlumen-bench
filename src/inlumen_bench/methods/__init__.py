"""Generation methods. Each one turns an exported workspace into a submission."""

from .base import EpisodeContext, Method, MethodResult
from .direct import Direct

METHODS: dict[str, type[Method]] = {Direct.name: Direct}


def get_method(name: str) -> Method:
    if name not in METHODS:
        raise KeyError(f"Unknown method {name!r}; available: {', '.join(METHODS)}")
    return METHODS[name]()


__all__ = ["METHODS", "EpisodeContext", "Method", "MethodResult", "get_method"]
