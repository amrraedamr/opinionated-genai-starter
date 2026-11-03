import asyncio
import base64
import json
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Generator
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_serializable(item) for item in obj)
    elif isinstance(obj, (set, frozenset)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode("utf-8")
    else:
        return obj


def parse_uuid(value: str | UUID | None) -> UUID | None:
    """Coerce a value to a UUID, or None when it is not one."""
    if isinstance(value, UUID):
        return value

    try:
        return UUID(value)
    except (ValueError, TypeError, AttributeError):
        return None


def sort_by_order_key(data: dict[str, Any]) -> OrderedDict[str, Any]:
    return OrderedDict(
        sorted(
            data.items(),
            key=lambda item: (
                item[1].get("display_order", float("inf"))
                if isinstance(item[1], dict)
                else float("inf")
            ),
        )
    )


async def asleep_until(
    condition_lambda: Any, sleep_time: float, polling_time: float
) -> bool:
    start = time.monotonic()
    while not condition_lambda():
        await asyncio.sleep(polling_time)
        if time.monotonic() - start >= sleep_time:
            return False
    return True


def isint(value: Any) -> bool:
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def isfloat(value: Any) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def getint(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def try_parse_json(content: str) -> tuple[bool, Any]:
    """Attempt to parse JSON, returning (success, parsed_value) to avoid double-parsing."""
    try:
        return True, json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False, None


def is_json(content: str) -> bool:
    ok, _ = try_parse_json(content)
    return ok


def parse_json_pairs(
    keys: tuple[str, ...] | list[str], values: list[str | None]
) -> dict[str, Any]:
    """Zip keys to values, JSON-decoding each value and keeping the raw string if it is not JSON."""
    parsed: dict[str, Any] = {}
    for index, key in enumerate(keys):
        raw = values[index]
        if raw is None:
            parsed[key] = None
            continue
        ok, value = try_parse_json(raw)
        parsed[key] = value if ok else raw
    return parsed


def parse_arguments(args: dict[str, str]) -> dict[str, Any]:
    function_args: dict[str, Any] = {}
    for arg_key, arg_val in args.items():
        if arg_val.startswith('"') and arg_val.endswith('"'):
            function_args[arg_key] = arg_val[1:-1]
        elif isint(arg_val):
            function_args[arg_key] = int(arg_val)
        elif isfloat(arg_val):
            function_args[arg_key] = float(arg_val)
        elif arg_val.lower() in ("true", "false"):
            function_args[arg_key] = arg_val.lower() == "true"
        else:
            ok, parsed = try_parse_json(arg_val)
            if ok:
                function_args[arg_key] = parsed
            else:
                function_args[arg_key] = arg_val
    return function_args


def parse_json_text(content: str | None = None) -> Any:
    if content is None:
        return None
    ok, parsed = try_parse_json(content)
    return parsed if ok else content


def generic_deep_deserialize(value: Any) -> Any:
    """Recursively deserialize doubly-serialized JSON strings.

    Handles cases where JSON strings are embedded inside already-parsed
    JSON structures (e.g. a dict value that is itself a JSON-encoded string).
    """
    try:
        deserialized_value = json.loads(value)
        if isinstance(deserialized_value, dict):
            return {
                k: generic_deep_deserialize(v) for k, v in deserialized_value.items()
            }
        elif isinstance(deserialized_value, list):
            return [generic_deep_deserialize(item) for item in deserialized_value]
        else:
            return deserialized_value
    except (json.JSONDecodeError, TypeError):
        return value


def decode_base64(data: str) -> str:
    padding_needed = len(data) % 4
    if padding_needed:
        data += "=" * (4 - padding_needed)
    return base64.b64decode(data).decode("utf-8")


def string_truncate(string: str, max_length: int, tail: str = "...") -> str:
    return string[:max_length] + tail if len(string) > max_length else string


def human_readable_duration_from_minutes(total_minutes: int) -> str:
    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
    else:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"


def normalize_emoji_colors(emoji: str) -> str:
    skin_tone_modifiers = [
        "\U0001f3fb",
        "\U0001f3fc",
        "\U0001f3fd",
        "\U0001f3fe",
        "\U0001f3ff",
    ]

    skin_tone_pattern = re.compile("|".join(skin_tone_modifiers))
    return skin_tone_pattern.sub("", emoji)


def stringify_document_values(doc: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) if not isinstance(v, str) else v for k, v in doc.items()}


def chunk_list(lst: list[Any], size: int) -> Generator[list[Any], None, None]:
    """Yield successive chunks of given size from list."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
