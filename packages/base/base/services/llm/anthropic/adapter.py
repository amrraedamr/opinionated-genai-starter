"""Turning this library's neutral messages into Anthropic's wire shapes."""

import json
from collections.abc import Sequence
from typing import Any

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.content import ContentPart, ImagePart, TextPart
from base.models.view_models.prompt import (
    RenderedPrompt,
    RenderedPromptInstruction,
    RenderedPromptMessage,
)
from base.services.tool_calling import Tool
from base.tools.sanitize import strip_invisible


class AnthropicMessageAdapter:
    """Anthropic keeps instructions out of the message list.

    `system` is a top-level request field rather than a role, and tool results
    are user-role content blocks rather than their own role, so this returns
    the two halves separately.
    """

    @classmethod
    def to_messages(
        cls,
        *,
        messages: Sequence[StoredMessage] | None = None,
        prompt: RenderedPrompt | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return `(messages, system)`."""
        if messages is None and prompt is None:
            raise ValueError("At least one of `messages` or `prompt` must be provided")

        instructions: list[str] = []
        items: list[dict[str, Any]] = []

        if prompt is not None:
            for instruction in prompt.instructions:
                instructions.append(cls._to_system_text(instruction))
            for message in prompt.messages:
                items.append(cls._to_prompt_message(message))

        if messages is not None:
            for stored in messages:
                if stored.role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
                    instructions.append(strip_invisible(stored.content or ""))
                    continue
                items.append(cls._to_stored_message(stored))

        return cls._merge_consecutive(items), "\n\n".join(instructions) or None

    @staticmethod
    def _to_system_text(instruction: RenderedPromptInstruction) -> str:
        return instruction.content

    @staticmethod
    def _to_prompt_message(message: RenderedPromptMessage) -> dict[str, Any]:
        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Prompt TOOL message is missing tool_call_id")
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }
        if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
            raise ValueError(f"Unsupported prompt message role: {message.role}")
        return {"role": message.role.value, "content": message.content}

    @classmethod
    def _to_stored_message(cls, message: StoredMessage) -> dict[str, Any]:
        content = strip_invisible(message.content or "")

        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Stored TOOL message is missing tool_call_id")
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": content,
                    }
                ],
            }

        if message.role == MessageRole.USER:
            if message.parts:
                return {"role": "user", "content": cls.to_parts(message.parts)}
            return {"role": "user", "content": content}

        if message.role == MessageRole.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tool_call in message.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": cls._decode_arguments(tool_call.function.arguments),
                    }
                )
            return {"role": "assistant", "content": blocks}

        raise ValueError(f"Unsupported stored message role: {message.role}")

    @staticmethod
    def _merge_consecutive(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold adjacent same-role turns into one.

        Anthropic rejects two consecutive messages with the same role, which a
        parallel tool call produces naturally: one result message per call.
        """
        merged: list[dict[str, Any]] = []
        for item in items:
            if merged and merged[-1]["role"] == item["role"]:
                previous = merged[-1]
                previous["content"] = AnthropicMessageAdapter._as_blocks(
                    previous["content"]
                ) + AnthropicMessageAdapter._as_blocks(item["content"])
                continue
            merged.append(dict(item))
        return merged

    @staticmethod
    def to_parts(parts: Sequence[ContentPart]) -> list[dict[str, Any]]:
        """Content parts as Anthropic content blocks."""
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                blocks.append({"type": "text", "text": strip_invisible(part.text)})
            elif isinstance(part, ImagePart):
                source = (
                    {"type": "url", "url": part.url}
                    if part.url
                    else {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": part.data,
                    }
                )
                blocks.append({"type": "image", "source": source})
            else:
                if part.file_id:
                    source = {"type": "file", "file_id": part.file_id}
                else:
                    source = {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": part.data,
                    }
                blocks.append({"type": "document", "source": source})
        return blocks

    @staticmethod
    def _as_blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return content

    @staticmethod
    def _decode_arguments(arguments: str) -> dict[str, Any]:
        """Arguments are stored as a JSON string; Anthropic wants the object."""
        try:
            decoded = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool call arguments are not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Tool call arguments must decode to an object")
        return decoded

    @staticmethod
    def to_tools(tools: Sequence[Tool[Any]]) -> list[dict[str, Any]]:
        """Tools as Anthropic wants them: schema under `input_schema`."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.json_schema,
            }
            for tool in tools
        ]
