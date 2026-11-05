"""Turning this library's neutral messages into Gemini's wire shapes.

Gemini differs from the other two in three ways that matter here. Instructions
are a request field rather than a role, the assistant's role is called `model`,
and a tool result is matched to its call by **function name** rather than by an
id — so the id this library carries has to be resolved back to the name the
model used before a result can be sent.
"""

import base64
import json
from collections.abc import Sequence
from typing import Any

from base.models.definitions.message import MessageRole
from base.models.view_models.chat_history import StoredMessage
from base.models.view_models.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    TextPart,
)
from base.models.view_models.prompt import RenderedPrompt, RenderedPromptMessage
from base.services.tool_calling import Tool
from base.tools.sanitize import strip_invisible

_ROLE_MAP: dict[MessageRole, str] = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "model",
    MessageRole.TOOL: "user",
}
"""Which turn each of our roles becomes.

Gemini calls the assistant `model`, and has no role of its own for a tool
result — it travels as a `function_response` part inside a user turn.
"""


class GeminiContentAdapter:
    """Neutral messages as Gemini `Content` dictionaries.

    Returns the contents and the system instruction separately, because Gemini
    takes the latter on the request rather than in the conversation.
    """

    @classmethod
    def to_contents(
        cls,
        *,
        messages: Sequence[StoredMessage] | None = None,
        prompt: RenderedPrompt | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return `(contents, system_instruction)`."""
        if messages is None and prompt is None:
            raise ValueError("At least one of `messages` or `prompt` must be provided")

        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        names = cls._tool_call_names(messages, prompt)

        if prompt is not None:
            for instruction in prompt.instructions:
                instructions.append(instruction.content)
            for message in prompt.messages:
                items.append(cls._from_prompt_message(message, names))

        if messages is not None:
            for stored in messages:
                if stored.role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
                    instructions.append(strip_invisible(stored.content or ""))
                    continue
                items.append(cls._from_stored_message(stored, names))

        return cls._merge_consecutive(items), "\n\n".join(instructions) or None

    @staticmethod
    def _tool_call_names(
        messages: Sequence[StoredMessage] | None,
        prompt: RenderedPrompt | None,
    ) -> dict[str, str]:
        """Which function each tool call id belongs to.

        Gemini pairs a result with its call by name, so a `tool_call_id` only
        means something once it is resolved against the call that produced it.
        """
        names: dict[str, str] = {}
        for stored in messages or []:
            for call in stored.tool_calls or []:
                names[call.id] = call.function.name
        for message in getattr(prompt, "messages", None) or []:
            for call in getattr(message, "tool_calls", None) or []:
                names[call.id] = call.function.name
        return names

    @classmethod
    def _from_prompt_message(
        cls, message: RenderedPromptMessage, names: dict[str, str]
    ) -> dict[str, Any]:
        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Prompt TOOL message is missing tool_call_id")
            return cls._function_response(
                message.tool_call_id, message.content or "", names
            )
        if message.role in (MessageRole.USER, MessageRole.ASSISTANT):
            return {
                "role": _ROLE_MAP[message.role],
                "parts": [{"text": message.content or ""}],
            }
        raise ValueError(f"Unsupported prompt message role: {message.role}")

    @classmethod
    def _from_stored_message(
        cls, message: StoredMessage, names: dict[str, str]
    ) -> dict[str, Any]:
        content = strip_invisible(message.content or "")

        if message.role == MessageRole.TOOL:
            if not message.tool_call_id:
                raise ValueError("Stored TOOL message is missing tool_call_id")
            return cls._function_response(message.tool_call_id, content, names)

        if message.role == MessageRole.USER:
            parts = cls.to_parts(message.parts) if message.parts else [{"text": content}]
            return {"role": _ROLE_MAP[message.role], "parts": parts}

        if message.role == MessageRole.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"text": content})
            for call in message.tool_calls or []:
                blocks.append(
                    {
                        "function_call": {
                            "name": call.function.name,
                            "args": cls._decode_arguments(call.function.arguments),
                        }
                    }
                )
            return {"role": _ROLE_MAP[message.role], "parts": blocks}

        raise ValueError(f"Unsupported stored message role: {message.role}")

    @staticmethod
    def _function_response(
        tool_call_id: str, content: str, names: dict[str, str]
    ) -> dict[str, Any]:
        """A tool result, named after the call it answers.

        Falling back to the id keeps a result that arrived without its call
        from being dropped, though the model is unlikely to match it.
        """
        name = names.get(tool_call_id, tool_call_id)
        try:
            response: Any = json.loads(content) if content else {}
        except json.JSONDecodeError:
            response = {"result": content}
        if not isinstance(response, dict):
            response = {"result": response}
        return {
            "role": _ROLE_MAP[MessageRole.TOOL],
            "parts": [{"function_response": {"name": name, "response": response}}],
        }

    @staticmethod
    def _merge_consecutive(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fold adjacent same-role turns into one.

        Parallel tool calls produce one result message each, and Gemini reads a
        run of same-role turns as a single turn anyway.
        """
        merged: list[dict[str, Any]] = []
        for item in items:
            if merged and merged[-1]["role"] == item["role"]:
                merged[-1]["parts"] = list(merged[-1]["parts"]) + list(item["parts"])
                continue
            merged.append({"role": item["role"], "parts": list(item["parts"])})
        return merged

    @classmethod
    def to_parts(cls, parts: Sequence[ContentPart]) -> list[dict[str, Any]]:
        """Content parts as Gemini parts."""
        out: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                out.append({"text": part.text})
            elif isinstance(part, ImagePart):
                out.append(cls._binary_part(part.url, part.data, part.media_type))
            elif isinstance(part, DocumentPart):
                out.append(
                    cls._binary_part(
                        None, part.data, part.media_type or "application/pdf"
                    )
                )
        return out

    @staticmethod
    def _binary_part(
        url: str | None, data: str | None, media_type: str | None
    ) -> dict[str, Any]:
        """Inline bytes, or a URI the model fetches itself."""
        if data:
            return {
                "inline_data": {
                    "mime_type": media_type or "application/octet-stream",
                    "data": base64.b64decode(data) if isinstance(data, str) else data,
                }
            }
        return {
            "file_data": {
                "mime_type": media_type or "application/octet-stream",
                "file_uri": url or "",
            }
        }

    @staticmethod
    def _decode_arguments(arguments: str) -> dict[str, Any]:
        try:
            decoded = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @classmethod
    def to_tools(cls, tools: Sequence[Tool[Any]]) -> list[dict[str, Any]]:
        """Tools as one Gemini tool carrying every declaration.

        Gemini takes a list of tools each holding function declarations rather
        than one entry per function, so all of them go in a single tool.
        """
        declarations = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": cls.to_schema(tool.parameters.model_json_schema()),
            }
            for tool in tools
        ]
        return [{"function_declarations": declarations}]

    @classmethod
    def to_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """A Pydantic JSON Schema in the subset Gemini accepts.

        Pydantic emits `$defs` and `$ref` for a nested model and a `title` on
        everything; Gemini rejects both, so references are inlined and the
        keywords it does not know are dropped.
        """
        defs = dict(schema.get("$defs", {}))

        def resolve(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    name = str(node["$ref"]).rsplit("/", 1)[-1]
                    return resolve(dict(defs.get(name, {})))
                return {
                    key: resolve(value)
                    for key, value in node.items()
                    if key not in ("title", "additionalProperties", "$schema", "$defs")
                }
            if isinstance(node, list):
                return [resolve(item) for item in node]
            return node

        resolved = resolve(schema)
        return resolved if isinstance(resolved, dict) else {}
