"""Tool declaration, dispatch, and the loop that runs a turn to an answer."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from base.models.definitions.message import MessageRole
from base.models.view_models import BaseModel
from base.models.view_models.chat_history import (
    ConversationId,
    StoredMessage,
    StoredToolCall,
    StoredToolCallFunction,
)
from base.models.view_models.llm_result import CompletionResult, ResponseResult
from base.services.tool_calling import Tool, Toolbox, ToolError, ToolRunner


class Weather(BaseModel):
    city: str
    units: str = "celsius"


async def _get_weather(args: Weather) -> str:
    return f"4 degrees {args.units} in {args.city}"


async def _explodes(args: Weather) -> str:
    raise RuntimeError("upstream is down")


async def _returns_an_object(args: Weather) -> dict[str, Any]:
    return {"city": args.city, "temperature": 4}


def _tool(name: str = "get_weather", handler=_get_weather) -> Tool[Weather]:
    return Tool(
        name=name,
        description="Current weather for a city.",
        parameters=Weather,
        handler=handler,
    )


def _call(name: str = "get_weather", **arguments: Any) -> StoredToolCall:
    return StoredToolCall(
        id=f"call_{uuid4().hex[:8]}",
        function=StoredToolCallFunction(
            name=name, arguments=json.dumps(arguments or {"city": "Oslo"})
        ),
    )


class TestToolSchema:
    def test_it_describes_the_arguments_from_the_model(self):
        schema = _tool().schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["description"]

        properties = schema["function"]["parameters"]["properties"]
        assert set(properties) == {"city", "units"}

    def test_a_required_argument_is_marked_required(self):
        required = _tool().schema["function"]["parameters"]["required"]
        assert required == ["city"]


class TestToolInvocation:
    async def test_it_parses_arguments_and_calls_the_handler(self):
        assert await _tool().ainvoke('{"city": "Oslo"}') == "4 degrees celsius in Oslo"

    async def test_a_default_argument_is_applied(self):
        result = await _tool().ainvoke('{"city": "Oslo", "units": "fahrenheit"}')
        assert "fahrenheit" in result

    async def test_bad_arguments_raise_rather_than_reach_the_handler(self):
        with pytest.raises(ToolError):
            await _tool().ainvoke('{"units": "celsius"}')

    async def test_malformed_json_raises(self):
        with pytest.raises(ToolError):
            await _tool().ainvoke("not json")

    async def test_a_non_string_result_is_serialized(self):
        result = await _tool(handler=_returns_an_object).ainvoke('{"city": "Oslo"}')
        assert json.loads(result)["temperature"] == 4


class TestToolbox:
    def test_it_reports_its_tools(self):
        toolbox = Toolbox(_tool(), _tool("get_time"))
        assert len(toolbox) == 2
        assert toolbox.names == ["get_time", "get_weather"]
        assert "get_weather" in toolbox
        assert len(toolbox.schemas) == 2

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(ToolError):
            Toolbox(_tool(), _tool())

    async def test_it_dispatches_to_the_named_tool(self):
        assert "Oslo" in await Toolbox(_tool()).ainvoke(_call(city="Oslo"))

    async def test_an_unknown_tool_is_reported_to_the_model(self):
        result = await Toolbox(_tool()).ainvoke(_call("get_time"))
        assert result.startswith("Error:")
        assert "get_weather" in result

    async def test_a_failing_handler_is_reported_to_the_model(self):
        result = await Toolbox(_tool(handler=_explodes)).ainvoke(_call(city="Oslo"))
        assert result.startswith("Error:")
        assert "upstream is down" in result

    async def test_bad_arguments_are_reported_to_the_model(self):
        toolbox = Toolbox(_tool())
        call = StoredToolCall(
            id="call_1",
            function=StoredToolCallFunction(name="get_weather", arguments="{}"),
        )
        assert (await toolbox.ainvoke(call)).startswith("Error:")


def _turn(
    content: str | None = None, tool_calls: list[StoredToolCall] | None = None
) -> CompletionResult:
    """A real CompletionResult, so the runner sees the shape the SDK produces."""
    return CompletionResult(
        message=ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id=call.id,
                    type="function",
                    function=Function(
                        name=call.function.name, arguments=call.function.arguments
                    ),
                )
                for call in tool_calls or []
            ]
            or None,
        )
    )


class _StubLLM:
    """Replays a scripted sequence of model turns and records what it was sent."""

    def __init__(self, *turns: CompletionResult):
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def acompletions(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self._turns.pop(0)


def _conversation() -> list[StoredMessage]:
    return [
        StoredMessage(
            conversation_id=ConversationId(uuid4()),
            sequence=1,
            role=MessageRole.USER,
            content="What is the weather in Oslo?",
        )
    ]


@pytest.fixture
def runner() -> ToolRunner:
    return ToolRunner(Toolbox(_tool()))


class TestToolRunner:
    async def test_a_direct_answer_takes_one_round(self, runner):
        runner._llm_service = _StubLLM(_turn("It is cold."))

        outcome = await runner.arun(_conversation())

        assert outcome.rounds == 1
        assert outcome.result.content == "It is cold."
        assert outcome.messages == []

    async def test_a_tool_call_is_executed_and_fed_back(self, runner):
        call = _call(city="Oslo")
        stub = _StubLLM(
            _turn(tool_calls=[call]),
            _turn("It is 4 degrees in Oslo."),
        )
        runner._llm_service = stub

        outcome = await runner.arun(_conversation())

        assert outcome.rounds == 2
        assert outcome.result.content == "It is 4 degrees in Oslo."

        assistant, tool_message = outcome.messages
        assert assistant.role == MessageRole.ASSISTANT
        assert assistant.tool_calls == [call]
        assert tool_message.role == MessageRole.TOOL
        assert tool_message.tool_call_id == call.id
        assert "4 degrees celsius in Oslo" in (tool_message.content or "")

    async def test_the_tool_result_reaches_the_next_request(self, runner):
        stub = _StubLLM(
            _turn(tool_calls=[_call(city="Oslo")]),
            _turn("Done."),
        )
        runner._llm_service = stub

        await runner.arun(_conversation())

        second_request = json.dumps(stub.calls[1]["messages"], default=str)
        assert "4 degrees celsius in Oslo" in second_request

    async def test_the_toolbox_is_offered_to_the_model(self, runner):
        stub = _StubLLM(_turn("Hi."))
        runner._llm_service = stub

        await runner.arun(_conversation())

        offered = stub.calls[0]["kwargs"]["tools"]
        assert offered[0]["function"]["name"] == "get_weather"

    async def test_produced_messages_keep_increasing_sequences(self, runner):
        runner._llm_service = _StubLLM(
            _turn(tool_calls=[_call(city="A"), _call(city="B")]),
            _turn("Done."),
        )

        outcome = await runner.arun(_conversation())

        sequences = [message.sequence for message in outcome.messages]
        assert sequences == sorted(sequences)
        assert sequences[0] > 1

    async def test_the_input_conversation_is_not_mutated(self, runner):
        runner._llm_service = _StubLLM(
            _turn(tool_calls=[_call(city="Oslo")]),
            _turn("Done."),
        )
        messages = _conversation()

        await runner.arun(messages)

        assert len(messages) == 1

    async def test_it_stops_a_model_that_will_not_stop(self):
        runner = ToolRunner(Toolbox(_tool()), max_rounds=2)
        runner._llm_service = _StubLLM(
            _turn(tool_calls=[_call(city="Oslo")]),
            _turn(tool_calls=[_call(city="Oslo")]),
        )

        with pytest.raises(ToolError, match="max_rounds"):
            await runner.arun(_conversation())

    async def test_an_empty_conversation_is_rejected(self, runner):
        with pytest.raises(ToolError):
            await runner.arun([])

    def test_max_rounds_must_allow_at_least_one_call(self):
        with pytest.raises(ToolError):
            ToolRunner(Toolbox(_tool()), max_rounds=0)


def _response_turn(
    content: str | None = None, tool_calls: list[StoredToolCall] | None = None
) -> ResponseResult:
    """A real ResponseResult, so the Responses path is exercised as it ships."""
    output: list[Any] = []
    if content is not None:
        output.append(
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(type="output_text", text=content, annotations=[])
                ],
            )
        )
    for call in tool_calls or []:
        output.append(
            ResponseFunctionToolCall(
                type="function_call",
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
        )
    return ResponseResult(
        response=Response(
            id="resp_1",
            created_at=0,
            model="gpt-test",
            object="response",
            output=output,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
    )


class _StubResponsesLLM:
    def __init__(self, *turns: ResponseResult):
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def aresponses(self, input_, **kwargs):
        self.calls.append({"input": input_, "kwargs": kwargs})
        return self._turns.pop(0)


class TestResponsesApi:
    async def test_the_flat_tool_shape_is_offered(self):
        runner = ToolRunner(Toolbox(_tool()), api="responses")
        stub = _StubResponsesLLM(_response_turn("Hi."))
        runner._llm_service = stub

        await runner.arun(_conversation())

        offered = stub.calls[0]["kwargs"]["tools"]
        assert offered[0]["name"] == "get_weather"
        assert "function" not in offered[0]

    async def test_a_tool_call_is_executed_and_fed_back(self):
        runner = ToolRunner(Toolbox(_tool()), api="responses")
        call = _call(city="Oslo")
        stub = _StubResponsesLLM(
            _response_turn(tool_calls=[call]),
            _response_turn("It is 4 degrees in Oslo."),
        )
        runner._llm_service = stub

        outcome = await runner.arun(_conversation())

        assert outcome.rounds == 2
        assert outcome.result.content == "It is 4 degrees in Oslo."
        assistant, tool_message = outcome.messages
        assert assistant.tool_calls == [call]
        assert "4 degrees celsius in Oslo" in (tool_message.content or "")

    async def test_the_tool_result_reaches_the_next_request(self):
        runner = ToolRunner(Toolbox(_tool()), api="responses")
        stub = _StubResponsesLLM(
            _response_turn(tool_calls=[_call(city="Oslo")]),
            _response_turn("Done."),
        )
        runner._llm_service = stub

        await runner.arun(_conversation())

        second_request = json.dumps(stub.calls[1]["input"], default=str)
        assert "4 degrees celsius in Oslo" in second_request
