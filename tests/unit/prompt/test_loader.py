"""`PromptLoader` is the registry a service renders every prompt through.

Templates arrive from code via `@register_prompt`, and a store may override
them. The rules worth pinning are what a store may and may not do: it supplies
a template for a key code already declared, and a key nobody declared is
skipped rather than invented, because a prompt has a name and a description
that only the code knows.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from base.models.view_models.prompt import (
    Prompt,
    PromptTemplate,
    SystemMessageTemplate,
    UserMessageTemplate,
)
from base.services.common.meta import Singleton
from base.services.configuration.mixins import PromptSettingsMixin
from base.services.prompt.decorator import register_prompt
from base.services.prompt.loader import PromptLoader
from base.services.prompt.protocol import PromptTemplateStore
from base.services.prompt.service import PromptService


@pytest.fixture(autouse=True)
def _fresh_registry() -> Iterator[None]:
    """`PromptLoader` is a process-wide singleton; give each test an empty one."""
    Singleton.purge()
    yield
    Singleton.purge()


def template(text: str = "You are terse.") -> PromptTemplate:
    return PromptTemplate.from_messages([SystemMessageTemplate(content=text)])


def prompt(key: str = "answer", text: str = "You are terse.") -> Prompt:
    return Prompt(
        key=key,
        name=key.title(),
        description=f"The {key} prompt.",
        template=template(text),
    )


class StaticStore:
    """A `PromptTemplateStore` that answers with whatever it was handed."""

    def __init__(self, templates: dict[str, PromptTemplate]) -> None:
        self.templates = templates
        self.reads = 0

    async def aget_active_templates(self) -> dict[str, PromptTemplate]:
        self.reads += 1
        return dict(self.templates)


# Registration


def test_a_registered_prompt_is_retrievable_by_key() -> None:
    loader = PromptLoader()
    loader.register(prompt())

    assert loader.get("answer").description == "The answer prompt."
    assert loader.list_keys() == ["answer"]


def test_an_unregistered_key_says_which_one_is_missing() -> None:
    with pytest.raises(KeyError, match="Prompt not found: nobody"):
        PromptLoader().get("nobody")


def test_registering_the_same_key_twice_replaces_it() -> None:
    loader = PromptLoader()
    loader.register(prompt(text="first"))
    loader.register(prompt(text="second"))

    assert loader.list_keys() == ["answer"]
    assert loader.get("answer").template.messages[0].content == "second"


def test_the_decorator_registers_at_import_time_and_returns_the_prompt() -> None:
    @register_prompt
    def answer() -> Prompt:
        return prompt(key="answer")

    assert isinstance(answer, Prompt)
    assert PromptLoader().get("answer") is answer


def test_the_decorator_refuses_a_function_that_returns_something_else() -> None:
    with pytest.raises(TypeError, match="must return a Prompt"):

        @register_prompt
        def broken() -> Prompt:
            return "not a prompt"  # type: ignore[return-value]


def test_the_decorator_refuses_a_key_that_does_not_match_the_function() -> None:
    with pytest.raises(ValueError, match="must match function name"):

        @register_prompt
        def mismatched() -> Prompt:
            return prompt(key="something_else")


# Template packages


def test_templates_are_imported_from_the_packages_a_service_names(
    tmp_path: Any,
) -> None:
    package = _template_package(tmp_path, "scratch_templates", "greeting")
    PromptLoader().load_templates([package])

    assert PromptLoader().get("greeting").key == "greeting"


def test_templates_are_only_loaded_once(tmp_path: Any) -> None:
    package = _template_package(tmp_path, "scratch_templates_twice", "greeting")
    loader = PromptLoader()
    loader.load_templates([package])
    loader._templates_loaded = True

    loader._prompts.clear()
    loader.load_templates([package])

    assert loader.list_keys() == []


# Initialization and stored overrides


async def test_initialize_loads_code_templates_and_ignores_a_store_by_default(
    make_settings: Any,
) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=False)
    loader = PromptLoader()
    loader.register(prompt(text="from code"))
    store = StaticStore({"answer": template("from the database")})

    await loader.initialize(store=store)

    assert store.reads == 0
    assert loader.get("answer").template.messages[0].content == "from code"


async def test_a_stored_template_overrides_the_one_in_code(
    make_settings: Any,
) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=True)
    loader = PromptLoader()
    loader.register(prompt(text="from code"))
    store = StaticStore({"answer": template("from the database")})

    await loader.initialize(store=store)
    await loader.shutdown()

    assert store.reads == 1
    assert loader.get("answer").template.messages[0].content == "from the database"


async def test_an_override_keeps_the_name_and_description_from_code(
    make_settings: Any,
) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=True)
    loader = PromptLoader()
    loader.register(prompt())
    await loader.initialize(store=StaticStore({"answer": template("replaced")}))
    await loader.shutdown()

    stored = loader.get("answer")
    assert stored.name == "Answer"
    assert stored.description == "The answer prompt."


async def test_a_stored_key_nobody_declared_in_code_is_skipped(
    make_settings: Any,
) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=True)
    loader = PromptLoader()
    loader.register(prompt())
    await loader.initialize(
        store=StaticStore({"answer": template("replaced"), "ghost": template("x")})
    )
    await loader.shutdown()

    assert loader.list_keys() == ["answer"]


async def test_reloading_counts_only_the_overrides_it_applied(
    make_settings: Any,
) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=True)
    loader = PromptLoader()
    loader.register(prompt())
    loader._store = StaticStore(
        {"answer": template("replaced"), "ghost": template("x")}
    )

    assert await loader._areload() == 1


async def test_reloading_without_a_store_applies_nothing() -> None:
    assert await PromptLoader()._areload() == 0


async def test_opting_in_without_a_store_still_serves_the_code_templates(
    make_settings: Any,
) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=True)
    loader = PromptLoader()
    loader.register(prompt(text="from code"))

    await loader.initialize(store=None)

    assert loader.get("answer").template.messages[0].content == "from code"
    assert loader._watcher is None


async def test_shutdown_stops_following_reloads(make_settings: Any) -> None:
    make_settings(PromptSettingsMixin, load_prompts_from_db=True)
    loader = PromptLoader()
    loader.register(prompt())

    await loader.initialize(store=StaticStore({}))
    assert loader._watcher is not None

    await loader.shutdown()
    assert loader._watcher is None


async def test_shutdown_without_a_watcher_is_harmless() -> None:
    loader = PromptLoader()

    await loader.shutdown()

    assert loader._watcher is None


def test_the_reload_channel_is_namespaced_by_service(make_settings: Any) -> None:
    make_settings(PromptSettingsMixin, service_name="chat_api")
    assert PromptLoader()._channel() == "chat_api:prompt_reload"


# Protocol conformance


async def test_the_registry_needs_nothing_from_a_store_but_the_protocol_method() -> (
    None
):
    class MinimalStore:
        def __init__(self) -> None:
            self.calls = 0

        async def aget_active_templates(self) -> dict[str, PromptTemplate]:
            self.calls += 1
            return {"answer": template("from the database")}

    store: PromptTemplateStore = MinimalStore()
    loader = PromptLoader()
    loader.register(prompt())
    loader._store = store

    assert await loader._areload() == 1
    assert store.calls == 1  # type: ignore[attr-defined]
    assert loader.get("answer").template.messages[0].content == "from the database"


# PromptService


def test_the_service_reads_through_the_same_registry() -> None:
    PromptLoader().register(prompt())
    service = PromptService()

    assert service.get("answer").key == "answer"
    assert service.get_raw("answer") == template()
    assert service.list_keys() == ["answer"]


def test_the_service_renders_a_registered_prompt() -> None:
    PromptLoader().register(
        Prompt(
            key="answer",
            name="Answer",
            description="d",
            template=PromptTemplate.from_messages(
                [UserMessageTemplate(content="{question}")]
            ),
        )
    )

    rendered = PromptService().render("answer", question="why")

    assert [m.content for m in rendered.messages] == ["why"]


def test_the_service_formats_a_string_prompt() -> None:
    PromptLoader().register(
        Prompt(
            key="note",
            name="Note",
            description="d",
            template=PromptTemplate.from_template("Hello {name}."),
        )
    )

    assert PromptService().format("note", name="Ada") == "Hello Ada."


def _template_package(tmp_path: Any, name: str, key: str) -> types.ModuleType:
    """Build an importable one-module template package under `tmp_path`."""
    directory = tmp_path / name
    directory.mkdir()
    (directory / "__init__.py").write_text("")
    (directory / "greeting.py").write_text(
        "from base.models.view_models.prompt import (\n"
        "    Prompt,\n"
        "    PromptTemplate,\n"
        "    SystemMessageTemplate,\n"
        ")\n"
        "from base.services.prompt.decorator import register_prompt\n"
        "\n"
        "\n"
        "@register_prompt\n"
        f"def {key}() -> Prompt:\n"
        "    return Prompt(\n"
        f'        key="{key}",\n'
        '        name="Greeting",\n'
        '        description="d",\n'
        "        template=PromptTemplate.from_messages(\n"
        '            [SystemMessageTemplate(content="Hello.")]\n'
        "        ),\n"
        "    )\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))
