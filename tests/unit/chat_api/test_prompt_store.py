"""`PromptStore` is where a deployed prompt beats the one in the source tree.

It holds one active version per key and hands the loader a template for each,
so the rules worth pinning are that a row parses back into the template it was
stored from and that a key with no active row simply is not there.
"""

from __future__ import annotations

import pytest

from base.models.view_models.prompt import (
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessageTemplate,
    UserMessageTemplate,
)
from chat_api.models.orm.prompt import ActivePromptDTO, PromptDTO
from chat_api.stores.prompt import PromptStore


@pytest.fixture
def store(tables: None) -> PromptStore:
    return PromptStore()


def template(text: str = "You are terse.") -> PromptTemplate:
    return PromptTemplate.from_messages(
        [
            SystemMessageTemplate(content=text),
            MessagesPlaceholder(variable_name="history"),
            UserMessageTemplate(content="{user_query}"),
        ]
    )


def activate(store: PromptStore, key: str, text: str, version: int = 1) -> PromptDTO:
    """Store a prompt version and make it the active one for its key."""
    prompt = store._sql_service.upsert_row(
        PromptDTO(key=key, template=template(text).jsonify(), version=version)
    )
    store._sql_service.upsert_row(ActivePromptDTO(key=key, prompt_id=prompt.id))
    return prompt


def test_a_store_with_nothing_active_hands_back_nothing(store: PromptStore) -> None:
    assert store.get_active_templates() == {}


async def test_an_async_store_with_nothing_active_hands_back_nothing(
    store: PromptStore,
) -> None:
    assert await store.aget_active_templates() == {}


def test_an_active_row_parses_back_into_the_template_it_was_stored_from(
    store: PromptStore,
) -> None:
    activate(store, "answer", "You are terse.")

    templates = store.get_active_templates()
    assert list(templates) == ["answer"]
    assert templates["answer"].jsonify() == template("You are terse.").jsonify()


def test_a_template_read_from_the_store_renders_what_it_was_written_as(
    store: PromptStore,
) -> None:
    activate(store, "answer", "You are terse.")

    rendered = store.get_active_templates()["answer"].render_prompt(
        history=[], user_query="Capital of France?"
    )
    assert [i.content for i in rendered.instructions] == ["You are terse."]
    assert [m.content for m in rendered.messages] == ["Capital of France?"]


async def test_an_async_read_parses_back_into_the_same_template(
    store: PromptStore,
) -> None:
    activate(store, "answer", "You are terse.")

    templates = await store.aget_active_templates()
    assert templates["answer"].jsonify() == template("You are terse.").jsonify()


def test_every_active_key_is_returned(store: PromptStore) -> None:
    activate(store, "answer", "A")
    activate(store, "route", "B")

    assert set(store.get_active_templates()) == {"answer", "route"}


def test_a_version_nobody_activated_is_not_returned(store: PromptStore) -> None:
    store._sql_service.upsert_row(
        PromptDTO(key="draft", template=template("draft").jsonify(), version=2)
    )
    assert store.get_active_templates() == {}


def test_pointing_a_key_at_another_version_changes_what_it_renders(
    store: PromptStore,
) -> None:
    activate(store, "answer", "old")
    newer = store._sql_service.upsert_row(
        PromptDTO(key="answer", template=template("new").jsonify(), version=2)
    )
    active = store.edit(
        "answer", ActivePromptDTO(key="answer", prompt_id=newer.id), "key"
    )
    assert active is not None
    assert store.get_active_templates()["answer"].jsonify() == template("new").jsonify()


def test_a_deleted_activation_stops_being_returned(store: PromptStore) -> None:
    activate(store, "answer", "A")
    store.delete("answer", "key")

    assert store.get_active_templates() == {}


def test_the_key_a_row_is_filed_under_is_protected_from_edits() -> None:
    assert "key" in PromptStore.protected_fields
    assert PromptStore.search_fields == ("key",)


def test_the_store_is_one_instance_per_process(store: PromptStore) -> None:
    assert PromptStore() is store
