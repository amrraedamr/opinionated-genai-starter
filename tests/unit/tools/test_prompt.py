"""`tools/prompt.py`: reading placeholders out of a template, and filling them.

Templates are `str.format`-style and are rendered with keyword arguments only,
so a positional placeholder is a template a renderer could never fill — caught
when the template is read rather than when a request renders it.
"""

import pytest

from base.tools.prompt import extract_template_variables, safe_format


def test_a_template_with_no_placeholders_declares_no_variables() -> None:
    assert extract_template_variables("plain text") == set()


def test_each_named_placeholder_is_reported_once() -> None:
    assert extract_template_variables("{greeting}, {name}. {greeting} again.") == {
        "greeting",
        "name",
    }


def test_an_attribute_access_reports_only_its_root() -> None:
    assert extract_template_variables("{user.profile.name}") == {"user"}


def test_an_index_access_reports_only_its_root() -> None:
    assert extract_template_variables("{items[0]}") == {"items"}
    assert extract_template_variables("{items[0].name}") == {"items"}


def test_a_format_spec_does_not_become_a_variable() -> None:
    assert extract_template_variables("{amount:.2f}") == {"amount"}


def test_an_escaped_brace_is_not_a_placeholder() -> None:
    assert extract_template_variables("{{literal}}") == set()


def test_an_empty_placeholder_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="Positional placeholders"):
        extract_template_variables("Hello {}")


def test_a_numbered_placeholder_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="Positional placeholders"):
        extract_template_variables("Hello {0}")


def test_formatting_fills_the_variables_it_was_given() -> None:
    assert safe_format("Hello {name}", {"name": "Ada"}) == "Hello Ada"


def test_a_string_with_no_brace_is_passed_through_untouched() -> None:
    assert safe_format("call_abc123", {}) == "call_abc123"


def test_an_identifier_with_no_brace_survives_even_with_no_variables() -> None:
    """The reason the guard exists: a tool_call_id must not crash a render."""
    assert safe_format("tool-call-id-with-no-braces", {}) == (
        "tool-call-id-with-no-braces"
    )


def test_a_missing_variable_is_reported_rather_than_left_in_place() -> None:
    with pytest.raises(KeyError):
        safe_format("Hello {name}", {})


def test_extra_variables_are_ignored() -> None:
    assert safe_format("Hello {name}", {"name": "Ada", "unused": 1}) == "Hello Ada"


def test_an_empty_string_is_passed_through() -> None:
    assert safe_format("", {"name": "Ada"}) == ""
