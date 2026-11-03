import string
from typing import Any


def extract_template_variables(template: str) -> set[str]:
    """Extract root variable names from a str.format-style template.

    Raises on positional placeholders (e.g. ``{}`` or ``{0}``) since the
    renderer uses keyword arguments exclusively.
    """
    formatter = string.Formatter()
    variables: set[str] = set()

    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError(
                f"Positional placeholders are not supported in templates: {template!r}"
            )
        root_name = field_name.split(".", 1)[0].split("[", 1)[0]
        variables.add(root_name)

    return variables


def safe_format(value: str, variables: dict[str, Any]) -> str:
    """Format only when the string contains placeholders, otherwise pass through.

    Prevents literal braces in opaque identifiers (tool_call_id, name) from
    crashing ``str.format``.
    """
    if "{" not in value:
        return value
    return value.format(**variables)
