from collections.abc import Callable

from base.models.view_models.prompt import Prompt
from base.services.prompt.loader import PromptLoader


def register_prompt(func: Callable[[], Prompt]) -> Prompt:
    """
    Decorator that instantiates and registers a Prompt at import time.

    The decorated function is executed immediately, and the returned Prompt
    is registered in the PromptLoader singleton. The function name is then
    replaced with the Prompt instance itself.

    Usage::

        @register_prompt
        def my_prompt() -> Prompt:
            return Prompt(
                key="my_prompt",
                name="My Prompt",
                template=PromptTemplate.from_messages([...]),
                description="...",
            )

        # my_prompt is now a Prompt instance
        my_prompt.key  # "my_prompt"
    """
    prompt = func()

    if not isinstance(prompt, Prompt):
        raise TypeError(
            f"@register_prompt function '{func.__name__}' must return a Prompt, "
            f"got {type(prompt).__name__}"
        )

    # Optional safety: ensure key matches function name
    if prompt.key != func.__name__:
        raise ValueError(
            f"Prompt key '{prompt.key}' must match function name '{func.__name__}' "
            "to avoid registration confusion."
        )

    PromptLoader().register(prompt)
    return prompt
