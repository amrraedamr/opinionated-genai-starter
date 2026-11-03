from base.models.view_models.prompt import (
    MessagesPlaceholder,
    Prompt,
    PromptTemplate,
    SystemMessageTemplate,
    UserMessageTemplate,
)
from base.services.prompt.decorator import register_prompt

TEST_SYSTEM_PROMPT = """You are a helpful assistant."""

USER_INPUT_TEMPLATE = """{user_query}"""


@register_prompt
def test_prompt() -> Prompt:
    return Prompt(
        key="test_prompt",
        name="Test Prompt",
        template=PromptTemplate.from_messages(
            [
                SystemMessageTemplate(content=TEST_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                UserMessageTemplate(content=USER_INPUT_TEMPLATE),
            ],
        ),
        description="A simple test prompt with no variables.",
    )
