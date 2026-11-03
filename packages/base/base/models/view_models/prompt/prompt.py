from typing import Any

from base.models.view_models import BaseModel
from base.models.view_models.prompt.prompt_template import PromptTemplate
from base.models.view_models.prompt.rendered_prompt import RenderedPrompt


class Prompt(BaseModel):
    key: str
    name: str
    description: str
    template: PromptTemplate

    def render(self, **kwargs: Any) -> RenderedPrompt:
        rendered = self.template.render_prompt(**kwargs)
        return RenderedPrompt(
            key=self.key,
            name=self.name,
            description=self.description,
            variables=rendered.variables,
            instructions=rendered.instructions,
            messages=rendered.messages,
        )
