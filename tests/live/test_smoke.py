"""Proves the live marker reaches a real provider."""

import pytest

from tests.conftest import credential


@pytest.mark.live
async def test_gateway_answers(make_settings):
    from base.models.definitions.model import ProviderType
    from base.models.view_models.model import ModelConfigurations
    from base.services.configuration.mixins import (
        LLMSettingsMixin,
        OpenAILLMSettingsMixin,
    )
    from base.services.llm.openai import OpenAIService

    make_settings(LLMSettingsMixin, OpenAILLMSettingsMixin)
    service = OpenAIService(
        ModelConfigurations(
            api_host=credential("OPENAI_LLM_ENDPOINT"),
            api_key=credential("OPENAI_LLM_API_KEY"),
            model_name="azure.gpt-5.6-terra",
            api_type=ProviderType.OPENAI,
            api_retries=1,
        )
    )
    from tests.fakes import build_prompt

    result = await service.acomplete(prompt=build_prompt(), max_output_tokens=2048)
    assert result.content
