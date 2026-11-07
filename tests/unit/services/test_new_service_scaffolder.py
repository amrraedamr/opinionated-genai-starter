"""The scaffolder has to pick the SDK the provider actually ships in.

`--llm-provider` chooses two things at once: the settings mixin a generated
service composes, and the `base` extra that installs the SDK behind it. The two
tables are separate, so they can disagree — `gemini` once generated a service
that declared `base[openai]` and could not talk to Gemini at all.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools"


@pytest.fixture
def scaffolder(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Any]:
    """`tools/new_service.py`, writing into a throwaway services directory."""
    sys.path.insert(0, str(TOOLS))
    try:
        import new_service
    finally:
        sys.path.remove(str(TOOLS))

    monkeypatch.setattr(new_service, "ROOT", tmp_path)
    monkeypatch.setattr(new_service, "SERVICES", tmp_path / "services")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
    )
    yield new_service


def generate(scaffolder: Any, tmp_path: pathlib.Path, **kwargs: Any) -> pathlib.Path:
    name = kwargs.pop("name", "billing")
    scaffolder.build(
        name,
        kwargs.pop("storage", "none"),
        kwargs.pop("http", True),
        kwargs.pop("capabilities", ["llm"]),
        None,
        kwargs.pop("llm_provider", "azure-openai"),
    )
    return tmp_path / "services" / name


def declared_extras(directory: pathlib.Path) -> list[str]:
    manifest = tomllib.loads((directory / "pyproject.toml").read_text())
    dependency = manifest["project"]["dependencies"][0]
    return dependency.removeprefix("base[").removesuffix("]").split(",")


# The two tables


def test_both_tables_cover_the_same_four_providers(scaffolder: Any) -> None:
    assert set(scaffolder.MIXINS_FOR_LLM_PROVIDER) == set(
        scaffolder.EXTRAS_FOR_LLM_PROVIDER
    )
    assert set(scaffolder.MIXINS_FOR_LLM_PROVIDER) == {
        "azure-openai",
        "openai",
        "anthropic",
        "gemini",
    }


def test_every_provider_mixin_exists_in_base(scaffolder: Any) -> None:
    import base.services.configuration.mixins as mixins

    for provider, mixin in scaffolder.MIXINS_FOR_LLM_PROVIDER.items():
        assert hasattr(mixins, mixin), f"{provider} names unknown mixin {mixin}"


def test_every_provider_extra_exists_in_base(scaffolder: Any) -> None:
    manifest = tomllib.loads(
        (ROOT / "packages" / "base" / "pyproject.toml").read_text()
    )
    available = set(manifest["project"]["optional-dependencies"])

    for provider, extra in scaffolder.EXTRAS_FOR_LLM_PROVIDER.items():
        assert extra in available, f"{provider} names unknown extra {extra}"


def test_only_the_two_openai_providers_share_an_sdk(scaffolder: Any) -> None:
    extras = scaffolder.EXTRAS_FOR_LLM_PROVIDER

    assert extras["azure-openai"] == extras["openai"] == "openai"
    assert extras["anthropic"] == "anthropic"
    assert extras["gemini"] == "gemini"


def test_each_provider_has_its_own_settings_mixin(scaffolder: Any) -> None:
    mixins = scaffolder.MIXINS_FOR_LLM_PROVIDER

    assert len(set(mixins.values())) == len(mixins)


# What the scaffolder writes


@pytest.mark.parametrize(
    ("provider", "extra"),
    [
        ("azure-openai", "openai"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("gemini", "gemini"),
    ],
)
def test_the_llm_capability_installs_the_sdk_its_provider_ships_in(
    scaffolder: Any, tmp_path: pathlib.Path, provider: str, extra: str
) -> None:
    directory = generate(scaffolder, tmp_path, llm_provider=provider)

    assert extra in declared_extras(directory)


def test_choosing_gemini_does_not_install_openai(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    directory = generate(scaffolder, tmp_path, llm_provider="gemini")

    extras = declared_extras(directory)
    assert "gemini" in extras
    assert "openai" not in extras


def test_choosing_anthropic_does_not_install_openai(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    extras = declared_extras(generate(scaffolder, tmp_path, llm_provider="anthropic"))

    assert "anthropic" in extras
    assert "openai" not in extras


@pytest.mark.parametrize(
    ("provider", "mixin"),
    [
        ("azure-openai", "AzureOpenAILLMSettingsMixin"),
        ("openai", "OpenAILLMSettingsMixin"),
        ("anthropic", "AnthropicSettingsMixin"),
        ("gemini", "GeminiSettingsMixin"),
    ],
)
def test_the_generated_settings_compose_the_providers_mixin(
    scaffolder: Any, tmp_path: pathlib.Path, provider: str, mixin: str
) -> None:
    directory = generate(scaffolder, tmp_path, llm_provider=provider)
    settings = (directory / "billing" / "settings.py").read_text()

    assert mixin in settings
    assert "LLMSettingsMixin" in settings


def test_the_literal_llm_capability_never_reaches_the_dependency_list(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    directory = generate(scaffolder, tmp_path, llm_provider="gemini")

    assert "llm" not in declared_extras(directory)


def test_a_service_without_the_llm_capability_declares_no_sdk(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    directory = generate(scaffolder, tmp_path, capabilities=[])

    extras = declared_extras(directory)
    settings = (directory / "billing" / "settings.py").read_text()
    assert extras == ["server"]
    assert "LLMSettingsMixin" not in settings
    assert "GeminiSettingsMixin" not in settings


def test_other_capabilities_are_passed_through_as_extras(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    directory = generate(
        scaffolder, tmp_path, capabilities=["redis", "llm"], llm_provider="gemini"
    )

    assert set(declared_extras(directory)) == {"server", "redis", "gemini"}


def test_an_http_service_gets_the_server_extra_and_a_serve_entrypoint(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    directory = generate(scaffolder, tmp_path, capabilities=[])
    manifest = (directory / "pyproject.toml").read_text()

    assert "server" in declared_extras(directory)
    assert 'serve = "billing.app:start"' in manifest


def test_a_service_with_no_http_surface_gets_neither(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    directory = generate(scaffolder, tmp_path, http=False, capabilities=[])
    manifest = (directory / "pyproject.toml").read_text()

    assert declared_extras(directory) == [""]
    assert "serve =" not in manifest


def test_generating_over_an_existing_service_is_refused(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    generate(scaffolder, tmp_path, capabilities=[])

    with pytest.raises(SystemExit, match="already exists"):
        generate(scaffolder, tmp_path, capabilities=[])


def test_the_generated_package_is_named_after_the_service(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    directory = generate(scaffolder, tmp_path, name="billing", capabilities=[])

    assert (directory / "billing" / "__init__.py").exists()
    assert (directory / "billing" / "app.py").exists()
    assert (directory / "billing" / "settings.py").exists()


def test_a_provider_outside_the_tables_cannot_be_generated(
    scaffolder: Any, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(KeyError):
        generate(scaffolder, tmp_path, llm_provider="cohere")


def test_the_command_line_refuses_a_provider_the_tables_do_not_know() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "new_service.py"),
            "--name",
            "x",
            "--llm-provider",
            "bogus",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "invalid choice: 'bogus'" in completed.stderr
    for provider in ("azure-openai", "openai", "anthropic", "gemini"):
        assert provider in completed.stderr
    assert not (ROOT / "services" / "x").exists()
