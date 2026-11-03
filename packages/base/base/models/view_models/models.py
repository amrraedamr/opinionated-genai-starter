from enum import StrEnum


class OpenAIModels(StrEnum):
    """OpenAI text generation and embedding models available via the API.

    Includes only non-specialized models: chat/completion models, reasoning
    models (o-series), and embeddings. Excludes codex, deep research, image,
    audio, realtime, computer-use, and search-focused models.

    Legacy models (GPT-4, GPT-4o, GPT-4.1, o-series pre-4) are retained
    because API access remains even after ChatGPT retirement. Remove from
    this enum only when OpenAI deprecates them from the API itself.
    """

    # GPT-5.4 — current flagship
    GPT_5_4 = "gpt-5.4"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"
    GPT_5_4_PRO = "gpt-5.4-pro"

    # GPT-5.3
    GPT_5_3 = "gpt-5.3"

    # GPT-5.2
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_MINI = "gpt-5.2-mini"
    GPT_5_2_NANO = "gpt-5.2-nano"
    GPT_5_2_PRO = "gpt-5.2-pro"

    # GPT-5.1
    GPT_5_1 = "gpt-5.1"
    GPT_5_1_MINI = "gpt-5.1-mini"
    GPT_5_1_NANO = "gpt-5.1-nano"

    # GPT-5
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_PRO = "gpt-5-pro"

    # GPT-4.1 family (legacy, still in API)
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"

    # GPT-4o family (legacy, still in API)
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # GPT-4 family (legacy)
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"

    # GPT-3.5 (legacy)
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # O-series reasoning models
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PRO = "o1-pro"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O3_PRO = "o3-pro"
    O4_MINI = "o4-mini"

    # Embeddings
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
