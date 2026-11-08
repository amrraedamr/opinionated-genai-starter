"""Span and attribute names, kept in one place because the conventions move.

OpenTelemetry's GenAI semantic conventions are still changing, so every
attribute name this repository emits is written here once. A convention
release becomes an edit to this file rather than a search across adapters.
"""

from typing import Final

# Resource
SERVICE_NAME: Final = "service.name"

# Correlation, which predates tracing here and stays alongside it
CORRELATION_ID: Final = "correlation.id"
TENANT_ID: Final = "tenant.id"

# GenAI, per the OpenTelemetry semantic conventions
GENAI_SYSTEM: Final = "gen_ai.system"
GENAI_OPERATION: Final = "gen_ai.operation.name"
GENAI_REQUEST_MODEL: Final = "gen_ai.request.model"
GENAI_RESPONSE_MODEL: Final = "gen_ai.response.model"
GENAI_RESPONSE_FINISH_REASON: Final = "gen_ai.response.finish_reasons"
GENAI_USAGE_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
GENAI_USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
GENAI_USAGE_CACHE_READ_TOKENS: Final = "gen_ai.usage.cache_read_input_tokens"
GENAI_USAGE_CACHE_WRITE_TOKENS: Final = "gen_ai.usage.cache_creation_input_tokens"
GENAI_USAGE_COST: Final = "gen_ai.usage.cost"

# Content, emitted only where a service opted in
GENAI_PROMPT: Final = "gen_ai.prompt"
GENAI_COMPLETION: Final = "gen_ai.completion"

# Messaging
MESSAGING_DESTINATION: Final = "messaging.destination.name"
MESSAGING_MESSAGE_ID: Final = "messaging.message.id"
MESSAGING_OPERATION: Final = "messaging.operation.name"
