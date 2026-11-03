from enum import StrEnum


class ExceptionCategory(StrEnum):
    API = "API"
    LLM = "LLM"
    INTERNAL = "INTERNAL"
