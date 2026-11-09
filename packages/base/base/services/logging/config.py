import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from base.models.definitions.environment import Environment
from base.services.guardrails.sinks import redact_pii
from base.tools.correlation import current_correlation_id

_QUOTED_VALUE = re.compile(
    r"""(?ix)
    (["']?(?:password|token|secret|api[_-]?key|private[_-]?key)["']?\s*[:=]\s*)
    (?:"[^"]*"|'[^']*'|[^\s,;}]+)
    """
)
_AUTHORIZATION = re.compile(
    r"""(?ix)
    (["']?authorization["']?\s*[:=]\s*)
    (?:"[^"]*"|'[^']*'|(?:bearer\s+)?[^\s,;}]+)
    """
)


def redact(value: Any) -> str:
    """Strip credentials, and personal data where a service asked for it.

    Credentials go unconditionally: nothing is gained by logging one and a
    leaked key is an incident on its own. Personal data is the guardrail's
    business and follows whatever that service configured, so the two are not
    the same decision and are not made in the same place.
    """
    redacted = _AUTHORIZATION.sub(r"\1[REDACTED]", str(value))
    redacted = _QUOTED_VALUE.sub(r"\1[REDACTED]", redacted)
    return redact_pii(redacted)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
            "correlation_id": str(current_correlation_id()),
        }
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def logging_config(
    *, environment: Environment = Environment.LOCAL, level: str = "INFO"
) -> dict[str, Any]:
    formatter = "colored" if environment == Environment.LOCAL else "json"
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s[%(asctime)s] %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {"()": "base.services.logging.config.JsonFormatter"},
        },
        "filters": {
            "redaction": {"()": "base.services.logging.config.RedactionFilter"}
        },
        "handlers": {
            "console": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": formatter,
                "filters": ["redaction"],
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"handlers": ["console"], "level": level},
    }


LOGGING_CONFIG = logging_config()
