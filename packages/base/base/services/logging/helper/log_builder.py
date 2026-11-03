import datetime
from typing import Any


class LazyLogMessage:
    """Defers string formatting until __str__ is called."""

    __slots__ = ("_args", "_fmt")

    def __init__(self, fmt: str, *args: Any) -> None:
        self._fmt: str = fmt
        self._args: tuple[Any, ...] = args

    def __str__(self) -> str:
        return self._fmt % self._args if self._args else self._fmt


class LogMessage:
    def __init__(self) -> None:
        self.tags: list = []
        self.message: str = ""
        self.status: int | str = ""
        self.duration: float = 0.0

    def to_dict(self):
        tags = [tag.value if hasattr(tag, "value") else tag for tag in self.tags]

        return {
            "status": self.status,
            "tags": tags,
            "message": self.message,
            "timestamp": datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "duration": self.duration,
        }

    def to_message(self):
        log_dict = self.to_dict()
        return f"{log_dict['timestamp']} - {log_dict['status']} - {log_dict['message']} - {log_dict['tags']} - {log_dict['duration']}"


class LogMessageBuilder:
    def __init__(self):
        self.log_message = LogMessage()

    def tags(self, tags):
        self.log_message.tags.extend(tags)
        return self

    def message(self, message):
        self.log_message.message = message
        return self

    def status(self, status):
        self.log_message.status = status
        return self

    def duration(self, duration):
        self.log_message.duration = duration
        return self

    def build(self) -> LogMessage:
        return self.log_message
