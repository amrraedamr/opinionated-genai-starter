import logging
import logging.config
from typing import Any

from base.models.definitions.exceptions import ExceptionCategory
from base.services.common.meta import Singleton
from base.services.logging import LogMessage
from base.services.logging.config import logging_config


class ConsoleLoggerService(metaclass=Singleton):
    def __init__(self) -> None:
        try:
            from base.services.configuration import get_settings

            settings = get_settings()
            config = logging_config(
                environment=settings.environment, level=settings.log_level
            )
        except RuntimeError:
            config = logging_config()
        logging.config.dictConfig(config)
        self._logger = logging.getLogger()

    def info(self, message: LogMessage, *args: Any) -> None:
        self._logger.info(message, *args)

    def error(self, message: LogMessage, *args: Any, **kwargs: Any) -> None:
        self._logger.error(self._format_error(message, **kwargs), *args)

    def exception(self, message: LogMessage, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(self._format_error(message, **kwargs), *args)

    def warning(self, message: LogMessage, *args: Any) -> None:
        self._logger.warning(message, *args)

    def debug(self, message: LogMessage, *args: Any) -> None:
        self._logger.debug(message, *args)

    @staticmethod
    def _format_error(
        message: LogMessage,
        *,
        status: str | int | None = None,
        category: ExceptionCategory | None = ExceptionCategory.INTERNAL,
        location: str | None = None,
        tag: str = "Internal",
    ) -> str:
        return f"[Code: {status}] [Category: {category}] [Location: {location}] [Tag: {tag}] {message}"
