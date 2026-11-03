from enum import Enum

from base.services.logging.helper.log_builder import LazyLogMessage

LogMessage = Enum | str | LazyLogMessage
