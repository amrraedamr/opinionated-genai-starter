from base.models.definitions.canned_messages import CannedMessages


class FlowAbortError(Exception):
    """Raised by :meth:`AbstractAgent.arun` when an agent fails fatally.

    Carries the user-facing :class:`CannedMessages` to surface. Caught
    at the pipeline boundary; never per-agent. The pipeline maps the
    canned message into ``flow_variables.final_answer`` and the request
    completes normally from the client's point of view.
    """

    def __init__(
        self,
        canned_message: CannedMessages,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(canned_message.name)
        self.canned_message: CannedMessages = canned_message
        self.cause: Exception | None = cause
