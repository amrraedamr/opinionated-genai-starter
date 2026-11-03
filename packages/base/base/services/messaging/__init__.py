"""Reliable delivery either side of the broker.

`outbox` makes a write and its event commit together, then publishes what
committed. `inbox` absorbs the duplicates that at-least-once delivery
produces. Each service owns the tables; these drive them.
"""

from base.services.messaging.inbox import EventHandler, InboxProcessor, InboxState
from base.services.messaging.outbox import OutboxDispatcher, add_event
from base.services.messaging.protocol import InboxRow, OutboxRow

__all__ = [
    "EventHandler",
    "InboxProcessor",
    "InboxRow",
    "InboxState",
    "OutboxDispatcher",
    "OutboxRow",
    "add_event",
]
