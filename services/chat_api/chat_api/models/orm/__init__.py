"""Declarative root for chat_api."""

from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

from base.models.orm import SerializableMixin, TimestampMixin


class ChatBase(MappedAsDataclass, DeclarativeBase, SerializableMixin):
    """Declarative root for every table chat_api owns."""


__all__ = ["ChatBase", "SerializableMixin", "TimestampMixin"]
