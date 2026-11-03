"""Declarative root for identity."""

from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass

from base.models.orm import SerializableMixin, TimestampMixin


class IdentityBase(MappedAsDataclass, DeclarativeBase, SerializableMixin):
    pass


__all__ = ["IdentityBase", "TimestampMixin"]
