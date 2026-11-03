import contextvars
import logging
from abc import ABCMeta
from typing import Any, ClassVar, TypeVar, cast
from uuid import UUID, uuid4

from pydantic._internal._model_construction import ModelMetaclass

from base.tools.hashing import hash_args

T = TypeVar("T")

# Context vars for scope tracking

scope_id: contextvars.ContextVar[UUID] = contextvars.ContextVar("scope_id")
processor_id: contextvars.ContextVar[UUID] = contextvars.ContextVar("processor_id")


def _get_scope_id(processor: bool = False) -> UUID:
    context_var = processor_id if processor else scope_id
    try:
        return context_var.get()
    except LookupError:
        new_id = uuid4()
        context_var.set(new_id)
        return new_id


# Singleton


class Singleton(type):
    """One instance per type across the entire application."""

    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super().__call__(*args, **kwargs)
            logging.info("STORE: Singleton %s Initialized", cls.__name__)
        return cast(T, Singleton._instances[cls])

    def replace(cls: type[T], instance: T) -> T:
        Singleton._instances[cls] = instance
        return instance

    @staticmethod
    def purge() -> None:
        Singleton._instances.clear()


class SingletonABC(Singleton, ABCMeta):
    pass


class SingletonSettings(Singleton, ModelMetaclass):
    pass


# Stored (Scoped)


class Stored(type):
    """One instance per context scope (tied to scope_id / processor_id).

    Set ``_processor: ClassVar[bool] = True`` on a class to bind it
    to the processor scope instead of the default request scope.
    """

    _scoped_instances: ClassVar[dict[UUID | None, dict[type, Any]]] = {}

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        is_processor: bool = getattr(cls, "_processor", False)
        scope_uuid = _get_scope_id(processor=is_processor)
        scope_instances = Stored._scoped_instances.setdefault(scope_uuid, {})

        if cls not in scope_instances:
            scope_instances[cls] = super().__call__(*args, **kwargs)
            logging.info(
                "STORE: Scoped %s Initialized (scope=%s)", cls.__name__, scope_uuid
            )
        return cast(T, scope_instances[cls])

    def replace(cls: type[T], instance: T) -> T:
        is_processor: bool = getattr(cls, "_processor", False)
        scope_uuid = _get_scope_id(processor=is_processor)
        scope_instances = Stored._scoped_instances.setdefault(scope_uuid, {})
        scope_instances[cls] = instance
        return instance

    @staticmethod
    def delete_scope(scope_uuid: UUID | None = None, processor: bool = False) -> None:
        if scope_uuid is None:
            context_var = processor_id if processor else scope_id
            try:
                scope_uuid = context_var.get()
            except LookupError:
                return
        Stored._scoped_instances.pop(scope_uuid, None)

    @staticmethod
    def purge() -> None:
        Stored._scoped_instances.clear()


class StoredABC(Stored, ABCMeta):
    pass


# KeyedSingleton


class KeyedSingleton(type):
    """One instance per (type, key) pair.

    The key can be provided explicitly via ``_key``, or derived
    automatically from the constructor arguments (serialized + hashed).

    Usage:
        class ConnPool(metaclass=KeyedSingleton): ...

        pool = ConnPool(_key="primary")
        replica = ConnPool(_key="replica")
        found = ConnPool.get("primary")         # None if absent
        ConnPool.replace("primary", new_pool)

        # Auto-keyed — same args produce the same instance:
        svc = OpenAIService(config=model_config)
    """

    _keyed_instances: ClassVar[dict[tuple[type, str], Any]] = {}

    def __call__(cls: type[T], *args: Any, _key: str = "", **kwargs: Any) -> T:
        if not _key:
            _key = hash_args(*args, **kwargs)

        lookup = (cls, _key)

        if lookup not in KeyedSingleton._keyed_instances:
            KeyedSingleton._keyed_instances[lookup] = super().__call__(*args, **kwargs)
            logging.info("STORE: KeyedSingleton %s[%s] Initialized", cls.__name__, _key)
        return cast(T, KeyedSingleton._keyed_instances[lookup])

    def get(cls: type[T], key: str) -> T | None:
        return cast(T | None, KeyedSingleton._keyed_instances.get((cls, key)))

    def replace(cls: type[T], key: str, instance: T) -> T:
        KeyedSingleton._keyed_instances[(cls, key)] = instance
        return instance

    @staticmethod
    def purge() -> None:
        KeyedSingleton._keyed_instances.clear()


class KeyedSingletonABC(KeyedSingleton, ABCMeta):
    pass


# ContextManager


class ContextManager(type):
    def __new__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> type:
        if "__enter__" not in dct:
            dct["__enter__"] = cls.default_enter
        if "__exit__" not in dct:
            dct["__exit__"] = cls.default_exit
        return super().__new__(cls, name, bases, dct)

    def default_enter(self: Any) -> Any:
        return self

    def default_exit(self: Any, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass


class ContextManagerSingleton(ContextManager, Singleton):
    pass


class ContextManagerABC(ContextManager, ABCMeta):
    pass


class ContextManagerStored(ContextManager, Stored):
    pass


# NonInitialize


class NonInitialize(type):
    def __call__(cls) -> type:
        return cls


# Global purge


def purge_all() -> None:
    Singleton.purge()
    Stored.purge()
    KeyedSingleton.purge()
