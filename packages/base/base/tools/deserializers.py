import importlib
import inspect
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from json import JSONDecoder
from typing import Any, ClassVar
from uuid import UUID

FLAGS = re.VERBOSE | re.MULTILINE | re.DOTALL
WHITESPACE = re.compile(r"[ \t\n\r]*", FLAGS)

# Module/class import cache — avoids repeated importlib calls
_import_cache: dict[str, type] = {}
_init_params_cache: dict[str, set[str]] = {}


def _resolve_class(class_path: str) -> type:
    """Cached class resolution. importlib.import_module is surprisingly expensive when called repeatedly."""
    if class_path not in _import_cache:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        _import_cache[class_path] = getattr(module, class_name)
    return _import_cache[class_path]


def _get_init_params(cls: type, class_path: str) -> set[str]:
    if class_path not in _init_params_cache:
        try:
            sig = inspect.signature(cls.__init__)
            _init_params_cache[class_path] = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            _init_params_cache[class_path] = set()
    return _init_params_cache[class_path]


def decode_datetime(obj: str) -> datetime:
    return datetime.fromisoformat(obj)


def decode_class_by_depopulation(obj: dict, _class: str) -> Any:
    cls = _resolve_class(_class)
    decoded_values = {
        field_name: ExpansiveDecoder.process_decoded_data(field_value)
        if ExpansiveDecoder.should_be_decoded(field_value)
        else field_value
        for field_name, field_value in obj.items()
    }
    try:
        return cls(**decoded_values)
    except (TypeError, ValueError):
        init_params = _get_init_params(cls, _class)
        init_kwargs = {k: v for k, v in decoded_values.items() if k in init_params}
        extra_kwargs = {k: v for k, v in decoded_values.items() if k not in init_params}

        clazz = cls(**init_kwargs)
        for field_name, field_value in extra_kwargs.items():
            if hasattr(clazz, field_name):
                setattr(clazz, field_name, field_value)
        return clazz


def decode_uuid(obj: str) -> UUID:
    return UUID(obj)


def decode_type(class_path: str) -> type:
    return _resolve_class(class_path)


class ExpansiveDecoder(JSONDecoder):
    _decoders: ClassVar[dict[str, Callable]] = {
        "datetime": decode_datetime,
        "dto": decode_class_by_depopulation,
        "app_model": decode_class_by_depopulation,
        "uuid": decode_uuid,
        "type": decode_type,
    }

    def decode(self, s: str, _w: Callable[..., Any] = WHITESPACE.match) -> Any:
        data = super().decode(s, _w)
        if self.should_be_decoded(data):
            result = self.process_decoded_data(data)
            return result

        logging.info(f"Defaulting to inherent deserializer for {type(data)}")
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return data

    @classmethod
    def process_decoded_data(cls, data: Any) -> Any:
        if isinstance(data, list):
            return [
                cls.process_decoded_data(item)
                if isinstance(item, (dict, list, tuple))
                else item
                for item in data
            ]
        if isinstance(data, tuple):
            return tuple(
                cls.process_decoded_data(item)
                if isinstance(item, (dict, list, tuple))
                else item
                for item in data
            )
        if isinstance(data, dict):
            _type = data.get("_type")
            if _type is not None:
                value = data.get("value")
                if not value:
                    return None

                decoder = cls._decoders.get(_type)
                if decoder is not None:
                    # Build kwargs from remaining keys (excluding _type and value)
                    extra = {
                        k: v for k, v in data.items() if k != "_type" and k != "value"
                    }
                    return decoder(value, **extra)

                logging.info(
                    f"Failed to deserialize for {_type} defaulting to inherent deserializer"
                )

            return {
                key: cls.process_decoded_data(value)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in data.items()
            }
        return data

    @classmethod
    def should_be_decoded(cls, data: Any) -> bool:
        return isinstance(data, (dict, tuple, list))


def deserialize(obj: str, **kwargs) -> Any:
    stripped = obj.strip()
    if len(stripped) < 10:
        try:
            return int(stripped)
        except ValueError:
            pass
    return json.loads(obj, cls=ExpansiveDecoder, **kwargs)
