import json
import logging
from collections.abc import Callable
from datetime import datetime
from inspect import isclass
from json import JSONEncoder
from typing import Any, ClassVar
from uuid import UUID

from fastapi import UploadFile
from pydantic import BaseModel

try:  # `sql` extra. Optional on purpose — this module is core.
    from sqlalchemy.orm import DeclarativeBase
except ModuleNotFoundError:  # pragma: no cover
    DeclarativeBase = None  # type: ignore[assignment, misc]


def encode_datetime(obj: datetime) -> dict[str, Any]:
    return {"_type": "datetime", "value": obj.isoformat()}


def encode_dto(obj: Any) -> dict[str, Any]:
    """Encode any service's declarative model, whatever its declarative root."""
    return {
        "_type": "dto",
        "_class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "value": {k: v for k, v in obj.__dict__.items() if not k.startswith("_")},
    }


def encode_base_model(obj: BaseModel) -> dict[str, Any]:
    return {
        "_type": "base_model",
        "_class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
        "value": obj.model_dump(mode="python"),
    }


def encode_uuid(obj: UUID) -> dict[str, Any]:
    return {"_type": "uuid", "value": str(obj)}


def encode_type(obj: type) -> dict[str, Any]:
    """Convert a type (class) to a string representation."""
    return {
        "_type": "type",
        "value": f"{obj.__module__}.{obj.__qualname__}",
    }


def encode_upload_file(obj: UploadFile) -> dict[str, Any]:
    return {
        "_type": "UploadFile",
        "filename": obj.filename,
        "content_type": obj.content_type,
    }


class ExpansiveEncoder(JSONEncoder):
    encoders: ClassVar[dict[type, Callable[[Any], dict[str, Any]]]] = {
        datetime: encode_datetime,
        BaseModel: encode_base_model,
        UUID: encode_uuid,
        type: encode_type,
        UploadFile: encode_upload_file,
    }

    # Only registered when the `sql` extra is installed; a service with no
    # SQLAlchemy has no declarative models to encode.
    if DeclarativeBase is not None:
        encoders[DeclarativeBase] = encode_dto

    def default(self, o: Any) -> Any:
        for cls, encoder in self.encoders.items():
            if isinstance(o, cls):
                return encoder(o)
        logging.info(
            f"Defaulting to inherent serializer for {o.__class__.__name__ if isclass(o) else type(o)}"
        )
        return super().default(o)


def serialize(obj: Any, **kwargs: Any) -> str:
    return (
        json.dumps(obj, cls=ExpansiveEncoder, **kwargs)
        if not isinstance(obj, str)
        else obj
    )
