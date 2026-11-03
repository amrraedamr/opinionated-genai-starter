from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass


class WorkerBase(MappedAsDataclass, DeclarativeBase):
    pass
