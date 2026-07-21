from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base. Tables land here starting in Phase 1 (Section 13)."""
