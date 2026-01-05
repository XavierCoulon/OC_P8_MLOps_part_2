"""Database configuration and session management."""

from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config.settings import settings

# Base class for ORM models (no DB dependency)
Base = declarative_base()

# Engine and session factory (created lazily on first use)
_engine = None
_SessionLocal = None


def _get_engine():
    """Get or create database engine (lazy initialization)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            echo=settings.debug,
        )
    return _engine


def _get_session_local():
    """Get or create session factory (lazy initialization)."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = _get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )
    return _SessionLocal


# Expose as module-level for backwards compatibility
def engine():
    """Get database engine."""
    return _get_engine()


def SessionLocal():
    """Get session factory."""
    return _get_session_local()


def get_session() -> Generator[Session, None, None]:
    """Get database session for dependency injection.

    Yields:
        Database session
    """
    db = _get_session_local()()
    try:
        yield db
    finally:
        db.close()


# Annotated dependency for cleaner route signatures
SessionDep = Annotated[Session, Depends(get_session)]


def create_db_and_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=_get_engine())
