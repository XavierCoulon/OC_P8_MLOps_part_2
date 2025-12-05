"""Database configuration and session management."""

from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config.settings import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    echo=settings.debug,
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for ORM models
Base = declarative_base()


def get_session() -> Generator[Session, None, None]:
    """Get database session for dependency injection.

    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Annotated dependency for cleaner route signatures
SessionDep = Annotated[Session, Depends(get_session)]


def create_db_and_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
