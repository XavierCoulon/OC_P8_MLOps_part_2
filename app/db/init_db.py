"""Database initialization script."""

import logging

from app.db.database import _get_engine
from app.db.models import Base

logger = logging.getLogger(__name__)


def init_db():
    """Create all database tables if they don't exist."""
    try:
        engine = _get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise


if __name__ == "__main__":
    init_db()
