"""Pytest configuration and shared fixtures."""
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import Settings
from app.db.database import get_session
from app.db.models import Base
from app.main import app


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with SQLite database."""
    return Settings(
        app_name="Rugby MLOps Test",
        app_version="0.1.0",
        debug=True,
        api_prefix="/api/v1",
        api_key="test-api-key-12345",
        hf_repo_id="XavierCoulon/rugby-kicks-model",
        database_url="sqlite:///:memory:",
    )


@pytest.fixture(scope="session")
def test_db_engine(test_settings):
    """Create test database engine."""
    engine = create_engine(
        test_settings.database_url,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def test_session_local(test_db_engine):
    """Create test session factory."""
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine,
    )


@pytest.fixture
def test_db(test_session_local) -> Generator[Session, None, None]:
    """Create test database session."""
    connection = test_session_local.kw["bind"].connect()
    transaction = connection.begin()
    session = test_session_local(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(test_db, test_settings):
    """Create test client with dependency overrides."""

    def override_get_session():
        return test_db

    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
