"""Pytest configuration and shared fixtures."""

import os
import sys
from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

# Disable profiling in tests
os.environ["TESTING"] = "true"
# Set API key for tests
os.environ["API_KEY"] = "test-api-key-12345"

# Mock psutil BEFORE any app imports
psutil_mock = MagicMock()
psutil_mock.Process.return_value.cpu_percent.return_value = 10.5
psutil_mock.Process.return_value.memory_info.return_value = MagicMock(
    rss=104857600
)  # 100 MB
sys.modules["psutil"] = psutil_mock

# Now safe to import app modules
from app.config.settings import Settings  # noqa: E402
from app.ml.model_manager import model_manager  # noqa: E402

# Configure model_manager for tests (ONNX Runtime)
model_manager.initialized = True
model_manager.model_name = "test-model"
# Mock ONNX session instead of scikit-learn model
mock_session = MagicMock()
model_manager._session = mock_session
# Mock the predict method to return test predictions
model_manager.predict = MagicMock(return_value=(1, 0.85))

test_settings = Settings(
    app_name="Rugby MLOps Test",
    app_version="0.1.0",
    debug=True,
    api_prefix="/api/v1",
    api_key="test-api-key-12345",
    hf_repo_id="XavierCoulon/rugby-kicks-model",
    database_url="sqlite:///:memory:",
)

# Create test engine
test_engine = create_engine(
    test_settings.database_url,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,  # Don't pool connections in tests
)

# Import database module and override settings
import app.db.database as db_module  # noqa: E402
from app.db.database import get_session  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.main import app  # noqa: E402

db_module._engine = test_engine
db_module._SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)

# Now safe to import app and models
Base.metadata.create_all(bind=test_engine)


@pytest.fixture(scope="session")
def test_settings_fixture() -> Settings:
    """Provide test settings."""
    return test_settings


@pytest.fixture(scope="session")
def test_db_engine():
    """Provide test database engine."""
    yield test_engine


@pytest.fixture(scope="session")
def test_session_local():
    """Provide test session factory."""
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine,
    )


@pytest.fixture(scope="function")
def test_db(test_session_local) -> Generator[Session, None, None]:
    """Create test database session."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = test_session_local(bind=connection)

    # Create tables for this test
    Base.metadata.create_all(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(test_db):
    """Create test client with dependency overrides."""

    def override_get_session():
        return test_db

    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
