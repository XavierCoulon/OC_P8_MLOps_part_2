"""Tests for database module."""

from sqlalchemy.orm import Session

from app.db.database import (
    Base,
    SessionDep,
    SessionLocal,
    _get_engine,
    _get_session_local,
    engine,
    get_session,
)


class TestDatabaseModule:
    """Test suite for database module."""

    def test_base_declarative_class(self):
        """Test that Base is a valid declarative base."""
        assert Base is not None
        assert hasattr(Base, "metadata")

    def test_get_engine_creates_engine(self):
        """Test that _get_engine creates a database engine."""
        db_engine = _get_engine()
        assert db_engine is not None
        assert hasattr(db_engine, "url")

    def test_get_engine_returns_same_instance(self):
        """Test that _get_engine returns the same engine instance (singleton)."""
        engine1 = _get_engine()
        engine2 = _get_engine()
        assert engine1 is engine2

    def test_get_session_local_creates_session_factory(self):
        """Test that _get_session_local creates a session factory."""
        session_factory = _get_session_local()
        assert session_factory is not None
        assert callable(session_factory)

    def test_get_session_local_returns_same_instance(self):
        """Test that _get_session_local returns the same factory (singleton)."""
        factory1 = _get_session_local()
        factory2 = _get_session_local()
        assert factory1 is factory2

    def test_engine_function(self):
        """Test the engine() convenience function."""
        db_engine = engine()
        assert db_engine is not None
        assert db_engine is _get_engine()

    def test_session_local_function(self):
        """Test the SessionLocal() convenience function."""
        session_factory = SessionLocal()
        assert session_factory is not None
        assert session_factory is _get_session_local()

    def test_get_session_generator(self):
        """Test that get_session yields a valid database session."""
        session_generator = get_session()
        db_session = next(session_generator)

        assert isinstance(db_session, Session)
        assert db_session.is_active

        # Clean up
        try:
            next(session_generator)
        except StopIteration:
            pass  # Expected behavior

    def test_get_session_closes_on_exit(self):
        """Test that get_session properly closes the session."""
        session_generator = get_session()
        db_session = next(session_generator)

        # Session should be active
        assert db_session.is_active

        # Trigger cleanup
        try:
            session_generator.close()
        except GeneratorExit:
            pass

        # After cleanup, create a new session to verify it works
        new_generator = get_session()
        new_session = next(new_generator)
        assert isinstance(new_session, Session)

        try:
            new_generator.close()
        except GeneratorExit:
            pass

    def test_session_dep_type_annotation(self):
        """Test that SessionDep is properly annotated."""
        # SessionDep should be an Annotated type
        assert hasattr(SessionDep, "__metadata__")

    def test_create_session_from_factory(self):
        """Test creating a session from the session factory."""
        factory = _get_session_local()
        session = factory()

        assert isinstance(session, Session)
        assert session.is_active

        # Clean up
        session.close()
