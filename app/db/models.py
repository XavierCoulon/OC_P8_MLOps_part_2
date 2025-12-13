"""Database ORM models using SQLAlchemy."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String

from app.db.database import Base


class PredictionInput(Base):
    """ORM model for prediction inputs stored in the database."""

    __tablename__ = "prediction_inputs"

    id = Column(Integer, primary_key=True, index=True)
    time_norm = Column(Float, nullable=False)
    distance = Column(Integer, nullable=False)
    angle = Column(Integer, nullable=False)
    wind_speed = Column(Float, nullable=False)
    precipitation_probability = Column(Float, nullable=False)
    is_left_footed = Column(Integer, nullable=False)
    game_away = Column(Integer, nullable=False)
    is_endgame = Column(Integer, nullable=False)
    is_start = Column(Integer, nullable=False)
    is_left_side = Column(Integer, nullable=False)
    has_previous_attempts = Column(Integer, nullable=False)
    prediction = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    status_code = Column(Integer, nullable=True, default=200)
    error_message = Column(String, nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        """String representation."""
        return (
            f"<PredictionInput(id={self.id}, "
            f"distance={self.distance}, angle={self.angle}, "
            f"prediction={self.prediction})>"
        )
