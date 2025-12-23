"""Database ORM models using SQLAlchemy."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class PredictionInput(Base):
    """ORM model for prediction inputs stored in the database."""

    __tablename__ = "prediction_inputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    time_norm: Mapped[float] = mapped_column(Float, nullable=False)
    distance: Mapped[int] = mapped_column(Integer, nullable=False)
    angle: Mapped[int] = mapped_column(Integer, nullable=False)
    wind_speed: Mapped[float] = mapped_column(Float, nullable=False)
    precipitation_probability: Mapped[float] = mapped_column(Float, nullable=False)
    is_left_footed: Mapped[int] = mapped_column(Integer, nullable=False)
    game_away: Mapped[int] = mapped_column(Integer, nullable=False)
    is_endgame: Mapped[int] = mapped_column(Integer, nullable=False)
    is_start: Mapped[int] = mapped_column(Integer, nullable=False)
    is_left_side: Mapped[int] = mapped_column(Integer, nullable=False)
    has_previous_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cpu_usage_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_usage_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status_code: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=200
    )
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
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
