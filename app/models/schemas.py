"""Pydantic models and schemas."""

from datetime import datetime

from pydantic import BaseModel
from pydantic.fields import Field


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str


class KickPredictionRequest(BaseModel):
    """Rugby kick prediction request model."""

    time_norm: float = Field(
        ge=0.01,
        le=1.00,
        description="Normalized time in match (0.01 - 1.00)",
        examples=[0.25, 0.5, 0.75],
    )
    distance: int = Field(
        ge=2,
        le=65,
        description="Distance to goal in meters (2 - 65)",
        examples=[15, 30, 50],
    )
    angle: int = Field(
        ge=0,
        le=180,
        description="Angle to goal in degrees (0 - 180)",
        examples=[0, 30, 45, 70],
    )
    wind_speed: float = Field(
        ge=0.00,
        le=18.17,
        description="Wind speed in km/h (0.00 - 18.17)",
        examples=[0.0, 5.2, 10.5, 18.0],
    )
    precipitation_probability: float = Field(
        ge=0.00,
        le=1.00,
        description="Precipitation probability (0.00 - 1.00)",
        examples=[0.0, 0.3, 0.7, 1.0],
    )
    is_left_footed: int = Field(
        ge=0,
        le=1,
        description="Player is left-footed (0 or 1)",
        examples=[0, 1],
    )
    game_away: int = Field(
        ge=0,
        le=1,
        description="Game is away (0 or 1)",
        examples=[0, 1],
    )
    is_endgame: int = Field(
        ge=0,
        le=1,
        description="Is endgame (0 or 1)",
        examples=[0, 1],
    )
    is_start: int = Field(
        ge=0,
        le=1,
        description="Is start of match (0 or 1)",
        examples=[0, 1],
    )
    is_left_side: int = Field(
        ge=0,
        le=1,
        description="Kick from left side (0 or 1)",
        examples=[0, 1],
    )
    has_previous_attempts: int = Field(
        ge=0,
        le=1,
        description="Player has previous attempts (0 or 1)",
        examples=[0, 1],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "time_norm": 0.5,
                "distance": 30,
                "angle": 45,
                "wind_speed": 5.2,
                "precipitation_probability": 0.3,
                "is_left_footed": 1,
                "game_away": 0,
                "is_endgame": 0,
                "is_start": 0,
                "is_left_side": 1,
                "has_previous_attempts": 0,
            }
        }
    }


class KickPredictionResponse(BaseModel):
    """Rugby kick prediction response model."""

    prediction: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of successful kick (0.0 - 1.0)",
        examples=[0.25, 0.5, 0.75, 0.95],
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction (0.0 - 1.0)",
        examples=[0.8, 0.85, 0.92, 0.98],
    )

    model_config = {
        "json_schema_extra": {"example": {"prediction": 0.75, "confidence": 0.92}}
    }


class PredictionInputResponse(BaseModel):
    """Full prediction input record with all attributes."""

    id: int
    time_norm: float
    distance: int
    angle: int
    wind_speed: float
    precipitation_probability: float
    is_left_footed: int
    game_away: int
    is_endgame: int
    is_start: int
    is_left_side: int
    has_previous_attempts: int
    prediction: float
    confidence: float
    latency_ms: float | None = None
    cpu_usage_percent: float | None = None
    memory_usage_mb: float | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
