"""Pydantic models and schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str
