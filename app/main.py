"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.api.routes import health
from app.utils.logger import logger


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix=settings.api_prefix)

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Startup event handler."""
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event handler."""
        logger.info(f"Shutting down {settings.app_name}")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Running {settings.app_name} on 0.0.0.0:8000")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
