"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app

from app.api.routes import health, predictions
from app.config.settings import settings
from app.db.database import create_db_and_tables
from app.ml.model_manager import model_manager
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown).

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Create database tables
    try:
        create_db_and_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")

    # Load model from Hugging Face
    try:
        logger.info(f"Loading model: {settings.hf_repo_id}")
        model_manager.load_model(hf_repo_id=settings.hf_repo_id)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load model at startup: {str(e)}")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
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
    app.include_router(predictions.router, prefix=settings.api_prefix)

    # Mount Gradio interface on /
    try:
        from gradio_app import build_interface

        demo = build_interface()
        app = mount_gradio_app(app, demo, path="/")
        logger.info("Gradio interface mounted on /")
    except Exception as e:
        logger.warning(f"Failed to mount Gradio interface: {e}")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Running {settings.app_name} on 0.0.0.0 : 8000")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
