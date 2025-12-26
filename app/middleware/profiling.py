"""Profiling middleware for API performance analysis."""

import cProfile
import io
import logging
import os
import pstats
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Create dedicated profiling logger
profiling_logger = logging.getLogger("profiling")
profiling_logger.setLevel(logging.INFO)

# Create profiles directory only if not in test mode
PROFILES_DIR = Path("/app/profiles")
if not os.getenv("TESTING"):
    PROFILES_DIR.mkdir(exist_ok=True, parents=True)

    # Configure file handler for profiling logs
    log_handler = logging.FileHandler(PROFILES_DIR / "profiling.log")
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    profiling_logger.addHandler(log_handler)

# Always add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - PROFILING - %(message)s"))
profiling_logger.addHandler(console_handler)


class ProfilingMiddleware(BaseHTTPMiddleware):
    """Middleware to profile API endpoint performance using cProfile.

    This middleware profiles each request and saves:
    - Profiling stats in profiles/profiling.log
    - Binary profile data in profiles/*.prof files

    Enable it only in development or when debugging performance issues.
    """

    def __init__(
        self,
        app,
        top_results: int = 10,
        save_binary: bool = True,
        exclude_paths: list[str] | None = None,
        include_only_prefix: str | None = None,
    ):
        """Initialize the profiling middleware.

        Args:
            app: The FastAPI application instance
            top_results: Number of top functions to display in stats (default: 10)
            save_binary: Whether to save binary .prof files (default: True)
            exclude_paths: List of paths to exclude from profiling (default: ["/health"])
            include_only_prefix: If set, only profile paths starting with this prefix (e.g., "/api")
        """
        super().__init__(app)
        self.top_results = top_results
        self.save_binary = save_binary
        self.exclude_paths = exclude_paths or ["/health", "/api/v1/health"]
        self.include_only_prefix = include_only_prefix

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Profile the request and log performance statistics.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            The HTTP response
        """
        request_path = request.url.path

        # If include_only_prefix is set, only profile paths with that prefix
        if self.include_only_prefix:
            if not request_path.startswith(self.include_only_prefix):
                return await call_next(request)

        # Skip profiling for excluded paths (exact match or prefix)
        for excluded in self.exclude_paths:
            if request_path == excluded or request_path.startswith(excluded + "/"):
                return await call_next(request)

        # Skip profiling if X-Skip-Profiling header is present
        if request.headers.get("X-Skip-Profiling"):
            return await call_next(request)

        # Skip profiling for batch/automation scripts (check User-Agent)
        user_agent = request.headers.get("User-Agent", "")
        if "python-requests" in user_agent.lower():
            return await call_next(request)

        # Start profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Track request time
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Stop profiling
        profiler.disable()
        duration = time.time() - start_time

        # Log profiling results
        self._log_profile_stats(profiler, request, duration)

        # Add duration header to response
        response.headers["X-Process-Time"] = str(duration)

        return response

    def _log_profile_stats(
        self, profiler: cProfile.Profile, request: Request, duration: float
    ):
        """Log the profiling statistics and save to files.

        Args:
            profiler: The cProfile profiler instance
            request: The HTTP request
            duration: Total request duration in seconds
        """
        # Create stats from profiler
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)

        # Sort by total time and get top results
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats(self.top_results)

        # Log the results to profiling.log
        profiling_logger.info(
            f"Profile for {request.method} {request.url.path} "
            f"(Duration : {duration: .4f}s): \n{stream.getvalue()}"
        )

        # Save binary profile file if enabled
        if self.save_binary:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            endpoint = request.url.path.replace("/", "_")
            profile_file = PROFILES_DIR / f"{timestamp}{endpoint}.prof"

            profiler.dump_stats(str(profile_file))
            profiling_logger.info(f"Binary profile saved to: {profile_file}")
