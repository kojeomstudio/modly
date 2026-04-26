"""
Local-only API token middleware.

Defends against same-machine cross-origin requests (browser tabs hitting
http://127.0.0.1:8765). Electron generates a per-launch random token,
passes it to uvicorn via the MODLY_API_TOKEN env var, and injects the
same token into all renderer/main-process HTTP calls via the
X-Modly-Token header.

If MODLY_API_TOKEN is empty (e.g., manual `uvicorn main:app` for local
debugging), enforcement is disabled so the dev workflow keeps working.

Exempt paths are file-serving GETs that have to be reachable from
browser anchor/img elements where custom headers cannot be attached.
Their threat surface is bounded by the existing path-traversal guards
on WORKSPACE_DIR.
"""
import os

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


TOKEN_HEADER = "X-Modly-Token"

# Exact paths that bypass token check entirely.
_EXEMPT_EXACT = frozenset({
    "/health",
    "/optimize/serve-file",
    "/optimize/export",
})

# Path prefixes that bypass token check (only for GET).
_EXEMPT_GET_PREFIXES = ("/workspace/",)


def _expected_token() -> str:
    return os.environ.get("MODLY_API_TOKEN", "")


def _is_exempt(method: str, path: str) -> bool:
    if path in _EXEMPT_EXACT:
        return True
    if method == "GET" and any(path.startswith(p) for p in _EXEMPT_GET_PREFIXES):
        return True
    return False


class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        expected = _expected_token()
        if not expected:
            # No token configured → enforcement disabled (dev fallback).
            return await call_next(request)

        if _is_exempt(request.method, request.url.path):
            return await call_next(request)

        if request.headers.get(TOKEN_HEADER) != expected:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API token"},
            )

        return await call_next(request)
