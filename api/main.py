"""
Modly FastAPI backend.
Runs locally within the Electron app to provide AI inference endpoints.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import HTTPException

from routers import generation, model, optimize, status, settings, extensions, export, workflow_runs
from services.auth import TokenAuthMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize the registry (instantiates all adapters)
    from services.generator_registry import generator_registry
    generator_registry.initialize()
    yield
    # Shutdown: unload all models
    generator_registry.unload_all()


class _StatusFilter(logging.Filter):
    def filter(self, record):
        return "/generate/status/" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_StatusFilter())


app = FastAPI(
    title="Modly API",
    version="0.3.4",
    lifespan=lifespan,
)

# Token-based auth — defends against same-origin browser tabs probing the
# loopback API. No-op when MODLY_API_TOKEN env var is unset (manual dev).
# Added FIRST so it sits INSIDE CORSMiddleware (Starlette inserts middleware
# at index 0 → last-added is outermost). This lets CORS handle OPTIONS
# preflight responses (which never carry custom headers like X-Modly-Token)
# before they reach the auth check.
app.add_middleware(TokenAuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(settings.router)
app.include_router(model.router,      prefix="/model")
app.include_router(generation.router, prefix="/generate")
app.include_router(optimize.router,    prefix="/optimize")
app.include_router(extensions.router, prefix="/extensions")
app.include_router(export.router,          prefix="/export")
app.include_router(workflow_runs.router,   prefix="/workflow-runs")

# Serve generated files from workspace — dynamic so path changes take effect immediately
@app.get("/workspace/{full_path:path}")
async def serve_workspace_file(full_path: str):
    import services.generator_registry as reg
    workspace_root = reg.WORKSPACE_DIR.resolve()
    file_path = (workspace_root / full_path).resolve()
    # Reject anything that resolves outside WORKSPACE_DIR (path traversal guard).
    try:
        file_path.relative_to(workspace_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))
