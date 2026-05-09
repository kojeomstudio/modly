import asyncio
import json
import subprocess
import sys
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["extensions"])


@router.post("/reload")
async def reload_extensions():
    """
    Re-scans the extensions/ folder and reloads the registry without restarting FastAPI.
    Unloads all currently loaded generators before reloading.
    """
    from services.generator_registry import generator_registry
    generator_registry.reload()
    return {
        "reloaded": True,
        "models":   list(generator_registry._generators.keys()),
        "errors":   generator_registry.load_errors(),
    }


@router.post("/setup/{ext_id}")
async def setup_extension(ext_id: str):
    """
    Creates the isolated venv for an extension by running its setup.py.
    Called automatically after installing an extension from GitHub.
    Runs setup.py with Modly's embedded Python and the detected GPU SM.
    """
    from services.generator_registry import EXTENSIONS_DIR

    if EXTENSIONS_DIR is None or not EXTENSIONS_DIR.exists():
        raise HTTPException(400, "EXTENSIONS_DIR not configured")

    ext_dir  = EXTENSIONS_DIR / ext_id
    setup_py = ext_dir / "setup.py"

    if not ext_dir.exists():
        raise HTTPException(404, f"Extension '{ext_id}' not found in {EXTENSIONS_DIR}")
    if not setup_py.exists():
        # No setup.py → legacy extension, nothing to do
        return {"status": "skipped", "reason": "no setup.py"}

    # Detect GPU compute capability + CUDA toolkit version. Both are needed
    # because mini/turbo setup.py picks a torch wheel index based on the
    # combination — passing only gpu_sm collapsed Blackwell (cu128) onto the
    # cu124 path. We pass JSON in a single argv so the schema stays stable.
    gpu_sm       = _detect_gpu_sm()
    cuda_version = _detect_cuda_version()
    args = json.dumps({
        "python_exe":   sys.executable,
        "ext_dir":      str(ext_dir),
        "gpu_sm":       gpu_sm,
        "cuda_version": cuda_version,
    })

    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [sys.executable, str(setup_py), args],
            capture_output=True,
            text=True,
        )
    )

    if result.returncode != 0:
        raise HTTPException(500, f"setup.py failed:\n{result.stderr}")

    return {
        "status":       "ok",
        "gpu_sm":       gpu_sm,
        "cuda_version": cuda_version,
        "output":       result.stdout,
    }


@router.get("/errors")
async def extension_errors():
    """Returns extension loading errors (invalid manifest, failed import, etc.)."""
    from services.generator_registry import generator_registry
    return generator_registry.load_errors()


def _detect_gpu_sm() -> int:
    """Returns GPU compute capability as integer (e.g. 86 for SM 8.6), or 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return major * 10 + minor
    except Exception:
        pass
    return 0


def _detect_cuda_version() -> int:
    """Returns CUDA toolkit version torch was built against, encoded as int.

    Example: 12.4 → 124, 12.8 → 128, 11.8 → 118. Returns 0 when CUDA is
    unavailable (macOS, CPU-only torch). The mini/turbo setup.py uses this
    to pick between cu124 and cu128 wheel indexes — without it Blackwell
    GPUs end up on the cu124 path and re-downloading torch on first run.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        ver = torch.version.cuda  # e.g. "12.4"
        if not ver:
            return 0
        major, minor = ver.split(".")[:2]
        return int(major) * 10 + int(minor)
    except Exception:
        return 0
