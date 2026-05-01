"""
BaseGenerator — contract that each model adapter must implement.
"""
from abc import ABC, abstractmethod
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple


class GenerationCancelled(Exception):
    """Raised by generators when a cancel_event is set mid-generation."""


def pick_device() -> Tuple[str, "object"]:
    """
    Selects the best available torch device and a sensible default dtype.

    Returns (device_str, torch_dtype). Order of preference:
        1. CUDA — fp16 (broad op coverage, big speedup)
        2. MPS  — fp32 (Apple Silicon Metal; fp16 has frequent op-level
                  fallbacks that silently move tensors to CPU and end up
                  slower than staying in fp32)
        3. CPU  — fp32

    Side effect on MPS: sets PYTORCH_ENABLE_MPS_FALLBACK=1 unless the
    caller pinned it explicitly. Without this, any unsupported op (e.g.
    aten::col2im for some volume decoders) raises RuntimeError mid-run.
    Falling back to CPU for those ops is slow but the alternative is a
    crash partway through generation.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps", torch.float32
    return "cpu", torch.float32


def release_device_memory(device: Optional[str] = None) -> None:
    """
    Releases cached GPU memory for the active device. No-op on CPU.

    Accepts an explicit device string (passed by callers that already
    know what they were running on) or auto-detects via pick_device()
    when called without args from cleanup paths.
    """
    try:
        import torch
    except ImportError:
        return
    dev = device or pick_device()[0]
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dev == "mps" and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def smooth_progress(
    progress_cb: Callable[[int, str], None],
    start: int,
    end: int,
    label: str,
    stop: threading.Event,
    interval: float = 3.0,
) -> None:
    """
    Asymptotically increments progress between start and end-2 while a
    long-running operation runs without being able to emit callbacks.

    Earlier versions used a fixed linear schedule that capped at end-2
    after ~30 seconds. On MPS / CPU some pipelines run for 20+ minutes,
    so the bar would sit motionless at e.g. 80% for the entire compute
    phase and the UI looked dead. The asymptotic schedule keeps ticking
    until stop is set, so it never gives up early and never overshoots.

    Decay / floor are tuned by range so the same function fits both the
    short pre-load phase (0→9, target ~30 s) and the long inference
    phase (12→82, target many minutes): a wide range plateaus much
    later, a narrow one converges quickly.
    """
    current = float(start)
    target  = float(end - 2)
    if target <= current:
        return
    span  = end - start
    # Wide ranges (multi-minute compute) — slow climb, stays under 80%
    # for the bulk of the run. Narrow ranges (~10 s loads) — quick climb.
    decay = 0.02 if span > 20 else 0.10
    floor = 0.05 if span > 20 else 0.30
    last_reported = -1

    while not stop.is_set():
        stop.wait(interval)
        if stop.is_set():
            break
        step    = max(floor, (target - current) * decay)
        current = min(target, current + step)
        rounded = int(current)
        if rounded != last_reported:
            progress_cb(rounded, label)
            last_reported = rounded


class BaseGenerator(ABC):
    # ------------------------------------------------------------------ #
    # Metadata — override in each subclass
    # ------------------------------------------------------------------ #
    MODEL_ID:     str = ""
    DISPLAY_NAME: str = ""
    VRAM_GB:      int = 0   # Minimum recommended VRAM (in GB)

    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        self.model_dir         = model_dir
        self.outputs_dir       = outputs_dir
        self._model            = None
        # Injected by the registry from the manifest
        self.hf_repo:          str  = ""
        self.hf_skip_prefixes: list = []
        self.download_check:   str  = ""   # relative path to check in model_dir
        self._params_schema:   list = []   # params declared in the manifest

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        """
        Checks that model files are present on disk.
        Uses download_check from the manifest if available,
        otherwise checks that model_dir exists and is non-empty.
        Can be overridden in generator.py for custom logic.
        """
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (GPU/CPU)."""
        ...

    def unload(self) -> None:
        """Release memory. Can be overridden if needed."""
        self._model = None
        import gc
        gc.collect()
        release_device_memory()
        # Force the OS to reclaim unused memory from this process
        try:
            import ctypes
            if sys.platform == "win32":
                kernel32 = ctypes.windll.kernel32
                kernel32.SetProcessWorkingSetSizeEx(
                    kernel32.GetCurrentProcess(), -1, -1, 0
                )
        except Exception:
            pass

    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    @abstractmethod
    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """
        Starts 3D generation from an image.
        Returns the path to the generated .glb file.
        progress_cb(percent: int, step_label: str)
        cancel_event: set this to interrupt generation between steps.
        """
        ...

    def _check_cancelled(self, cancel_event: Optional[threading.Event]) -> None:
        """Raises GenerationCancelled if cancel_event is set."""
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

    # ------------------------------------------------------------------ #
    # Parameter schema (for the UI)
    # ------------------------------------------------------------------ #

    def params_schema(self) -> list:
        """
        Returns the parameter schema for the UI.
        Reads _params_schema injected from the manifest.
        Can be overridden in generator.py for custom logic.
        """
        return self._params_schema

    # ------------------------------------------------------------------ #
    # Standard download
    # ------------------------------------------------------------------ #

    def _auto_download(self) -> None:
        """
        Downloads weights from self.hf_repo (injected by the registry).
        Used as a fallback when is_downloaded() returns False.
        Extensions can override this method for custom logic.
        """
        if not self.hf_repo:
            raise RuntimeError(
                f"[{self.MODEL_ID}] Cannot download: hf_repo not configured. "
                "Check the extension's manifest.json."
            )

        from huggingface_hub import snapshot_download

        print(f"[{self.__class__.__name__}] Downloading {self.hf_repo} → {self.model_dir} …")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        ignore = list(self.hf_skip_prefixes) + [
            "*.md", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes",
        ]
        snapshot_download(
            repo_id=self.hf_repo,
            local_dir=str(self.model_dir),
            ignore_patterns=ignore,
        )
        print(f"[{self.__class__.__name__}] Download complete.")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _report(
        self,
        progress_cb: Optional[Callable[[int, str], None]],
        pct: int,
        step: str,
    ) -> None:
        if progress_cb:
            progress_cb(pct, step)
