"""
ExtensionProcess — manages a generator running in an isolated subprocess.

Each extension runs in its own venv via runner.py.
Communication is done via newline-delimited JSON on stdin/stdout.

Interface is intentionally compatible with direct BaseGenerator usage
so GeneratorRegistry can treat both transparently.
"""
import base64
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

_RUNNER_PATH = Path(__file__).parent.parent / "runner.py"


def _venv_python(ext_dir: Path) -> Path:
    """Returns the path to the venv's Python executable."""
    if platform.system() == "Windows":
        return ext_dir / "venv" / "Scripts" / "python.exe"
    return ext_dir / "venv" / "bin" / "python"


class ExtensionProcess:
    """
    Wraps an extension subprocess. Presents the same interface as a
    direct generator (load / unload / generate / is_loaded / params_schema).
    """

    def __init__(self, ext_dir: Path, manifest: dict) -> None:
        self.ext_dir       = ext_dir
        self.manifest      = manifest
        self.model_dir     = None   # set by registry after init
        self.outputs_dir   = None   # set by registry after init

        self._proc:   Optional[subprocess.Popen] = None
        self._queue:  queue.Queue                = queue.Queue()
        # _send_lock guards stdin writes so two concurrent _send() calls can't
        # interleave bytes inside one JSON line.
        self._send_lock:    threading.Lock = threading.Lock()
        # _request_lock guards an entire send/recv cycle (load/generate/unload).
        # Without it, two background tasks calling load()/generate() at the
        # same time would both pull from self._queue and steal each other's
        # responses — manifesting as 'Unexpected response to load: {ready…}'.
        self._request_lock: threading.Lock = threading.Lock()
        self._loaded: bool                       = False
        # Set when the parent intentionally killed this subprocess (cancel
        # path). Suppresses post-kill stderr drain output that would
        # otherwise look like the process is still running.
        self._cancelled: bool = False

        # Mirrors BaseGenerator attributes used by the registry
        self.hf_repo          = manifest.get("hf_repo", "")
        self.hf_skip_prefixes = manifest.get("hf_skip_prefixes", [])
        self.download_check   = manifest.get("download_check", "")
        self._params_schema   = manifest.get("params_schema", [])

        # Public metadata
        self.MODEL_ID     = manifest.get("id", "")
        self.DISPLAY_NAME = manifest.get("name", "")
        self.VRAM_GB      = manifest.get("vram_gb", 0)

    # ------------------------------------------------------------------ #
    # Subprocess lifecycle
    # ------------------------------------------------------------------ #

    def _build_env(self) -> dict:
        from services.generator_registry import MODELS_DIR, WORKSPACE_DIR
        env = os.environ.copy()
        env["EXTENSION_DIR"] = str(self.ext_dir)
        env["MODELS_DIR"]    = str(MODELS_DIR)
        env["WORKSPACE_DIR"] = str(WORKSPACE_DIR)
        env["MODLY_API_DIR"] = str(Path(__file__).parent.parent)
        # Pass the exact model_dir so runner.py doesn't have to re-derive it
        # from manifest["id"] (which is the ext_id, not the composite node id).
        if self.model_dir is not None:
            env["MODEL_DIR"] = str(self.model_dir)
        # Extension venvs are based on python-embed which ships without a CA bundle.
        # Only set SSL_CERT_FILE if not already provided (preserves corporate/custom certs).
        if "SSL_CERT_FILE" not in env:
            try:
                import certifi
                env["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                pass
        return env

    def _start(self) -> None:
        """Launch the subprocess and wait for the 'ready' signal."""
        python = _venv_python(self.ext_dir)
        if not python.exists():
            raise RuntimeError(
                f"[{self.MODEL_ID}] venv not found at {python}. "
                "Run the extension's setup.py first."
            )

        self._proc = subprocess.Popen(
            [str(python), str(_RUNNER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._build_env(),
        )

        # Background thread: read stdout → queue
        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()

        # Background thread: forward stderr to our stderr
        stderr_fwd = threading.Thread(target=self._stderr_loop, daemon=True)
        stderr_fwd.start()

        # Wait for ready — runner sends params_schema in this message
        msg = self._recv(timeout=None)
        if msg.get("type") != "ready":
            self._proc.kill()
            raise RuntimeError(f"[{self.MODEL_ID}] Expected 'ready', got: {msg}")

        # Override params_schema with what the generator class actually declares
        if msg.get("params_schema"):
            self._params_schema = msg["params_schema"]

        print(f"[ExtensionProcess] {self.MODEL_ID} subprocess started (pid {self._proc.pid})")

    def _read_loop(self) -> None:
        """Continuously reads stdout and pushes parsed JSON to the queue."""
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if line:
                    try:
                        self._queue.put(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"[{self.MODEL_ID}] bad JSON: {line}", file=sys.stderr)
        finally:
            self._queue.put(None)  # sentinel: process is done

    def _stderr_loop(self) -> None:
        """Forwards subprocess stderr to the main process stderr.

        Once the subprocess has been intentionally killed (cancel path),
        we still drain the pipe to EOF so the kernel can free it, but
        drop the lines instead of printing them — otherwise tqdm output
        already buffered before the kill keeps streaming for seconds and
        the user sees a 'log keeps running after cancel' phantom.
        """
        for line in self._proc.stderr:
            if self._cancelled:
                continue
            print(f"[{self.MODEL_ID}] {line}", end="", file=sys.stderr)

    def _send(self, msg: dict) -> None:
        with self._send_lock:
            # Capture once — cancel_job() may null self._proc concurrently.
            # Without this guard, the cancel-event drain branch in
            # generate() can hit AttributeError when writing 'cancel' to a
            # subprocess that the parent is tearing down.
            proc = self._proc
            if proc is None or proc.stdin is None or proc.stdin.closed:
                raise BrokenPipeError(
                    f"[{self.MODEL_ID}] subprocess stdin is closed"
                )
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

    def _recv(self, timeout: float | None = 120.0) -> dict:
        try:
            msg = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"[{self.MODEL_ID}] No response from subprocess after {timeout}s")
        if msg is None:
            raise RuntimeError(f"[{self.MODEL_ID}] Subprocess died unexpectedly")
        return msg

    def _ensure_started(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            # Drain any stale messages from a previous (now dead) subprocess
            # before spawning a new one. Without this, a leftover sentinel
            # (None) or trailing message would be picked up by the next
            # _recv() instead of the new process's 'ready' line.
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._loaded = False
            self._cancelled = False
            self._start()

    def mark_cancelled(self) -> None:
        """Signal the cancel path: silence post-kill stderr drain.

        Called by the cancel route just before _proc.kill(). The stderr
        forwarder reads the flag inside its drain loop and drops further
        lines so users don't see seconds of buffered tqdm output after
        pressing Cancel.
        """
        self._cancelled = True

    # ------------------------------------------------------------------ #
    # BaseGenerator-compatible interface
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    def is_loaded(self) -> bool:
        return self._loaded and self._proc is not None and self._proc.poll() is None

    def load(self) -> None:
        with self._request_lock:
            if self._loaded and self._proc is not None and self._proc.poll() is None:
                return
            self._ensure_started()
            self._send({"action": "load"})

            msg = self._recv(timeout=None)  # model load can be arbitrarily slow
            if msg.get("type") == "loaded":
                self._loaded = True
            elif msg.get("type") == "error":
                raise RuntimeError(msg.get("traceback") or msg.get("message"))
            else:
                raise RuntimeError(f"[{self.MODEL_ID}] Unexpected response to load: {msg}")

    def unload(self) -> None:
        with self._request_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._send({"action": "unload"})
                    self._recv(timeout=30.0)
                except Exception:
                    pass
            self._loaded = False

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        from services.generators.base import GenerationCancelled

        # Serialise the whole IPC round-trip. Two concurrent generate() calls
        # would otherwise share self._queue and pick up each other's messages
        # (progress, done, error) in arbitrary order.
        with self._request_lock:
            req_id = str(uuid.uuid4())
            self._send({
                "action":      "generate",
                "id":          req_id,
                "image_b64":   base64.b64encode(image_bytes).decode(),
                "params":      params,
                "outputs_dir": str(self.outputs_dir) if self.outputs_dir else None,
            })

            while True:
                # ── Cancellation paths ──────────────────────────────────
                # 1. self._cancelled (set by mark_cancelled() from the
                #    FastAPI cancel route, paired with proc.kill()) — the
                #    subprocess is already dead or dying, just exit.
                # 2. cancel_event (cooperative) — proc is presumed alive,
                #    ask it to wind down gracefully and drain its terminal
                #    message before raising.
                if self._cancelled:
                    raise GenerationCancelled()
                if cancel_event and cancel_event.is_set():
                    # _send may hit a broken pipe if the proc was killed
                    # underneath us; that's fine, we still know we're
                    # cancelling.
                    try:
                        self._send({"action": "cancel", "id": req_id})
                    except Exception:
                        pass
                    deadline = time.monotonic() + 30.0
                    while time.monotonic() < deadline:
                        try:
                            ack = self._queue.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        # None sentinel = proc died, treat as ack.
                        if ack is None or ack.get("type") in ("cancelled", "done", "error"):
                            break
                    raise GenerationCancelled()

                # Poll queue with short timeout so we can re-check cancel
                # signals on every iteration.
                try:
                    msg = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if msg is None:
                    # Subprocess pipe closed. Disambiguate three causes:
                    #   a) FastAPI cancel route killed it → GenerationCancelled
                    #   b) macOS jetsam / Linux OOM-killer SIGKILL  → friendly hint
                    #   c) anything else (segfault, unhandled exception) → raw rc
                    if self._cancelled:
                        raise GenerationCancelled()
                    # Capture proc once — cancel_job nulls gen._proc and we
                    # can't dereference it twice safely without a lock.
                    proc = self._proc
                    rc   = proc.poll() if proc else None
                    if rc is not None and rc < 0 and abs(rc) == 9:
                        raise RuntimeError(
                            f"[{self.MODEL_ID}] Backend was killed (signal 9). "
                            "Most often this is the OS reclaiming memory under "
                            "pressure. Try a lower octree_resolution, fewer "
                            "inference steps, or close other apps."
                        )
                    raise RuntimeError(
                        f"[{self.MODEL_ID}] Subprocess died during generation "
                        f"(returncode={rc})"
                    )

                t = msg.get("type")

                if t == "progress":
                    if progress_cb:
                        progress_cb(msg.get("pct", 0), msg.get("step", ""))

                elif t == "done":
                    return Path(msg["output_path"])

                elif t == "error":
                    raise RuntimeError(msg.get("traceback") or msg.get("message", "Unknown error"))

                elif t == "cancelled":
                    raise GenerationCancelled()

                elif t == "log":
                    print(f"[{self.MODEL_ID}] {msg.get('message', '')}", file=sys.stderr)

    def params_schema(self) -> list:
        return self._params_schema

    def stop(self) -> None:
        """Gracefully shut down the subprocess."""
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"action": "shutdown"})
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
        self._loaded = False
        self._proc   = None
