# Running Modly on macOS (Apple Silicon)

Modly works on M-series Macs but the supported feature set differs from the
Windows/Linux + CUDA build. This document captures what's different and how
to avoid the common pitfalls.

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4 …). Intel Macs are not supported.
- macOS 12.0 (Monterey) or newer.
- Python 3.10–3.12 on `PATH` (3.13 is not yet supported by the PyTorch
  releases this project pulls in).
- ~10 GB free disk for the Hunyuan3D weights and the per-extension venvs.

## Don't run with sudo

The dev launcher writes to `datas/` and Electron's userData
(`~/Library/Application Support/Modly`). If `datas/` ends up owned by
`root`, the next non-sudo launch fails with a permission probe error
that points you at the fix:

```
sudo chown -R "$(id -un):$(id -gn)" datas
sudo chown -R "$(id -un):$(id -gn)" ~/Library/Application\ Support/Modly
```

The HuggingFace cache lives in `~/.cache/huggingface` and is shared
across all model downloads. The same sudo-leftover pattern there
shows up at startup as:

```
[python-bridge] HuggingFace cache is not writable in N path(s): …
```

Fix it the same way:

```
sudo chown -R "$(id -un):$(id -gn)" ~/.cache/huggingface
rm -rf ~/.cache/huggingface/xet     # stale Xet protocol state
```

The warning is non-fatal — downloads still complete via the
huggingface_hub fallback path — but ignoring it leaves "Ignored error
while writing commit hash" lines in every log.

## Compute device

The runner auto-selects MPS (Apple Silicon Metal) when available, falling
back to CPU. fp32 is used on MPS — fp16 has too many op-level fallbacks
to be reliable. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so
that ops without a Metal kernel fall through to CPU instead of crashing
mid-run.

## Sensible parameter defaults

The Hunyuan3D-mini and Hunyuan3D-mini-turbo extensions ship with macOS-
specific parameter defaults:

- **Mesh Resolution (octree):** `Low (256)` — the `Medium (380)` default
  used on CUDA hosts reliably trips macOS jetsam (the kernel OOM killer)
  on 8–16 GB unified-memory machines during volume decoding. `256` still
  produces clean meshes and stays well under the limit.
- You can override the default in the params panel; if it errors out
  with "Backend was killed (signal 9)", drop the resolution.

## Unsupported features on macOS

- **Texture generation** — requires CUDA-only C++ extensions
  (`custom_rasterizer`, `differentiable_renderer`, `texture_baker`).
  The Texture toggle in the UI is disabled on macOS.
- **TRELLIS.2** and **TRELLIS.2 GGUF** extensions — depend on
  CUDA-only wheels (`cumesh`, `flex_gemm`). Manifest declares this and
  the registry blocks them at install time.

## Dev mode (`run-local.sh`)

`./run-local.sh` is the recommended dev launcher. It:

1. Checks Python and node, creates `api/.venv`, installs requirements.
2. Verifies `datas/` is writable by the current user (refuses to run
   under sudo).
3. **Mirrors `extensions/<submodule>/` into `datas/extensions/<id>/`**
   on every launch. Submodule edits to `generator.py`/`manifest.json`
   apply immediately; the per-extension `venv/` is preserved across
   launches so you don't pay the install cost again.
4. Wires `userData/settings.json` to point at `datas/` so the packaged
   first-run setup screen is bypassed.

Note: any manual edit you make under `datas/extensions/<id>/` to
`generator.py`, `manifest.json`, `setup.py`, `build_vendor.py`, or
`README.md` will be **overwritten** on the next `./run-local.sh`. Edit
the source under `extensions/<submodule>/` instead.

## When `setup.py` changes for an extension

`run-local.sh` does not rebuild a stale venv automatically. If a
submodule bumps a dependency or its setup logic, delete the venv and
re-launch:

```
rm -rf datas/extensions/hunyuan3d-mini/venv
./run-local.sh
```
