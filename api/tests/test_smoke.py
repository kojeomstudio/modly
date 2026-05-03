"""
Smoke tests — fast, no model weights, no GPU.

These exist to catch the cheap regressions: an import that breaks across
platforms, a manifest that loses a required key, the platform filter
silently letting an incompatible extension through. Anything that needs
torch/diffusers belongs in a separate suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the api/ package importable when pytest is invoked from repo root.
API_DIR = Path(__file__).resolve().parent.parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

EXTENSIONS_DIR = API_DIR.parent / "extensions"


def test_base_module_imports() -> None:
    """The contract module loads without pulling torch."""
    from services.generators import base

    assert hasattr(base, "BaseGenerator")
    assert hasattr(base, "smooth_progress")
    assert hasattr(base, "GenerationCancelled")
    assert hasattr(base, "pick_device")
    assert hasattr(base, "release_device_memory")


def test_pick_device_returns_known_device() -> None:
    """pick_device picks one of the three documented devices."""
    pytest.importorskip("torch")
    from services.generators.base import pick_device

    device, _dtype = pick_device()
    assert device in {"cuda", "mps", "cpu"}


def test_release_device_memory_is_safe_on_cpu() -> None:
    """release_device_memory must be a no-op when called on cpu, even if
    torch is not installed (the helper guards on ImportError)."""
    from services.generators.base import release_device_memory

    release_device_memory("cpu")  # must not raise


def _all_manifests() -> list[Path]:
    if not EXTENSIONS_DIR.exists():
        pytest.skip(f"extensions dir not present: {EXTENSIONS_DIR}")
    return sorted(EXTENSIONS_DIR.glob("*/manifest.json"))


@pytest.mark.parametrize("manifest_path", _all_manifests(), ids=lambda p: p.parent.name)
def test_manifest_required_keys(manifest_path: Path) -> None:
    """Every manifest declares the keys the registry depends on."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in ("id", "name", "type", "generator_class"):
        assert key in data, f"{manifest_path.name} missing required key {key!r}"

    nodes = data.get("nodes", [])
    assert isinstance(nodes, list)
    for node in nodes:
        assert "id" in node, f"{manifest_path.name} has a node without an id"


@pytest.mark.parametrize("manifest_path", _all_manifests(), ids=lambda p: p.parent.name)
def test_manifest_compatibility_well_formed(manifest_path: Path) -> None:
    """`compatibility.platforms` (when present) must list valid keys."""
    data    = json.loads(manifest_path.read_text(encoding="utf-8"))
    compat  = data.get("compatibility")
    if compat is None:
        return  # legacy manifest; allowed
    assert isinstance(compat, dict)
    plats = compat.get("platforms")
    if plats is not None:
        assert isinstance(plats, list) and plats, "platforms must be a non-empty list"
        for p in plats:
            assert p in {"win32", "linux", "darwin"}, f"unknown platform {p!r}"


def test_platform_filter_blocks_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry filter rejects manifests whose platforms exclude the host."""
    from services.generator_registry import _is_platform_supported
    import services.generator_registry as reg

    monkeypatch.setattr(reg, "_CURRENT_PLATFORM", "darwin")
    ok, _ = _is_platform_supported({"compatibility": {"platforms": ["win32", "linux"]}})
    assert ok is False

    ok, _ = _is_platform_supported({"compatibility": {"platforms": ["darwin"]}})
    assert ok is True

    ok, _ = _is_platform_supported({})  # legacy manifest
    assert ok is True


def _write_stub_extension(dir: Path, ext_id: str, version: str, has_setup: bool = True) -> None:
    """Drop a minimal manifest+generator.py pair into `dir` for scan tests."""
    dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id":              ext_id,
        "name":            f"Stub {ext_id}",
        "type":            "model",
        "version":         version,
        "generator_class": "StubGen",
        "nodes":           [{"id": "default"}],
    }
    (dir / "manifest.json").write_text(json.dumps(manifest))
    (dir / "generator.py").write_text("# stub")
    if has_setup:
        (dir / "setup.py").write_text("# stub")


def test_scan_extensions_root_user_dir_overrides_builtin(tmp_path: Path) -> None:
    """Scanning builtin first then user must let the user-dir entry win.

    Built-ins ship as a read-only template; users who install (or
    promote-and-modify) an extension with the same id must not be
    silently shadowed.
    """
    from services.generator_registry import _scan_extensions_root

    builtin = tmp_path / "builtin"
    user    = tmp_path / "user"
    _write_stub_extension(builtin / "shared", "shared", "1.0.0-builtin")
    _write_stub_extension(user / "shared",    "shared", "2.0.0-user")

    result: dict = {}
    _scan_extensions_root(builtin, result)
    _scan_extensions_root(user, result)

    assert "shared/default" in result
    _cls, manifest, ext_dir = result["shared/default"]
    assert manifest["version"] == "2.0.0-user"
    assert ext_dir == user / "shared"


def test_scan_extensions_root_setup_py_forces_subprocess_mode(tmp_path: Path) -> None:
    """Extensions that ship a setup.py must NOT be loaded in legacy direct
    mode even when no venv exists yet.

    Direct mode imports generator.py into the FastAPI parent process.
    Bundled built-ins (no venv) used to fall through to that path and
    raise 'No module named PIL' before the user could click Repair.
    The fix forces subprocess_mode=True whenever setup.py is present.
    """
    from services.generator_registry import _scan_extensions_root

    ext_root = tmp_path / "ext_root"
    _write_stub_extension(ext_root / "fresh", "fresh", "1.0.0", has_setup=True)

    result: dict = {}
    _scan_extensions_root(ext_root, result)

    assert "fresh/default" in result
    cls, _manifest, _ext_dir = result["fresh/default"]
    # cls is None when subprocess_mode was selected; a real class when direct.
    assert cls is None, "setup.py-bearing extension must enter subprocess mode"
