#!/usr/bin/env bash
# Modly — One-touch local launcher.
# Installs JS + Python deps, points Electron's userData at <project>/datas,
# bypasses the in-app first-run setup, then starts dev mode.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATAS_DIR="${PROJECT_ROOT}/datas"
API_DIR="${PROJECT_ROOT}/api"
VENV_DIR="${API_DIR}/.venv"
REQ_FILE="${API_DIR}/requirements.txt"

OS="$(uname -s)"
case "$OS" in
  Darwin) USERDATA="${HOME}/Library/Application Support/Modly" ;;
  Linux)  USERDATA="${HOME}/.config/Modly" ;;
  *) echo "[ERROR] Unsupported OS for this script: $OS" >&2; exit 1 ;;
esac

log() { printf '\033[1;36m[run-local]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run-local][ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

log "Project: ${PROJECT_ROOT}"
log "Datas:   ${DATAS_DIR}"
log "UserData:${USERDATA}"

# ---- 1. Tooling checks ----
command -v node >/dev/null 2>&1 || die "Node.js not found. Install from https://nodejs.org"
command -v npm  >/dev/null 2>&1 || die "npm not found."

PYTHON_BIN=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 \
     && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' >/dev/null 2>&1; then
    PYTHON_BIN="$c"; break
  fi
done
[[ -n "$PYTHON_BIN" ]] || die "Python 3.10+ required (try: brew install python@3.11 / apt install python3 python3-venv)."
log "Using $("$PYTHON_BIN" --version 2>&1) (${PYTHON_BIN})"

# ---- 2. Local data directories ----
# Pre-flight: datas/ must be writable by the current user. If a previous
# Docker / sudo run left it root-owned, mkdir/cat in this script — and the
# Electron app at runtime — will silently fail to create models/workspace/etc.
CHOWN_TARGET="$(id -un):$(id -gn)"
if [[ -e "${DATAS_DIR}" && ! -w "${DATAS_DIR}" ]]; then
  die "datas/ is not writable by $(id -un). Fix with: sudo chown -R \"${CHOWN_TARGET}\" \"${DATAS_DIR}\""
fi
mkdir -p \
  "${DATAS_DIR}/models" \
  "${DATAS_DIR}/workspace" \
  "${DATAS_DIR}/extensions" \
  "${DATAS_DIR}/workflows" \
  "${DATAS_DIR}/dependencies" \
  || die "Failed to create datas/ subdirectories. Check permissions on ${DATAS_DIR}."

# Quick write test — surfaces ownership issues on the subdirs themselves.
for sub in models workspace extensions workflows dependencies; do
  probe="${DATAS_DIR}/${sub}/.write-test"
  if ! ( : > "${probe}" ) 2>/dev/null; then
    die "datas/${sub} is not writable. Fix with: sudo chown -R \"${CHOWN_TARGET}\" \"${DATAS_DIR}\""
  fi
  rm -f "${probe}"
done

# ---- 3. JS deps ----
if [[ ! -d "${PROJECT_ROOT}/node_modules" ]]; then
  log "Installing npm dependencies…"
  (cd "${PROJECT_ROOT}" && npm install)
else
  log "node_modules present — skipping npm install."
fi

# ---- 4. Python venv + requirements ----
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "Creating Python venv at ${VENV_DIR}…"
  "$PYTHON_BIN" -m venv "${VENV_DIR}"
fi

REQ_HASH="$(sha256_of "${REQ_FILE}")"
HASH_MARKER="${VENV_DIR}/.requirements.sha256"
if [[ ! -f "${HASH_MARKER}" || "$(cat "${HASH_MARKER}" 2>/dev/null || true)" != "${REQ_HASH}" ]]; then
  log "Installing Python requirements (this can take a few minutes the first time)…"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/python" -m pip install -r "${REQ_FILE}"
  printf '%s\n' "${REQ_HASH}" > "${HASH_MARKER}"
else
  log "Python requirements up to date — skipping pip install."
fi

# ---- 5. Wire Electron userData → datas/ and bypass setup screen ----
log "Configuring Electron userData → ${USERDATA}"
mkdir -p "${USERDATA}" || die "Cannot create ${USERDATA}. Check permissions."

# settings.json — read by electron/main/settings-store.ts on every getSettings()
cat > "${USERDATA}/settings.json" <<EOF
{
  "modelsDir":       "${DATAS_DIR}/models",
  "workspaceDir":    "${DATAS_DIR}/workspace",
  "workflowsDir":    "${DATAS_DIR}/workflows",
  "extensionsDir":   "${DATAS_DIR}/extensions",
  "dependenciesDir": "${DATAS_DIR}/dependencies"
}
EOF

# Symlink userData/venv → api/.venv so checkSetupNeeded() in
# electron/main/python-setup.ts is satisfied without re-running the in-app setup.
USERDATA_VENV="${USERDATA}/venv"
if [[ -L "${USERDATA_VENV}" || -e "${USERDATA_VENV}" ]]; then
  rm -rf "${USERDATA_VENV}"
fi
ln -s "${VENV_DIR}" "${USERDATA_VENV}"

# python_setup.json — version + sha256 must match python-setup.ts
# (SETUP_VERSION = 3, hash of api/requirements.txt)
cat > "${USERDATA}/python_setup.json" <<EOF
{"version":3,"requirementsHash":"${REQ_HASH}"}
EOF

# ---- 6. Launch ----
log "Setup complete. Launching Modly (npm run dev)…"
cd "${PROJECT_ROOT}"
exec npm run dev
