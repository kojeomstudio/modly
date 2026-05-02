import { ChildProcess, spawn } from 'child_process'
import { join } from 'path'
import { randomBytes } from 'crypto'
import { app, BrowserWindow } from 'electron'
import { existsSync, mkdirSync, writeFileSync, unlinkSync } from 'fs'
import axios from 'axios'
import { getSettings } from './settings-store'
import { logger } from './logger'
import { cleanPythonEnv, getVenvPythonExe } from './python-setup'
import { getBuiltinExtensionsDir } from './builtin-sync'

/**
 * Ensures `dir` exists and is writable by the current user.
 *
 * On macOS we have repeatedly seen userData / workspace / extensions
 * directories left root-owned after a one-off `sudo` run; subsequent
 * non-sudo launches then fail with cryptic EACCES deep inside Python.
 * Probing with a write+unlink up front lets us surface a clear error
 * with the exact `chown` command to run.
 */
function ensureWritableDir(dir: string, label: string): string {
  const me  = `${process.getuid?.() ?? '?'}:${process.getgid?.() ?? '?'}`
  const hint = (code: string): string =>
    `${label} directory is not writable: ${dir} (${code}).\n` +
    `If you ever launched Modly with sudo, the directory may be owned ` +
    `by root. Run:\n` +
    `  sudo chown -R "$(id -un):$(id -gn)" "${dir}"\n` +
    `(current process uid:gid = ${me})`

  try {
    mkdirSync(dir, { recursive: true })
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code ?? ''
    if (code === 'EACCES' || code === 'EPERM') throw new Error(hint(code))
    throw err
  }

  const probe = join(dir, `.modly-write-probe-${process.pid}`)
  try {
    writeFileSync(probe, '')
    unlinkSync(probe)
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code ?? ''
    throw new Error(hint(code))
  }
  return dir
}

const API_PORT = 8765
const API_HOST = '127.0.0.1'
export const API_BASE_URL = `http://${API_HOST}:${API_PORT}`
export const API_TOKEN_HEADER = 'X-Modly-Token'

export class PythonBridge {
  private process: ChildProcess | null = null
  private ready = false
  private startPromise: Promise<void> | null = null
  private getWindow: (() => BrowserWindow | null) | null = null
  private intentionalStop = false
  // Per-launch random token. Same value passed to FastAPI via env (MODLY_API_TOKEN)
  // and to the renderer via app:info IPC, so all callers can attach the
  // X-Modly-Token header. Defends loopback API against same-machine browser tabs.
  private readonly apiToken: string = randomBytes(32).toString('hex')

  constructor() {
    // Inject token into the global axios defaults used by the main process so
    // every axios.* call originating from main automatically authenticates.
    axios.defaults.headers.common[API_TOKEN_HEADER] = this.apiToken
  }

  getApiToken(): string {
    return this.apiToken
  }

  setWindowGetter(fn: () => BrowserWindow | null): void {
    this.getWindow = fn
  }

  async start(): Promise<void> {
    if (this.ready) return
    if (this.startPromise) return this.startPromise
    this.startPromise = this._start()
    try {
      await this.startPromise
    } finally {
      this.startPromise = null
    }
  }

  private async _start(): Promise<void> {
    if (this.process) {
      await this.waitUntilReady()
      return
    }

    // Verify userData itself is writable before anything else touches it.
    // sudo-created caches deep under it surface as opaque EACCES later.
    ensureWritableDir(app.getPath('userData'), 'User data')

    const pythonExecutable = this.resolvePythonExecutable()
    const apiDir = this.resolveApiDir()

    console.log('[PythonBridge] Starting FastAPI at', apiDir)
    console.log('[PythonBridge] Python executable:', pythonExecutable)

    await this.killProcessOnPort()

    this.process = spawn(pythonExecutable, ['-m', 'uvicorn', 'main:app', '--host', API_HOST, '--port', String(API_PORT)], {
      cwd: apiDir,
      env: {
        ...cleanPythonEnv(),
        PYTHONUNBUFFERED:          '1',
        // No PYTHONPATH needed — the venv's Python has its own isolated site-packages
        MODELS_DIR:                this.resolveModelsDir(),
        WORKSPACE_DIR:             this.resolveWorkspaceDir(),
        EXTENSIONS_DIR:            this.resolveExtensionsDir(),
        // Built-ins shipped with the .app live in userData/builtin-extensions
        // and are scanned by the registry alongside user-installed extensions.
        BUILTIN_EXTENSIONS_DIR:    getBuiltinExtensionsDir(),
        SELECTED_MODEL_ID:         process.env['SELECTED_MODEL_ID'] ?? '',
        HUGGING_FACE_HUB_TOKEN:    this.resolveHfToken(),
        HF_TOKEN:                  this.resolveHfToken(),
        MODLY_API_TOKEN:           this.apiToken,
        // Force the legacy LFS download path. HuggingFace's experimental
        // Xet (Content-Addressed Storage) protocol is being rolled out for
        // some repos but the macOS hf_xet 1.0.x client has hit two real
        // problems for our users: (a) it logs to ~/.cache/huggingface/xet
        // which is often left root-owned by an old sudo run and (b) it
        // returns 416 Range Not Satisfiable from cas-server.xethub.hf.co
        // when its on-disk state diverges from the server side. Disabling
        // it falls back to plain HTTPS chunked downloads — slower in
        // theory, but reliable. Override at launch time if you want to
        // re-enable it for a specific debug session.
        HF_HUB_DISABLE_XET:        process.env['HF_HUB_DISABLE_XET'] ?? '1',
      }
    })

    this.process.stdout?.on('data', (data) => {
      const msg = data.toString().trim()
      console.log('[FastAPI]', msg)
      logger.python(msg)
      this.emitTqdmLog(msg)
    })

    this.process.stderr?.on('data', (data) => {
      const msg = data.toString().trim()
      console.error('[FastAPI]', msg)
      logger.python(`[stderr] ${msg}`)
      this.emitTqdmLog(msg)
    })

    this.process.on('exit', (code) => {
      const wasReady = this.ready
      console.log('[PythonBridge] Process exited with code', code)
      this.ready = false
      this.process = null
      if (wasReady && !this.intentionalStop) {
        this.getWindow?.()?.webContents.send('python:crashed', { code })
      }
    })

    await this.waitUntilReady()
  }

  async stop(): Promise<void> {
    if (!this.process) return
    const proc = this.process
    this.process = null
    this.ready = false
    if (process.platform === 'win32') {
      const { execSync } = require('child_process')
      try { execSync(`taskkill /PID ${proc.pid} /T /F`) } catch {}
    } else {
      proc.kill('SIGTERM')
    }
    console.log('[PythonBridge] Stopped')
  }

  async restart(): Promise<void> {
    console.log('[PythonBridge] Restarting to free memory…')
    this.intentionalStop = true
    await this.stop()
    this.intentionalStop = false
    await this.start()
  }

  private emitTqdmLog(raw: string): void {
    if (/INFO/.test(raw)) return
    if (!raw.trim()) return
    this.getWindow?.()?.webContents.send('python:log', raw.trim())
  }

  isReady(): boolean { return this.ready }
  getPort(): number { return API_PORT }

  private async killProcessOnPort(): Promise<void> {
    const { execSync } = require('child_process')

    if (process.platform !== 'win32') {
      try { execSync(`lsof -ti tcp:${API_PORT} | xargs kill -9 2>/dev/null || true`, { shell: true }) } catch {}
      return
    }

    for (let attempt = 0; attempt < 3; attempt++) {
      let output = ''
      try {
        output = execSync(`netstat -ano | findstr ":${API_PORT} "`, { encoding: 'utf8', shell: true }) as string
      } catch { break }

      const pids = new Set<string>()
      for (const line of output.split('\n')) {
        const match = line.trim().match(/\s+(\d+)$/)
        if (match && match[1] !== '0') pids.add(match[1])
      }
      if (pids.size === 0) break

      for (const pid of pids) {
        try { execSync(`taskkill /PID ${pid} /T /F`, { shell: true }) } catch {}
      }
      await new Promise((r) => setTimeout(r, 300))
    }
  }

  private async waitUntilReady(maxRetries = 180, delayMs = 500): Promise<void> {
    for (let i = 0; i < maxRetries; i++) {
      if (!this.process) throw new Error('FastAPI process exited unexpectedly during startup')
      try {
        await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 })
        this.ready = true
        console.log('[PythonBridge] FastAPI is ready')
        return
      } catch {
        await new Promise((r) => setTimeout(r, delayMs))
      }
    }
    throw new Error('FastAPI did not start in time')
  }

  private resolvePythonExecutable(): string {
    const userData = app.getPath('userData')
    const apiDir = this.resolveApiDir()

    // Primary: venv created during setup (bundled Python → isolated venv)
    const venvPython = getVenvPythonExe(userData)
    if (existsSync(venvPython)) return venvPython

    // Dev fallback: local .venv in the api directory
    const devCandidates = [
      join(apiDir, '.venv', 'Scripts', 'python.exe'),
      join(apiDir, '.venv', 'bin', 'python'),
    ]
    for (const c of devCandidates) {
      if (existsSync(c)) return c
    }

    // Never fall back to bare 'python' on Windows — it would be the user's system Python
    if (process.platform === 'win32') {
      throw new Error('Python venv not found. Please restart the application to re-run setup.')
    }
    return 'python3'
  }

  private resolveApiDir(): string {
    if (app.isPackaged) return join(process.resourcesPath, 'api')
    return join(app.getAppPath(), 'api')
  }

  private resolveModelsDir(): string {
    const s = getSettings(app.getPath('userData'))
    return ensureWritableDir(s.modelsDir, 'Models')
  }

  private resolveWorkspaceDir(): string {
    const s = getSettings(app.getPath('userData'))
    return ensureWritableDir(s.workspaceDir, 'Workspace')
  }

  private resolveExtensionsDir(): string {
    const s = getSettings(app.getPath('userData'))
    return ensureWritableDir(s.extensionsDir, 'Extensions')
  }

  private resolveHfToken(): string {
    return getSettings(app.getPath('userData')).hfToken ?? ''
  }
}
