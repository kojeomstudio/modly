import { ipcMain, BrowserWindow, dialog, app, shell } from 'electron'
import { buildSync } from 'esbuild'
import { autoUpdater } from 'electron-updater'
import { join } from 'path'
import { rm as rmAsync, readFile, writeFile, mkdir, readdir, rename, cp } from 'fs/promises'
import { existsSync, readdirSync, statSync } from 'fs'
import axios from 'axios'
import * as tar from 'tar'
import { PythonBridge, API_BASE_URL } from './python-bridge'
import {
  isModelDownloaded,
  listDownloadedModels,
  downloadModelFromHF,
} from './model-downloader'
import { getSettings, setSettings } from './settings-store'
import { checkSetupNeeded, markSetupDone, runFullSetup, getVenvPythonExe, ensureSslPatch } from './python-setup'
import { logger } from './logger'
import { getProcessRunner, getPythonProcessRunner, getExtPythonExe, terminateProcessRunner, terminateAllProcessRunners } from './process-runner'
import { getBuiltinExtensionsDir } from './builtin-sync'
import { spawn } from 'child_process'
import vendoredTrustedExtensions from './trusted-extensions.json'

type WindowGetter = () => BrowserWindow | null

// ─── GPU detect (best-effort, no Python required) ─────────────────────────────

interface GpuInfo { sm: number; cudaVersion: number }

function detectGpuInfo(): Promise<GpuInfo> {
  return new Promise((resolve) => {
    // Query compute cap + driver version in one call
    const proc = spawn('nvidia-smi', ['--query-gpu=compute_cap,driver_version', '--format=csv,noheader'], {
      stdio: ['ignore', 'pipe', 'ignore'],
    })
    let out = ''
    proc.stdout?.on('data', (d: Buffer) => { out += d.toString() })
    proc.on('close', (code) => {
      if (code === 0) {
        const line   = out.trim().split('\n')[0].trim()        // e.g. "8.6, 551.61"
        const parts  = line.split(',').map(s => s.trim())
        const sm     = Math.round(parseFloat(parts[0] ?? '') * 10)  // → 86
        // Derive max supported CUDA version from driver version
        // Driver ≥ 520 → CUDA 11.8, ≥ 525 → 12.0, ≥ 530 → 12.1, ≥ 535 → 12.2,
        // ≥ 545 → 12.3, ≥ 550 → 12.4, ≥ 555 → 12.5, ≥ 560 → 12.6
        const driverMajor = parseInt((parts[1] ?? '').split('.')[0] ?? '0', 10)
        let cudaVersion = 118  // safe minimum
        if      (driverMajor >= 570) cudaVersion = 128  // Blackwell (RTX 50xx, sm_120)
        else if (driverMajor >= 560) cudaVersion = 126
        else if (driverMajor >= 555) cudaVersion = 125
        else if (driverMajor >= 550) cudaVersion = 124
        else if (driverMajor >= 545) cudaVersion = 123
        else if (driverMajor >= 535) cudaVersion = 122
        else if (driverMajor >= 530) cudaVersion = 121
        else if (driverMajor >= 525) cudaVersion = 120
        else if (driverMajor >= 520) cudaVersion = 118
        resolve({ sm: isNaN(sm) ? 86 : sm, cudaVersion })
      } else {
        resolve({ sm: 86, cudaVersion: 118 })
      }
    })
    proc.on('error', () => resolve({ sm: 86, cudaVersion: 118 }))
  })
}

// ─── Run an extension's setup.py directly (no FastAPI needed) ─────────────────

function runExtensionSetup(
  extDir:      string,
  gpuSm:       number,
  cudaVersion: number,
  onLog?:      (line: string) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const userData  = app.getPath('userData')
    ensureSslPatch(userData)
    const pythonExe = getVenvPythonExe(userData)
    const setupPy   = join(extDir, 'setup.py')

    const args = JSON.stringify({ python_exe: pythonExe, ext_dir: extDir, gpu_sm: gpuSm, cuda_version: cudaVersion })
    const proc = spawn(pythonExe, [setupPy, args], {
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    const handleLine = (line: string) => { if (line) onLog?.(line) }

    let stderr = ''
    proc.stdout?.on('data', (d: Buffer) => d.toString().split('\n').forEach(handleLine))
    proc.stderr?.on('data', (d: Buffer) => {
      const s = d.toString()
      stderr += s
      s.split('\n').forEach(handleLine)
    })

    proc.on('close', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`setup.py exited with code ${code}\n${stderr.slice(-2000)}`))
    })
    proc.on('error', reject)
  })
}

export function setupIpcHandlers(pythonBridge: PythonBridge, getWindow: WindowGetter): void {
  const activeDownloads = new Map<string, { percent: number; file?: string; fileIndex?: number; totalFiles?: number }>()
  // Logging from renderer
  ipcMain.on('log:error', (_event, message: string) => logger.error(`[Renderer] ${message}`))
  ipcMain.handle('log:getPath', () => join(app.getPath('userData'), 'logs', 'modly.log'))
  ipcMain.handle('log:readAll', async (_event, session?: string) => {
    const logsDir = join(app.getPath('userData'), 'logs')
    const dir = session ? join(logsDir, 'sessions', session) : logsDir
    const files = ['modly.log', 'errors.log', 'runtime.log']
    const result: Record<string, string> = {}
    for (const file of files) {
      try {
        const filePath = join(dir, file)
        result[file] = existsSync(filePath) ? await readFile(filePath, 'utf-8') : ''
      } catch {
        result[file] = ''
      }
    }
    return result
  })
  ipcMain.handle('log:listSessions', () => {
    const sessionsDir = join(app.getPath('userData'), 'logs', 'sessions')
    if (!existsSync(sessionsDir)) return []
    try {
      return readdirSync(sessionsDir)
        .filter(f => statSync(join(sessionsDir, f)).isDirectory())
        .sort()
        .reverse()
    } catch {
      return []
    }
  })

  // Window controls (frameless window)
  ipcMain.on('window:minimize', () => getWindow()?.minimize())
  ipcMain.on('window:maximize', () => {
    const win = getWindow()
    if (!win) return
    win.isMaximized() ? win.restore() : win.maximize()
  })
  ipcMain.on('window:close', () => getWindow()?.close())

  // Setup handlers — skipped in dev (uses .venv instead of python-embed)
  ipcMain.handle('setup:check', async () => {
    const userData = app.getPath('userData')
    const defaultDataDir = join(app.getPath('documents'), 'Modly')
    return { needed: checkSetupNeeded(userData), defaultDataDir }
  })

  ipcMain.handle('setup:saveDataDir', async (_event, { baseDir }: { baseDir: string }) => {
    const userData = app.getPath('userData')
    setSettings(userData, {
      modelsDir:        join(baseDir, 'models'),
      workspaceDir:     join(baseDir, 'workspace'),
      workflowsDir:     join(baseDir, 'workflows'),
      extensionsDir:    join(baseDir, 'extensions'),
      dependenciesDir:  join(baseDir, 'dependencies'),
    })
  })

  ipcMain.handle('setup:run', async () => {
    const userData = app.getPath('userData')
    const win = getWindow()
    if (!win) return { success: false, error: 'No window available' }
    try {
      await runFullSetup(win, userData)
      markSetupDone(userData)
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  // Python bridge
  ipcMain.handle('python:start', async () => {
    try {
      await pythonBridge.start()
      return { success: true, port: pythonBridge.getPort() }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  ipcMain.handle('python:status', () => ({
    ready: pythonBridge.isReady(),
    apiUrl: API_BASE_URL
  }))

  // File system
  ipcMain.handle('fs:selectImage', async () => {
    const win = getWindow()
    if (!win) return null

    const result = await dialog.showOpenDialog(win, {
      title: 'Select an image',
      filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
      properties: ['openFile']
    })

    return result.canceled ? null : result.filePaths[0]
  })

  ipcMain.handle('fs:selectMeshFile', async () => {
    const win = getWindow()
    if (!win) return null

    const result = await dialog.showOpenDialog(win, {
      title: 'Select a 3D mesh file',
      filters: [{ name: '3D Mesh', extensions: ['glb', 'obj', 'stl', 'ply'] }],
      properties: ['openFile']
    })

    return result.canceled ? null : result.filePaths[0]
  })

  ipcMain.handle('fs:saveModel', async (_, defaultName: string) => {
    const win = getWindow()
    if (!win) return null

    const result = await dialog.showSaveDialog(win, {
      title: 'Save 3D Model',
      defaultPath: defaultName,
      filters: [
        { name: 'OBJ', extensions: ['obj'] },
        { name: 'GLB', extensions: ['glb'] },
        { name: 'STL', extensions: ['stl'] }
      ]
    })

    return result.canceled ? null : result.filePath
  })

  ipcMain.handle('fs:savePath', async (_, args: { filters: { name: string; extensions: string[] }[]; defaultPath?: string }) => {
    const win = getWindow()
    if (!win) return null
    const result = await dialog.showSaveDialog(win, {
      title:       'Choose output path',
      filters:     args.filters,
      defaultPath: args.defaultPath,
    })
    return result.canceled ? null : result.filePath
  })

  ipcMain.handle('model:unloadAll', async (): Promise<{ success: boolean; error?: string }> => {
    try {
      await axios.post(`${API_BASE_URL}/model/unload-all`, {}, { timeout: 10_000 })
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  ipcMain.handle('model:delete', async (_, modelId: string): Promise<{ success: boolean; error?: string }> => {
    const modelDir = join(getSettings(app.getPath('userData')).modelsDir, modelId)
    try {
      await axios.post(`${API_BASE_URL}/model/unload/${encodeURIComponent(modelId)}`, {}, { timeout: 5000 })
    } catch {
      // unload is best-effort — proceed with deletion anyway
    }
    try {
      await rmAsync(modelDir, { recursive: true, force: true })
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  ipcMain.handle('model:showInFolder', (_, modelId: string) => {
    const modelDir = join(getSettings(app.getPath('userData')).modelsDir, modelId)
    if (existsSync(modelDir)) {
      shell.openPath(modelDir)
    }
  })

  // Read local file → base64 (bypasses file:// restrictions in the renderer)
  ipcMain.handle('fs:readFileBase64', async (_, filePath: string) => {
    const buffer = await readFile(filePath)
    return buffer.toString('base64')
  })

  ipcMain.handle('fs:readScreenshotDataUrl', async (_, filename: string) => {
    const filePath = app.isPackaged
      ? join(process.resourcesPath, 'screenshots', filename)
      : join(app.getAppPath(), 'src/assets', filename)
    const buffer = await readFile(filePath)
    return `data:image/png;base64,${buffer.toString('base64')}`
  })

  // Model management
  ipcMain.handle('model:listDownloaded', () => {
    const modelsDir = getSettings(app.getPath('userData')).modelsDir
    return listDownloadedModels(modelsDir)
  })

  ipcMain.handle('model:isDownloaded', (_, modelId: string): boolean => {
    const modelsDir = getSettings(app.getPath('userData')).modelsDir
    return isModelDownloaded(modelsDir, modelId)
  })

  ipcMain.handle('model:activeDownloads', () =>
    [...activeDownloads.entries()].map(([modelId, progress]) => ({ modelId, ...progress }))
  )

  ipcMain.handle('model:download', async (event, { repoId, modelId, skipPrefixes }: { repoId: string; modelId: string; skipPrefixes?: string[] }) => {
    // Refuse downloads for extensions whose manifest declares them
    // incompatible with the current platform. The download itself would
    // succeed (it's just file transfer) but inference can't run, so the
    // user only ends up with disk usage and no working model. modelId is
    // either "<ext_id>" or "<ext_id>/<node_id>" — strip to ext_id.
    try {
      const extId = modelId.split('/')[0]
      const userExt    = join(getSettings(app.getPath('userData')).extensionsDir, extId, 'manifest.json')
      const builtinExt = join(getBuiltinExtensionsDir(), extId, 'manifest.json')
      const manifestPath = existsSync(userExt) ? userExt : (existsSync(builtinExt) ? builtinExt : null)
      if (manifestPath) {
        const manifest = JSON.parse(await readFile(manifestPath, 'utf-8')) as ParsedManifest
        const plats    = manifest.compatibility?.platforms
        if (plats && !plats.includes(process.platform as 'darwin' | 'linux' | 'win32')) {
          return {
            success: false,
            error:
              `'${manifest.name ?? extId}' is not supported on ${process.platform}. ` +
              `Supported: ${plats.join(', ')}.`,
          }
        }
      }
    } catch (err) {
      logger.warn(`[model:download] compatibility check failed for ${modelId}: ${err}`)
    }

    activeDownloads.set(modelId, { percent: 0 })
    try {
      await downloadModelFromHF(repoId, modelId, (progress) => {
        activeDownloads.set(modelId, progress)
        event.sender.send('model:downloadProgress', { modelId, ...progress })
      }, skipPrefixes, pythonBridge.getApiToken())
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    } finally {
      activeDownloads.delete(modelId)
    }
  })

  // Export mesh to GLB / STL / OBJ
  ipcMain.handle('model:export', async (_, { outputUrl, format }: { outputUrl: string; format: string }) => {
    const win = getWindow()
    if (!win) return { success: false, error: 'No window' }

    const meshPath = outputUrl.replace(/^\/workspace\//, '')
    const baseName = meshPath.split('/').pop()?.replace(/\.\w+$/, '') ?? 'model'

    const result = await dialog.showSaveDialog(win, {
      title: 'Export 3D Model',
      defaultPath: `${baseName}.${format}`,
      filters: [{ name: format.toUpperCase(), extensions: [format] }],
    })
    if (result.canceled || !result.filePath) return { success: false }

    try {
      const response = await axios.get(
        `${API_BASE_URL}/export/${format}?path=${encodeURIComponent(meshPath)}`,
        { responseType: 'arraybuffer' }
      )
      await writeFile(result.filePath, Buffer.from(response.data as ArrayBuffer))
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  // Shell — restrict openExternal to safe web/mail schemes only.
  // Without this, a renderer (or compromised dependency rendering untrusted text)
  // could pass file:// or custom OS-protocol URLs and trigger native handlers.
  const ALLOWED_OPEN_SCHEMES = new Set(['http:', 'https:', 'mailto:'])
  ipcMain.handle('shell:openExternal', (_, url: string) => {
    try {
      const scheme = new URL(url).protocol
      if (!ALLOWED_OPEN_SCHEMES.has(scheme)) {
        logger.warn(`[shell:openExternal] blocked unsupported scheme: ${scheme}`)
        return { success: false, error: `Unsupported URL scheme: ${scheme}` }
      }
      return shell.openExternal(url).then(() => ({ success: true }))
    } catch (err) {
      logger.warn(`[shell:openExternal] invalid URL: ${String(err)}`)
      return { success: false, error: 'Invalid URL' }
    }
  })

  // App info
  ipcMain.handle('app:info', () => ({
    version:   app.getVersion(),
    userData:  app.getPath('userData'),
    modelsDir: getSettings(app.getPath('userData')).modelsDir,
    apiUrl:    API_BASE_URL,
    apiToken:  pythonBridge.getApiToken(),
  }))

  // Settings — seed HF token into main-process env at startup
  {
    const initialToken = getSettings(app.getPath('userData')).hfToken ?? ''
    if (initialToken) {
      process.env['HUGGING_FACE_HUB_TOKEN'] = initialToken
      process.env['HF_TOKEN']               = initialToken
    }
  }

  ipcMain.handle('settings:get', () => {
    return getSettings(app.getPath('userData'))
  })

  ipcMain.handle('settings:set', async (_event, patch: { modelsDir?: string; workspaceDir?: string; extensionsDir?: string; hfToken?: string }) => {
    const updated = setSettings(app.getPath('userData'), patch)
    // Keep main-process env in sync so child processes spawned after token change inherit it
    if (patch.hfToken !== undefined) {
      process.env['HUGGING_FACE_HUB_TOKEN'] = patch.hfToken
      process.env['HF_TOKEN']               = patch.hfToken
      // Also push the token into the live FastAPI process env so extension
      // subprocesses spawned by ExtensionProcess._build_env() pick it up
      // without requiring a full app restart.
      try {
        await axios.post(`${API_BASE_URL}/settings/hf-token`, { token: patch.hfToken }, { timeout: 3000 })
      } catch { /* FastAPI may not be running yet — ignore */ }
    }
    return updated
  })

  // Directory picker
  ipcMain.handle('fs:selectDirectory', async () => {
    const win = getWindow()
    if (!win) return null
    const result = await dialog.showOpenDialog(win, {
      properties: ['openDirectory', 'createDirectory'],
    })
    return result.canceled ? null : result.filePaths[0]
  })

  // Cache clear — deletes and recreates the gen-cache folder
  // NOTE: userData/cache (lowercase) = Chromium disk cache on Windows (case-insensitive)
  //       → use a dedicated subfolder to avoid collision
  ipcMain.handle('cache:clear', async () => {
    const cacheDir = join(app.getPath('userData'), 'gen-cache')
    try {
      if (existsSync(cacheDir)) {
        await rmAsync(cacheDir, { recursive: true, force: true })
      }
      await mkdir(cacheDir, { recursive: true })
      return { success: true }
    } catch (err) {
      console.error('[cache:clear] error:', err)
      return { success: false, error: String(err) }
    }
  })

  // Workspace filesystem-based persistence
  const workspacePath = (...parts: string[]) =>
    join(getSettings(app.getPath('userData')).workspaceDir, ...parts)

  ipcMain.handle('workspace:listCollections', async () => {
    const base = workspacePath()
    await mkdir(base, { recursive: true })
    const entries = await readdir(base, { withFileTypes: true })
    return entries.filter(e => e.isDirectory()).map(e => e.name)
  })

  ipcMain.handle('workspace:createCollection', async (_, name: string) => {
    await mkdir(workspacePath(name), { recursive: true })
  })

  ipcMain.handle('workspace:renameCollection', async (_, { oldName, newName }: { oldName: string; newName: string }) => {
    await rename(workspacePath(oldName), workspacePath(newName))
  })

  ipcMain.handle('workspace:deleteCollection', async (_, name: string) => {
    await rmAsync(workspacePath(name), { recursive: true, force: true })
  })

  ipcMain.handle('workspace:listJobs', async (_, collection: string) => {
    try {
      const files = await readdir(workspacePath(collection))
      const metas = files.filter(f => f.endsWith('.meta.json'))
      return Promise.all(metas.map(async f => {
        const raw = await readFile(workspacePath(collection, f), 'utf-8')
        return JSON.parse(raw)
      }))
    } catch { return [] }
  })

  ipcMain.handle('workspace:saveJobMeta', async (_, { collection, filename, meta }: { collection: string; filename: string; meta: unknown }) => {
    const metaFile = filename.replace(/\.glb$/, '.meta.json')
    await writeFile(workspacePath(collection, metaFile), JSON.stringify(meta, null, 2), 'utf-8')
  })

  ipcMain.handle('workspace:deleteJob', async (_, { collection, filename }: { collection: string; filename: string }) => {
    await rmAsync(workspacePath(collection, filename), { force: true })
    await rmAsync(workspacePath(collection, filename.replace(/\.glb$/, '.meta.json')), { force: true })
  })

  // Directory utilities for settings
  ipcMain.handle('fs:listDir', async (_, dirPath: string) => {
    try {
      const entries = await readdir(dirPath, { withFileTypes: true })
      return entries.filter(e => e.isDirectory()).map(e => e.name)
    } catch {
      return []
    }
  })

  ipcMain.handle('fs:moveDirectory', async (_, { src, dest }: { src: string; dest: string }) => {
    try {
      await mkdir(dest, { recursive: true })
      await cp(src, dest, { recursive: true })
      await rmAsync(src, { recursive: true, force: true })
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  ipcMain.handle('fs:deleteDirectory', async (_, dirPath: string) => {
    const userData = app.getPath('userData')
    const settings = getSettings(userData)
    const allowedRoots = [
      settings.modelsDir,
      settings.workspaceDir,
      settings.extensionsDir,
      join(userData, 'gen-cache'),
    ]
    const resolved = join(dirPath)
    const isAllowed = allowedRoots.some((root) => resolved.startsWith(root))
    if (!isAllowed) {
      return { success: false, error: 'Path is outside allowed directories' }
    }
    try {
      await rmAsync(resolved, { recursive: true, force: true })
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  // Trusted-extensions allowlist.
  // Source of truth: vendored electron/main/trusted-extensions.json (in this
  // repo, auditable). Optional per-machine override: <userData>/trusted-extensions.json.
  // Replaces the previous design which fetched the list from a single
  // third-party GitHub raw URL — that URL pointed to an account whose name
  // resembled a typo squat, which made the 'trusted' badge tamperable by a
  // single account compromise. Local-ops scope assumes the user wants a
  // static, auditable list rather than a remote one.
  let registryCache: Set<string> | null = null

  function normalizeRepo(r: string): string {
    return r.toLowerCase().replace(/\/$/, '')
  }

  async function fetchTrustedRepos(): Promise<Set<string>> {
    if (registryCache) return registryCache
    const repos = new Set<string>()
    // 1. User override at <userData>/trusted-extensions.json wins if present.
    const userOverride = join(app.getPath('userData'), 'trusted-extensions.json')
    if (existsSync(userOverride)) {
      try {
        const raw  = await readFile(userOverride, 'utf-8')
        const data = JSON.parse(raw) as { trusted_repos?: string[] }
        for (const r of data.trusted_repos ?? []) repos.add(normalizeRepo(r))
      } catch (err) {
        logger.warn(`[trusted-extensions] failed to read ${userOverride}: ${String(err)}`)
      }
    }
    // 2. Fall back to the vendored allowlist bundled with the app.
    if (repos.size === 0) {
      const vendored = vendoredTrustedExtensions as { trusted_repos?: string[] }
      for (const r of vendored.trusted_repos ?? []) repos.add(normalizeRepo(r))
    }
    registryCache = repos
    return repos
  }

  function isTrustedSource(source: string | undefined, trustedRepos: Set<string>): boolean {
    if (!source) return false
    return trustedRepos.has(source.toLowerCase().replace(/\/$/, ''))
  }

  type ParsedManifest = {
    id?: string; name?: string; displayName?: string; version?: string
    description?: string; author?: string | { name?: string }
    source?: string; installed_ref?: string; generator_class?: string
    // extension type
    type?:  'model' | 'process'
    entry?: string
    nodes?: {
      id:                string
      name?:             string
      input?:            'mesh' | 'image' | 'text'
      inputs?:           ('mesh' | 'image' | 'text')[]
      output?:           'mesh' | 'image' | 'text'
      params_schema?:    unknown[]
      hf_repo?:          string
      download_check?:   string
      hf_skip_prefixes?: string[]
    }[]
    compatibility?: {
      platforms?: ('win32' | 'linux' | 'darwin')[]
      requires_cuda?: boolean
      features?: Record<string, { platforms?: ('win32' | 'linux' | 'darwin')[] }>
    }
  }

  function parseExtensionManifest(parsed: ParsedManifest, fallbackId: string, trustedRepos: Set<string>, builtin = false) {
    const common = {
      id:           parsed.id          ?? fallbackId,
      name:         parsed.displayName ?? parsed.name ?? fallbackId,
      version:      parsed.version,
      description:  parsed.description,
      author:       typeof parsed.author === 'string' ? parsed.author : parsed.author?.name,
      trusted:      builtin || isTrustedSource(parsed.source, trustedRepos),
      source:       parsed.source,
      installedRef: parsed.installed_ref,
      builtin,
      // Surface manifest compatibility so the renderer can disable
      // platform-incompatible toggles (e.g. texture on macOS) without a
      // round-trip per render.
      compatibility: parsed.compatibility,
    }

    const nodes = (parsed.nodes ?? []).map(n => ({
      id:             n.id,
      name:           n.name ?? n.id,
      input:          n.input  ?? 'image' as const,
      inputs:         n.inputs,
      output:         n.output ?? 'mesh'  as const,
      paramsSchema:   n.params_schema ?? [],
      hfRepo:         n.hf_repo,
      downloadCheck:  n.download_check,
      hfSkipPrefixes: n.hf_skip_prefixes,
    }))

    if (parsed.type === 'process') {
      return { ...common, type: 'process' as const, entry: parsed.entry ?? 'processor.js', nodes }
    }

    return { ...common, type: 'model' as const, nodes }
  }

  // Extensions — reads user extensions directory + built-in extensions directory
  ipcMain.handle('extensions:list', async () => {
    const userData      = app.getPath('userData')
    const extensionsDir = getSettings(userData).extensionsDir
    const builtinDir    = getBuiltinExtensionsDir()

    const trustedRepos = await fetchTrustedRepos()

    async function readExtensionsFromDir(dir: string, isBuiltin: boolean) {
      if (!existsSync(dir)) return []
      try {
        const entries = await readdir(dir, { withFileTypes: true })
        const dirs    = entries.filter(e => e.isDirectory())
        return Promise.all(dirs.map(async (entry) => {
          const base = { type: 'model' as const, id: entry.name, name: entry.name, trusted: isBuiltin, builtin: isBuiltin, nodes: [] }
          for (const manifestFile of ['manifest.json', 'package.json']) {
            const p = join(dir, entry.name, manifestFile)
            if (existsSync(p)) {
              try {
                const raw    = await readFile(p, 'utf-8')
                const parsed = JSON.parse(raw) as ParsedManifest
                return parseExtensionManifest(parsed, entry.name, trustedRepos, isBuiltin)
              } catch { /* ignore parse errors, fall through */ }
            }
          }
          return base
        }))
      } catch {
        return []
      }
    }

    const [userExts, builtinExts] = await Promise.all([
      readExtensionsFromDir(extensionsDir, false),
      readExtensionsFromDir(builtinDir,    true),
    ])

    // Built-ins come first, then user extensions
    return [...builtinExts, ...userExts]
  })

  // Install an extension from a GitHub repo URL
  ipcMain.handle('extensions:installFromGitHub', async (event, githubUrl: string) => {
    const win    = getWindow()
    const emit   = (data: object) => win?.webContents.send('extensions:installProgress', data)
    const tmpDir = app.getPath('temp')

    let tarPath    = ''
    let extractDir = ''

    try {
      // 1. Parse and validate GitHub URL.
      //
      // Accepted forms (all optionally end with #<ref> or /tree/<ref>
      // or /commit/<sha> to pin a specific commit/tag/branch):
      //   https://github.com/owner/repo
      //   https://github.com/owner/repo#<ref>
      //   https://github.com/owner/repo/tree/<ref>
      //   https://github.com/owner/repo/commit/<sha>
      //
      // The codeload mirror is used instead of the api.github.com tarball
      // endpoint so anonymous downloads aren't subject to GitHub's 60-req/h
      // rate limit and so we can address an exact ref deterministically.
      const parsed  = new URL(githubUrl.trim())
      if (parsed.hostname !== 'github.com') throw new Error('Invalid URL: must be a GitHub repository (github.com)')
      const parts = parsed.pathname.split('/').filter(Boolean)
      if (parts.length < 2) throw new Error('Invalid URL: expected format https://github.com/owner/repo')
      const [owner, repo] = parts

      // Resolve which ref to install. Default 'HEAD' preserves prior behavior
      // (always grab the default branch tip).
      let ref = 'HEAD'
      if (parts[2] === 'tree' && parts[3]) {
        ref = parts.slice(3).join('/')
      } else if (parts[2] === 'commit' && parts[3]) {
        ref = parts[3]
      } else if (parsed.hash) {
        ref = parsed.hash.slice(1) // strip leading '#'
      }
      // Codeload disallows '/' inside the ref segment, so encode it.
      const refPathSegment = encodeURIComponent(ref)

      emit({ step: 'downloading', percent: 0 })

      // 2. Download tarball from codeload (anonymous, no API rate limit).
      const tarballUrl = `https://codeload.github.com/${owner}/${repo}/tar.gz/${refPathSegment}`
      tarPath    = join(tmpDir, `modly-ext-${Date.now()}.tar.gz`)
      extractDir = join(tmpDir, `modly-ext-extract-${Date.now()}`)

      const response = await axios.get(tarballUrl, {
        responseType: 'arraybuffer',
        headers: {
          'Accept':     'application/vnd.github.v3.raw',
          'User-Agent': 'Modly-App',
        },
        onDownloadProgress: (evt) => {
          const pct = evt.total ? Math.round((evt.loaded / evt.total) * 80) : 40
          emit({ step: 'downloading', percent: pct })
        },
      })

      await writeFile(tarPath, Buffer.from(response.data as ArrayBuffer))

      // 3. Extract tarball (GitHub wraps contents in a top-level {owner}-{repo}-{sha}/ folder)
      emit({ step: 'extracting' })
      await mkdir(extractDir, { recursive: true })
      await tar.x({ file: tarPath, cwd: extractDir, strip: 1 })

      // 4. Validate manifest.json
      emit({ step: 'validating' })
      const manifestPath = join(extractDir, 'manifest.json')

      if (!existsSync(manifestPath)) throw new Error('manifest.json missing from repository')

      const manifestRaw = await readFile(manifestPath, 'utf-8')
      const manifest    = JSON.parse(manifestRaw) as ParsedManifest

      if (!manifest.id) throw new Error('manifest.json: required field "id" missing')
      if (!manifest.nodes?.length) throw new Error('manifest.json: required field "nodes" missing or empty')

      // Reject unsupported platforms up front. setup.py would fail anyway,
      // but the failure messages from CUDA-only wheels are confusing on
      // macOS — better to refuse at the install step.
      const compatPlats = manifest.compatibility?.platforms
      if (compatPlats && !compatPlats.includes(process.platform as 'darwin' | 'linux' | 'win32')) {
        throw new Error(
          `Extension "${manifest.id}" is not supported on ${process.platform}. ` +
          `Supported: ${compatPlats.join(', ')}.`,
        )
      }

      const isProcess = manifest.type === 'process'
      const entryFile = manifest.entry ?? 'processor.js'
      const isPythonProcess = isProcess && entryFile.endsWith('.py')

      if (isProcess) {
        // Process extension validation
        if (!existsSync(join(extractDir, entryFile)))
          throw new Error(`manifest.json: entry file "${entryFile}" missing from repository`)
      } else {
        // Model extension validation
        const generatorPath = join(extractDir, 'generator.py')
        if (!existsSync(generatorPath)) throw new Error('generator.py missing from repository')
        if (!manifest.generator_class)  throw new Error('manifest.json: required field "generator_class" missing')
      }

      // Override source field with the actual GitHub URL so trust is based on
      // origin. Also persist the resolved ref so the user can later audit
      // which exact tag/commit/branch this extension was installed from.
      manifest.source        = `https://github.com/${owner}/${repo}`
      manifest.installed_ref = ref
      await writeFile(manifestPath, JSON.stringify(manifest, null, 2), 'utf-8')

      // 5. Copy to extensions directory (overwrite if already present)
      const extensionsDir = getSettings(app.getPath('userData')).extensionsDir
      await mkdir(extensionsDir, { recursive: true })
      const destDir = join(extensionsDir, manifest.id)

      if (existsSync(destDir)) {
        await rmAsync(destDir, { recursive: true, force: true })
      }
      await cp(extractDir, destDir, { recursive: true })

      // Compile TypeScript entry to JS at install time (once, no runtime overhead)
      if (isProcess && entryFile.endsWith('.ts')) {
        emit({ step: 'setting_up', message: 'Compiling TypeScript entry…' })
        const compiledEntry = entryFile.replace(/\.ts$/, '.js')
        buildSync({
          entryPoints: [join(destDir, entryFile)],
          outfile:     join(destDir, compiledEntry),
          bundle:      true,
          platform:    'node',
          format:      'cjs',
          external:    ['electron'],
        })
        manifest.entry = compiledEntry
        await writeFile(join(destDir, 'manifest.json'), JSON.stringify(manifest, null, 2), 'utf-8')
      }

      if (isPythonProcess) {
        // 6a. Python process extension: run setup.py if present (same as model extensions)
        if (existsSync(join(destDir, 'setup.py'))) {
          emit({ step: 'setting_up', message: 'Setting up Python environment…' })
          const { sm: gpuSm, cudaVersion } = await detectGpuInfo()
          try {
            await runExtensionSetup(destDir, gpuSm, cudaVersion, (line) => {
              logger.info(`[ext-setup] ${line}`)
              emit({ step: 'setting_up', message: line })
            })
          } catch (err) {
            logger.warn(`[ext-setup] setup.py failed: ${err}`)
            emit({ step: 'setting_up', message: `Warning: setup failed — ${err}` })
          }
        }
      } else if (isProcess) {
        // 6b. JS process extension: npm install if package.json present
        if (existsSync(join(destDir, 'package.json'))) {
          emit({ step: 'setting_up', message: 'Installing dependencies…' })
          await new Promise<void>((resolve, reject) => {
            const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
            const child = spawn(npm, ['install', '--omit=dev', '--no-audit', '--no-fund'], {
              cwd:   destDir,
              stdio: 'pipe',
            })
            let buf = ''
            const onData = (chunk: Buffer) => {
              buf += chunk.toString()
              const lines = buf.split('\n')
              buf = lines.pop() ?? ''
              for (const raw of lines) {
                const line = raw.replace(/\x1b\[[0-9;]*m/g, '').trim()
                if (line) emit({ step: 'setting_up', message: line })
              }
            }
            child.stdout?.on('data', onData)
            child.stderr?.on('data', onData)
            child.on('close', (code) => code === 0 ? resolve() : reject(new Error(`npm install failed (exit ${code})`)))
            child.on('error', reject)
          })
        }
      } else {
        // 6b. Model extension: run setup.py directly (no FastAPI required)
        if (existsSync(join(destDir, 'setup.py'))) {
          emit({ step: 'setting_up', message: 'Setting up Python environment…' })
          const { sm: gpuSm, cudaVersion } = await detectGpuInfo()
          try {
            await runExtensionSetup(destDir, gpuSm, cudaVersion, (line) => {
              logger.info(`[ext-setup] ${line}`)
              emit({ step: 'setting_up', message: line })
            })
          } catch (setupErr: any) {
            throw new Error(`Extension setup failed: ${setupErr?.message ?? setupErr}`)
          }
        }

        try {
          await axios.post(`${API_BASE_URL}/extensions/reload`, {}, { timeout: 10_000 })
        } catch { /* Python might not be running yet */ }
      }

      emit({ step: 'done', extensionId: manifest.id })

      const trustedRepos = await fetchTrustedRepos()
      const ext = parseExtensionManifest(manifest, manifest.id, trustedRepos)
      return { success: true, extensionId: manifest.id, extension: ext }

    } catch (err) {
      emit({ step: 'error', message: String(err) })
      return { success: false, error: String(err) }
    } finally {
      // Cleanup temp files
      if (tarPath    && existsSync(tarPath))    rmAsync(tarPath,    { force: true }).catch(() => {})
      if (extractDir && existsSync(extractDir)) rmAsync(extractDir, { recursive: true, force: true }).catch(() => {})
    }
  })

  // Uninstall an extension — built-ins cannot be uninstalled
  ipcMain.handle('extensions:uninstall', async (_, extensionId: string) => {
    const userData      = app.getPath('userData')
    const builtinPath   = join(getBuiltinExtensionsDir(), extensionId)
    if (existsSync(builtinPath)) {
      return { success: false, error: `"${extensionId}" is a built-in extension and cannot be uninstalled.` }
    }

    const extensionsDir = getSettings(userData).extensionsDir
    const extPath       = join(extensionsDir, extensionId)
    try {
      // Terminate process runner if it's a process extension
      terminateProcessRunner(extensionId)

      await rmAsync(extPath, { recursive: true, force: true })
      // Hot-reload Python so it stops using the deleted model extension
      try {
        await axios.post(`${API_BASE_URL}/extensions/reload`, {}, { timeout: 10_000 })
      } catch { /* ignore if Python is not running */ }
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  // Re-run setup.py for a model extension (creates the venv if missing).
  //
  // Lookup order: user-installed extensions/<id>, then bundled
  // builtin-extensions/<id>. When only a built-in is present we PROMOTE
  // it (copy to the user dir) before running setup, so the resulting
  // venv survives the next-launch wipe-and-resync of builtin-extensions.
  ipcMain.handle('extensions:repair', async (_, extensionId: string) => {
    try {
      const userData      = app.getPath('userData')
      const extensionsDir = getSettings(userData).extensionsDir
      const userExtDir    = join(extensionsDir, extensionId)
      const builtinExtDir = join(getBuiltinExtensionsDir(), extensionId)

      let extDir: string
      if (existsSync(join(userExtDir, 'setup.py'))) {
        extDir = userExtDir
      } else if (existsSync(join(builtinExtDir, 'setup.py'))) {
        // Promote built-in template into the user-writable extensions dir.
        // Built-ins ship without a venv; the venv we'll build below would
        // be lost on next launch when syncBuiltinExtensions wipes the
        // resource-mirrored directory.
        await mkdir(extensionsDir, { recursive: true })
        await cp(builtinExtDir, userExtDir, { recursive: true })
        extDir = userExtDir
        logger.info(`[ext-repair] promoted built-in '${extensionId}' to ${userExtDir}`)
      } else {
        return { success: false, error: 'No setup.py found for this extension' }
      }

      const { sm: gpuSm, cudaVersion } = await detectGpuInfo()
      await runExtensionSetup(extDir, gpuSm, cudaVersion, (line) => logger.info(`[ext-repair] ${line}`))
      try {
        await axios.post(`${API_BASE_URL}/extensions/reload`, {}, { timeout: 10_000 })
      } catch { /* ignore if Python is not running yet */ }
      return { success: true }
    } catch (err: any) {
      return { success: false, error: `Repair failed: ${err?.message ?? err}` }
    }
  })

  // Trigger Python extension reload (without touching the filesystem)
  ipcMain.handle('extensions:reload', async () => {
    try {
      const res = await axios.post(`${API_BASE_URL}/extensions/reload`, {}, { timeout: 10_000 })
      return { success: true, errors: (res.data as { errors?: Record<string, string> }).errors ?? {} }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  // Run a process extension in an isolated worker thread
  ipcMain.handle('extensions:runProcess', async (_, extensionId: string, input: { filePath?: string; text?: string; nodeId?: string }, params: Record<string, unknown>) => {
    const userData        = app.getPath('userData')
    const { extensionsDir, workspaceDir } = getSettings(userData)

    // Resolve extension directory: check built-ins first, then user extensions
    const builtinExtDir = join(getBuiltinExtensionsDir(), extensionId)
    const userExtDir    = join(extensionsDir, extensionId)
    const extDir        = existsSync(builtinExtDir) ? builtinExtDir : userExtDir

    if (!existsSync(extDir)) return { success: false, error: `Extension "${extensionId}" not found` }

    try {
      const manifestRaw = await readFile(join(extDir, 'manifest.json'), 'utf-8')
      const manifest    = JSON.parse(manifestRaw) as ParsedManifest
      if (manifest.type !== 'process') return { success: false, error: `Extension "${extensionId}" is not a process extension` }

      const entry           = manifest.entry ?? 'processor.js'
      const isPythonEntry   = entry.endsWith('.py')
      const userData        = app.getPath('userData')

      let runner
      if (isPythonEntry) {
        const pythonExe = getExtPythonExe(extDir) ?? getVenvPythonExe(userData)
        runner = getPythonProcessRunner(extensionId, pythonExe, extDir, entry, workspaceDir, app.getPath('temp'))
      } else {
        runner = getProcessRunner(extensionId, extDir, entry, workspaceDir, app.getPath('temp'))
      }

      const result = await runner.run(input, params)
      return { success: true, result }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  // Terminate all process runners on app quit
  app.on('before-quit', () => terminateAllProcessRunners())

  // Auto-updater
  ipcMain.handle('updater:check', async () => {
    if (!app.isPackaged) return { success: false }
    try {
      await autoUpdater.checkForUpdates()
      return { success: true }
    } catch (err) {
      logger.error(`[updater:check] ${err}`)
      return { success: false }
    }
  })

  ipcMain.handle('updater:quitAndInstall', () => {
    app.removeAllListeners('window-all-closed')
    BrowserWindow.getAllWindows().forEach(w => w.destroy())
    autoUpdater.quitAndInstall(true, true)
  })

  // Update FastAPI paths at runtime (without restarting)
  ipcMain.handle('api:updatePaths', async (_event, patch: { modelsDir?: string; workspaceDir?: string; extensionsDir?: string }) => {
    try {
      await axios.post(`${API_BASE_URL}/settings/paths`, {
        models_dir:     patch.modelsDir,
        workspace_dir:  patch.workspaceDir,
        extensions_dir: patch.extensionsDir,
      })
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  // ── Workflows ────────────────────────────────────────────────────────────

  function workflowsDir(): string {
    const dir = getSettings(app.getPath('userData')).workflowsDir
    if (!existsSync(dir)) require('fs').mkdirSync(dir, { recursive: true })
    return dir
  }

  ipcMain.handle('workflows:list', async () => {
    const dir = workflowsDir()
    const files = readdirSync(dir).filter(f => f.endsWith('.json'))
    const workflows = []
    for (const file of files) {
      try {
        const raw = await readFile(join(dir, file), 'utf-8')
        workflows.push(JSON.parse(raw))
      } catch { /* skip corrupted files */ }
    }
    return workflows.sort((a: { updatedAt?: string }, b: { updatedAt?: string }) =>
      (b.updatedAt ?? '').localeCompare(a.updatedAt ?? '')
    )
  })

  ipcMain.handle('workflows:save', async (_, workflow: { id: string; [key: string]: unknown }) => {
    try {
      const path = join(workflowsDir(), `${workflow.id}.json`)
      await writeFile(path, JSON.stringify(workflow, null, 2), 'utf-8')
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  ipcMain.handle('workflows:delete', async (_, id: string) => {
    try {
      await rmAsync(join(workflowsDir(), `${id}.json`), { force: true })
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  ipcMain.handle('workflows:import', async () => {
    const win = getWindow()
    const result = await dialog.showOpenDialog(win!, {
      title: 'Import Workflow',
      filters: [{ name: 'Workflow', extensions: ['json'] }],
      properties: ['openFile'],
    })
    if (result.canceled || result.filePaths.length === 0) return { success: false }
    try {
      const raw = await readFile(result.filePaths[0], 'utf-8')
      const workflow = JSON.parse(raw)
      if (!workflow.id || !workflow.nodes) return { success: false, error: 'Invalid workflow file' }
      await writeFile(join(workflowsDir(), `${workflow.id}.json`), JSON.stringify(workflow, null, 2), 'utf-8')
      return { success: true, workflow }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })

  ipcMain.handle('workflows:export', async (_, workflow: { id: string; name?: string; [key: string]: unknown }) => {
    const win = getWindow()
    const result = await dialog.showSaveDialog(win!, {
      title: 'Export Workflow',
      defaultPath: `${workflow.name ?? workflow.id}.json`,
      filters: [{ name: 'Workflow', extensions: ['json'] }],
    })
    if (result.canceled || !result.filePath) return { success: false }
    try {
      await writeFile(result.filePath, JSON.stringify(workflow, null, 2), 'utf-8')
      return { success: true }
    } catch (err) {
      return { success: false, error: String(err) }
    }
  })
}
