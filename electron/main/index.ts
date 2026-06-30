import { app, BrowserWindow, shell, session } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { setupIpcHandlers } from './ipc-handlers'
import { PythonBridge } from './python-bridge'
import { logger, archiveCurrentSession } from './logger'
import { initAutoUpdater } from './updater'
import { syncBuiltinExtensions } from './builtin-sync'

let mainWindow: BrowserWindow | null = null
let pythonBridge: PythonBridge | null = null

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    show: false,
    frame: false,
    backgroundColor: '#111113',
    titleBarStyle: 'hidden',
    icon: join(__dirname, '../../resources/icons/icon.png'),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false,
      // Generation runs can take 30+ minutes; users routinely switch to
      // another window while waiting. Chromium's default timer throttling
      // would slow our 1.2 s poll loop in workflowRunStore down to once per
      // minute when the window is unfocused, which makes the progress bar
      // appear frozen and delays the 'done' transition long after the GLB
      // is on disk.
      backgroundThrottling: false,
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  // Restrict window.open / target=_blank to safe web/mail schemes only.
  // Mirrors the allowlist applied to ipcMain shell:openExternal.
  const ALLOWED_OPEN_SCHEMES = new Set(['http:', 'https:', 'mailto:'])

  mainWindow.webContents.on('before-input-event', (event, input) => {
    const isMacQuitShortcut =
      process.platform === 'darwin' &&
      input.type === 'keyDown' &&
      input.key.toLowerCase() === 'q' &&
      input.meta &&
      !input.control &&
      !input.alt

    if (isMacQuitShortcut) {
      event.preventDefault()
      app.quit()
    }
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    try {
      const scheme = new URL(details.url).protocol
      if (ALLOWED_OPEN_SCHEMES.has(scheme)) {
        shell.openExternal(details.url)
      }
    } catch {
      // invalid URL — silently drop
    }
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.setName('Modly')

process.on('uncaughtException', (err) => {
  logger.error(`Uncaught exception: ${err.stack ?? err.message}`)
  mainWindow?.webContents.send('app:error', err.stack ?? err.message)
})

process.on('unhandledRejection', (reason) => {
  const msg = String(reason)
  logger.error(`Unhandled rejection: ${msg}`)
  mainWindow?.webContents.send('app:error', msg)
})

app.whenReady().then(async () => {
  archiveCurrentSession()
  logger.info(`App started — version ${app.getVersion()}`)
  electronApp.setAppUserModelId('com.modly.app')

  // Clear Chromium disk cache on startup to recover from any corruption
  await session.defaultSession.clearCache()

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // Sync built-in extensions from app resources to userData
  syncBuiltinExtensions()

  // Start Python FastAPI backend
  pythonBridge = new PythonBridge()
  pythonBridge.setWindowGetter(() => mainWindow)
  setupIpcHandlers(pythonBridge, () => mainWindow)
  initAutoUpdater(() => mainWindow)

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  // Modly holds a multi-GB Python subprocess; leaving it running in the
  // Dock after the window closes (the Mac default) is the wrong behavior
  // for this app. Closing the window means quit.
  app.quit()
})

app.on('before-quit', (event) => {
  if (!pythonBridge) return
  event.preventDefault()
  pythonBridge.stop().finally(() => {
    pythonBridge = null
    app.quit()
  })
})
