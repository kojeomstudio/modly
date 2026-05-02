import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import axios from 'axios'

export type BackendStatus = 'not_started' | 'starting' | 'ready' | 'error'
export type SetupStatus = 'idle' | 'checking' | 'needed' | 'installing' | 'done' | 'error'
export interface SetupProgress { step: string; percent: number; currentPackage?: string }

export type GenerationStatus =
  | 'idle'
  | 'uploading'
  | 'generating'
  | 'done'
  | 'error'

export interface GenerationJob {
  id: string
  imageFile: string
  status: GenerationStatus
  progress: number
  step?: string
  outputUrl?: string
  originalOutputUrl?: string   // mesh URL before any optimization
  thumbnailUrl?: string
  modelId?: string             // model used for this generation
  originalTriangles?: number   // polygon count of the original mesh
  generationOptions?: GenerationOptions
  error?: string
  createdAt: number
}

export interface GenerationOptions {
  modelId: string
  remesh: 'quad' | 'triangle' | 'none'
  enableTexture: boolean
  textureResolution: number
  modelParams: Record<string, any>
}

const DEFAULT_OPTIONS: GenerationOptions = {
  modelId: '',
  remesh: 'quad',
  enableTexture: false,
  textureResolution: 512,
  modelParams: {},
}

interface AppState {
  // Backend
  backendStatus: BackendStatus
  apiUrl: string
  apiToken: string
  backendError: string | null

  // Current generation
  currentJob: GenerationJob | null

  // Selected image (shared between ImageUpload and the Generate button)
  selectedImagePath: string | null
  setSelectedImagePath: (path: string | null) => void
  selectedImagePreviewUrl: string | null
  setSelectedImagePreviewUrl: (url: string | null) => void
  selectedImageData: string | null   // base64 content for drag & drop (when path is unavailable)
  setSelectedImageData: (data: string | null) => void

  // Generation options
  generationOptions: GenerationOptions

  // Mesh stats (set by Viewer3D, read by GenerationHUD)
  meshStats: { vertices: number; triangles: number } | null
  setMeshStats: (stats: { vertices: number; triangles: number } | null) => void

  // Setup
  setupStatus:    SetupStatus
  setupProgress:  SetupProgress | null
  setupError:     string | null
  defaultDataDir: string
  checkSetup:     () => Promise<void>
  runSetup:       () => Promise<void>
  saveDataDir:    (baseDir: string) => Promise<void>

  // Patch auto-update
  patchUpdateReady: boolean
  setPatchUpdateReady: (ready: boolean) => void

  // Error modal
  errorModal: string | null
  showError: (message: string) => void
  hideError: () => void

  // Mesh URL history (undo/redo)
  meshHistory: string[]
  historyIndex: number
  pushMeshUrl: (url: string) => void
  undoMesh: () => void
  redoMesh: () => void
  clearMeshHistory: () => void

  // Actions
  initApp: () => Promise<void>
  setCurrentJob: (job: GenerationJob | null) => void
  updateCurrentJob: (patch: Partial<GenerationJob>) => void
  setGenerationOptions: (patch: Partial<GenerationOptions>) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      backendStatus: 'not_started',
      apiUrl: '',
      apiToken: '',
      backendError: null,

      setupStatus: 'idle',
      setupProgress: null,
      setupError: null,
      defaultDataDir: '',

      checkSetup: async () => {
        set({ setupStatus: 'checking' })
        const { needed, defaultDataDir } = await window.electron.setup.check()
        set({ setupStatus: needed ? 'needed' : 'done', defaultDataDir })
      },

      saveDataDir: async (baseDir: string) => {
        await window.electron.setup.saveDataDir(baseDir)
        get().runSetup()
      },

      runSetup: async () => {
        set({ setupStatus: 'installing', setupProgress: null, setupError: null })

        window.electron.setup.offProgress()
        window.electron.setup.offComplete()
        window.electron.setup.offError()

        window.electron.setup.onProgress((data) => {
          set({ setupProgress: data })
        })
        window.electron.setup.onComplete(() => {
          set({ setupStatus: 'done', setupProgress: null })
        })
        window.electron.setup.onError((data) => {
          set({ setupStatus: 'error', setupError: data.message })
        })

        // Fire and forget — progress comes via IPC events
        window.electron.setup.run()
      },

      patchUpdateReady: false,
      setPatchUpdateReady: (ready) => set({ patchUpdateReady: ready }),

      errorModal: null,
      showError: (message) => set({ errorModal: message }),
      hideError: () => set({ errorModal: null }),

      meshHistory: [],
      historyIndex: -1,

      pushMeshUrl: (url) => {
        const { meshHistory, historyIndex } = get()
        const next = [...meshHistory.slice(0, historyIndex + 1), url]
        set({ meshHistory: next, historyIndex: next.length - 1 })
      },

      undoMesh: () => {
        const { meshHistory, historyIndex } = get()
        if (historyIndex <= 0) return
        const newIndex = historyIndex - 1
        set({ historyIndex: newIndex })
        get().updateCurrentJob({ outputUrl: meshHistory[newIndex] })
      },

      redoMesh: () => {
        const { meshHistory, historyIndex } = get()
        if (historyIndex >= meshHistory.length - 1) return
        const newIndex = historyIndex + 1
        set({ historyIndex: newIndex })
        get().updateCurrentJob({ outputUrl: meshHistory[newIndex] })
      },

      clearMeshHistory: () => set({ meshHistory: [], historyIndex: -1 }),

      currentJob: null,
      selectedImagePath: null,
      setSelectedImagePath: (path) => set({ selectedImagePath: path }),
      selectedImagePreviewUrl: null,
      setSelectedImagePreviewUrl: (url) => set({ selectedImagePreviewUrl: url }),
      selectedImageData: null,
      setSelectedImageData: (data) => set({ selectedImageData: data }),
      generationOptions: DEFAULT_OPTIONS,
      meshStats: null,
      setMeshStats: (stats) => set({ meshStats: stats }),
      initApp: async () => {
        set({ backendStatus: 'starting', backendError: null })

        // Texture generation isn't supported on macOS (custom_rasterizer is
        // CUDA-only). Wipe a stale persisted enableTexture=true so the
        // toggle's disabled UI matches the actual value the API sees.
        if (window.electron.platform === 'darwin' && get().generationOptions.enableTexture) {
          set({
            generationOptions: { ...get().generationOptions, enableTexture: false },
          })
        }

        window.electron.python.offCrashed()
        window.electron.python.onCrashed(({ code }) => {
          const msg = `FastAPI process crashed unexpectedly (exit code: ${code ?? 'unknown'})`
          set({ backendStatus: 'error', apiUrl: '', apiToken: '', backendError: msg })
          get().showError(msg)
        })

        try {
          const result = await window.electron.python.start()
          if (!result.success) throw new Error(result.error ?? 'Failed to start backend')
          const { apiUrl, apiToken } = await window.electron.app.info()
          // Inject the per-launch API token into axios defaults so every
          // axios.create({ baseURL: apiUrl }) call (and direct axios.*) carries
          // the X-Modly-Token header. Defends against same-machine cross-origin
          // requests from arbitrary browser tabs hitting 127.0.0.1:8765.
          if (apiToken) {
            axios.defaults.headers.common['X-Modly-Token'] = apiToken
          }
          set({ backendStatus: 'ready', apiUrl, apiToken })
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err)
          set({ backendStatus: 'error', backendError: msg })
          get().showError(msg)
        }
      },

      setCurrentJob: (job) => set({ currentJob: job, meshStats: job === null ? null : get().meshStats }),

      updateCurrentJob: (patch) => {
        const current = get().currentJob
        if (!current) return
        set({ currentJob: { ...current, ...patch } })
      },

      setGenerationOptions: (patch) => {
        set((state) => ({ generationOptions: { ...state.generationOptions, ...patch } }))
      },
    }),
    {
      name: 'modly-store',
      partialize: (state) => ({
        generationOptions: state.generationOptions,
      }),
    }
  )
)
