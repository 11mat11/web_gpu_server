import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

type CudaInputMode = 'random' | 'custom'

type NativeCudaResult = {
  output: Float32Array | null
  generationDurationMs: number | null
  multiplyDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp'
}

type NativeCudaAddon = {
  multiplyMatrixCuda: (params: Record<string, unknown>) => Promise<NativeCudaResult>
}

export interface MultiplyMatrixCudaParams {
  size: number
  inputMode: CudaInputMode
  optimized: boolean
  readback?: boolean
  randomMin?: number
  randomMax?: number
  randomSeed?: number
  matrixA?: Float32Array
  matrixB?: Float32Array
}

export interface MultiplyMatrixCudaResult {
  output: Float32Array | null
  generationDurationMs: number | null
  multiplyDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp'
}

export interface CudaRuntimeState {
  enabled: boolean
  reason: string
}

let cachedAddon: NativeCudaAddon | null = null
let cachedRuntimeState: CudaRuntimeState | null = null

function parseBooleanEnv(raw: string | undefined): boolean | null {
  if (raw === undefined) {
    return null
  }

  const normalized = raw.trim().toLowerCase()
  if (normalized === 'auto') {
    return null
  }

  if (['1', 'true', 'yes', 'on'].includes(normalized)) {
    return true
  }
  if (['0', 'false', 'no', 'off'].includes(normalized)) {
    return false
  }

  console.warn(`[CUDA] Ignoring invalid CUDA_ENABLED value: "${raw}". Use true/false.`)
  return null
}

function hasNvidiaGpu(): boolean {
  try {
    const output = execFileSync('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 2000,
    }).trim()

    return output.length > 0
  } catch {
    return false
  }
}

function getSeed(seed?: number): number {
  if (typeof seed === 'number' && Number.isFinite(seed)) {
    return seed >>> 0
  }
  const coarse = Date.now() >>> 0
  const fine = Number(process.hrtime.bigint() & 0xffffffffn) >>> 0
  return (coarse ^ fine) >>> 0
}

function resolveAddonPath(): string | null {
  const currentFilePath = fileURLToPath(import.meta.url)
  const rootDir = path.resolve(path.dirname(currentFilePath), '..', '..')
  const releasePath = path.join(rootDir, 'build', 'Release', 'cuda_matrix_addon.node')
  if (existsSync(releasePath)) {
    return releasePath
  }

  const debugPath = path.join(rootDir, 'build', 'Debug', 'cuda_matrix_addon.node')
  if (existsSync(debugPath)) {
    return debugPath
  }

  return null
}

export function getCudaRuntimeState(): CudaRuntimeState {
  if (cachedRuntimeState) {
    return cachedRuntimeState
  }

  const envEnabled = parseBooleanEnv(process.env.CUDA_ENABLED)
  if (envEnabled === false) {
    cachedRuntimeState = {
      enabled: false,
      reason: 'disabled by CUDA_ENABLED=false',
    }
    return cachedRuntimeState
  }

  if (!hasNvidiaGpu()) {
    cachedRuntimeState = {
      enabled: false,
      reason: 'no NVIDIA GPU detected (nvidia-smi unavailable or no devices)',
    }
    return cachedRuntimeState
  }

  const addonPath = resolveAddonPath()
  if (!addonPath) {
    cachedRuntimeState = {
      enabled: false,
      reason: 'CUDA addon not built (missing build/Release|Debug/cuda_matrix_addon.node)',
    }
    return cachedRuntimeState
  }

  cachedRuntimeState = {
    enabled: true,
    reason: envEnabled === true ? 'enabled by CUDA_ENABLED=true' : 'enabled (auto-detected NVIDIA + addon present)',
  }
  return cachedRuntimeState
}

function getAddon(): NativeCudaAddon {
  if (cachedAddon) {
    return cachedAddon
  }

  const state = getCudaRuntimeState()
  if (!state.enabled) {
    throw new Error(`CUDA backend is unavailable: ${state.reason}`)
  }

  const require = createRequire(import.meta.url)
  const addonPath = resolveAddonPath()
  if (!addonPath) {
    throw new Error('CUDA addon path could not be resolved at runtime.')
  }
  cachedAddon = require(addonPath) as NativeCudaAddon
  return cachedAddon
}

export async function multiplyMatrixCuda(params: MultiplyMatrixCudaParams): Promise<MultiplyMatrixCudaResult> {
  const addon = getAddon()

  const request: Record<string, unknown> = {
    size: params.size,
    inputMode: params.inputMode,
    optimized: params.optimized,
    readback: params.readback ?? true,
    randomMin: params.randomMin ?? 0,
    randomMax: params.randomMax ?? 1,
  }

  if (params.inputMode === 'random') {
    request.randomSeed = getSeed(params.randomSeed)
  }

  if (params.inputMode === 'custom') {
    if (!params.matrixA || !params.matrixB) {
      throw new Error('Custom CUDA mode requires matrixA and matrixB.')
    }
    request.matrixA = params.matrixA
    request.matrixB = params.matrixB
  }

  const result = await addon.multiplyMatrixCuda(request)
  return {
    output: result.output,
    generationDurationMs: result.generationDurationMs,
    multiplyDurationMs: result.multiplyDurationMs,
    totalDurationMs: result.totalDurationMs,
    timingSource: result.timingSource,
  }
}



