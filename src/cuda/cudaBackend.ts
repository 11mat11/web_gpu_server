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

type NativeMlpLoadResult = {
  status: 'loaded'
  memoryEstimate: {
    gpuAllocatedBytes: number
    gpuAllocatedMiB: number
    hostAllocatedBytes: number
    hostAllocatedMiB: number
  }
}

type NativeMlpPredictResult = {
  logits: Float32Array
  gpuDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp'
}

type NativeCnnPredictResult = {
  logits: Float32Array
  gpuDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp'
  memoryEstimate: {
    gpuAllocatedBytes: number
    gpuAllocatedMiB: number
    hostAllocatedBytes: number
    hostAllocatedMiB: number
  }
}

type NativeMlpUnloadResult = {
  status: 'unloaded'
}

type NativeCudaAddon = {
  multiplyMatrixCuda: (params: Record<string, unknown>) => Promise<NativeCudaResult>
  loadModel: (params: Record<string, unknown>) => Promise<NativeMlpLoadResult>
  predict: (params: Record<string, unknown>) => Promise<NativeMlpPredictResult>
  unloadModel: () => Promise<NativeMlpUnloadResult>
  loadCnnModel: (params: Record<string, unknown>) => Promise<NativeMlpLoadResult>
  predictCnn: (params: Record<string, unknown>) => Promise<NativeCnnPredictResult>
  unloadCnnModel: () => Promise<NativeMlpUnloadResult>
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

export interface CudaMlpMemoryEstimate {
  gpuAllocatedBytes: number
  gpuAllocatedMiB: number
  hostAllocatedBytes: number
  hostAllocatedMiB: number
}

export interface CudaMlpLoadResult {
  status: 'loaded'
  memoryEstimate: CudaMlpMemoryEstimate
}

export interface CudaMlpPredictResult {
  logits: Float32Array
  gpuDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp'
}

export interface CudaCnnPredictResult {
  logits: Float32Array
  gpuDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp'
  memoryEstimate: CudaMlpMemoryEstimate
}

export interface CudaMlpUnloadResult {
  status: 'unloaded'
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

export async function loadModelCuda(weights: Float32Array): Promise<CudaMlpLoadResult> {
  const addon = getAddon()
  const result = await addon.loadModel({ weights })
  return {
    status: result.status,
    memoryEstimate: result.memoryEstimate,
  }
}

export async function predictMlpCuda(input: Float32Array): Promise<CudaMlpPredictResult> {
  const addon = getAddon()
  const result = await addon.predict({ input })
  return {
    logits: result.logits,
    gpuDurationMs: result.gpuDurationMs,
    totalDurationMs: result.totalDurationMs,
    timingSource: result.timingSource,
  }
}

export async function unloadModelCuda(): Promise<CudaMlpUnloadResult> {
  const addon = getAddon()
  const result = await addon.unloadModel()
  return {
    status: result.status,
  }
}

export async function loadCnnModelCuda(weights: Float32Array): Promise<CudaMlpLoadResult> {
  const addon = getAddon()
  const result = await addon.loadCnnModel({ weights })
  return {
    status: result.status,
    memoryEstimate: result.memoryEstimate,
  }
}

export async function predictCnnCuda(input: Float32Array): Promise<CudaCnnPredictResult> {
  const addon = getAddon()
  const result = await addon.predictCnn({ input })
  return {
    logits: result.logits,
    gpuDurationMs: result.gpuDurationMs,
    totalDurationMs: result.totalDurationMs,
    timingSource: result.timingSource,
    memoryEstimate: result.memoryEstimate,
  }
}

export async function unloadCnnModelCuda(): Promise<CudaMlpUnloadResult> {
  const addon = getAddon()
  const result = await addon.unloadCnnModel()
  return {
    status: result.status,
  }
}



