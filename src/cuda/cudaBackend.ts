import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import type { MatrixMemoryEstimate } from '../gpu/matrixMul.js'

type CudaInputMode = 'random' | 'custom'

type NativeCudaResult = {
  output: Float32Array | null
  generationDurationMs: number | null
  multiplyDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp'
  memoryEstimate: MatrixMemoryEstimate
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
  memoryEstimate: MatrixMemoryEstimate
}

let cachedAddon: NativeCudaAddon | null = null

function getSeed(seed?: number): number {
  if (typeof seed === 'number' && Number.isFinite(seed)) {
    return seed >>> 0
  }
  const coarse = Date.now() >>> 0
  const fine = Number(process.hrtime.bigint() & 0xffffffffn) >>> 0
  return (coarse ^ fine) >>> 0
}

function resolveAddonPath(): string {
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

  throw new Error(
    'CUDA addon is not built. Run "npm run build:native" first and ensure CUDA Toolkit is installed.',
  )
}

function getAddon(): NativeCudaAddon {
  if (cachedAddon) {
    return cachedAddon
  }

  const require = createRequire(import.meta.url)
  const addonPath = resolveAddonPath()
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
    memoryEstimate: result.memoryEstimate,
  }
}



