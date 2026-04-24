import type { FastifyInstance } from 'fastify'
import { z } from 'zod'

import {
  multiplyRandomSquareMatricesWebGpu,
  multiplySquareMatricesWebGpu,
} from '../gpu/matrixMul.js'
import { getGpuDevice } from '../gpu/device.js'
import { getCudaRuntimeState, multiplyMatrixCuda } from '../cuda/cudaBackend.js'

const BackendSchema = z.enum(['webgpu', 'cuda', 'cpu'])
const InputModeSchema = z.enum(['random', 'custom'])
const TimingSourceSchema = z.enum(['gpu-timestamp', 'cpu-clock'])
const MATRIX_C_RESPONSE_LIMIT = 100
const U32_MAX = 0xffffffff

function getSeed(seed?: number): number {
  if (typeof seed === 'number' && Number.isFinite(seed)) {
    return seed >>> 0
  }

  const coarse = Date.now() >>> 0
  const fine = Number(process.hrtime.bigint() & 0xffffffffn) >>> 0
  return (coarse ^ fine) >>> 0
}

function createMulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const MatrixBodySchema = z.object({
  size: z.number().int().min(1).default(512),
  backend: BackendSchema.default('webgpu'),
  inputMode: InputModeSchema.default('random'),
  optimized: z.boolean().default(false),
  randomMin: z.number().default(0),
  randomMax: z.number().default(9),
  randomSeed: z.number().int().min(0).max(U32_MAX).optional(),
  matrixA: z.array(z.array(z.number().finite())).optional(),
  matrixB: z.array(z.array(z.number().finite())).optional(),
})

function randomMatrix(size: number, min: number, max: number, seed?: number): Float32Array {
  const out = new Float32Array(size * size)
  const low = Math.min(min, max)
  const high = Math.max(min, max)
  const nextRandom = createMulberry32(getSeed(seed))

  for (let i = 0; i < out.length; i++) {
    out[i] = nextRandom() * (high - low) + low
  }

  return out
}

function flattenMatrix(input: number[][], size: number, name: string): Float32Array {
  if (input.length !== size) {
    throw new Error(`${name} must have exactly ${size} rows`)
  }

  const out = new Float32Array(size * size)
  let idx = 0

  for (let row = 0; row < size; row++) {
    if (input[row].length !== size) {
      throw new Error(`${name} row ${row} must have exactly ${size} columns`)
    }

    for (let col = 0; col < size; col++) {
      out[idx++] = input[row][col]
    }
  }

  return out
}

function toMatrix2D(data: Float32Array, size: number): number[][] {
  const out: number[][] = []
  for (let row = 0; row < size; row++) {
    const start = row * size
    out.push(Array.from(data.subarray(start, start + size), (value) => Number(value.toFixed(4))))
  }
  return out
}

function multiplySquareMatricesCpu(size: number, matrixA: Float32Array, matrixB: Float32Array): Float32Array {
  const out = new Float32Array(size * size)

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      let sum = 0
      for (let k = 0; k < size; k++) {
        sum += matrixA[row * size + k] * matrixB[k * size + col]
      }
      out[row * size + col] = sum
    }
  }

  return out
}

function exceedsGpuMatrixLimits(gpu: GPUDevice, size: number): boolean {
  const matrixBytes = size * size * Float32Array.BYTES_PER_ELEMENT
  return matrixBytes > gpu.limits.maxBufferSize || matrixBytes > gpu.limits.maxStorageBufferBindingSize
}

type ProcessMemorySnapshot = {
  rss: number
  heapTotal: number
  heapUsed: number
  external: number
  arrayBuffers: number
}

type ProcessMemoryMetrics = {
  before: ProcessMemorySnapshot
  after: ProcessMemorySnapshot
}

function captureProcessMemory(): ProcessMemorySnapshot {
  const usage = process.memoryUsage()
  return {
    rss: usage.rss,
    heapTotal: usage.heapTotal,
    heapUsed: usage.heapUsed,
    external: usage.external,
    arrayBuffers: usage.arrayBuffers,
  }
}

function buildProcessMemoryMetrics(before: ProcessMemorySnapshot, after: ProcessMemorySnapshot): ProcessMemoryMetrics {
  return {
    before,
    after,
  }
}

export async function matrixRoute(server: FastifyInstance) {
  server.post(
    '/multiply',
    {
      schema: {
        tags: ['matrix'],
        summary: 'Matrix multiplication benchmark (WebGPU/CUDA/CPU)',
        body: {
          type: 'object',
          description:
            'Parametry wejściowe dla mnożenia macierzy NxN. backend wybiera silnik (webgpu/cuda/cpu), inputMode wybiera źródło danych (random/custom), a optimized przełącza kernel naiwny vs tiled dla backendów GPU.',
          examples: [
            {
              size: 512,
              backend: 'webgpu',
              inputMode: 'random',
              optimized: false,
              randomMin: -1,
              randomMax: 1,
              randomSeed: 123456,
            },
            {
              size: 512,
              backend: 'webgpu',
              inputMode: 'random',
              optimized: true,
              randomMin: -1,
              randomMax: 1,
              randomSeed: 123456,
            },
            {
              size: 3,
              backend: 'webgpu',
              inputMode: 'custom',
              optimized: true,
              matrixA: [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
              ],
              matrixB: [
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1],
              ],
            },
            {
              size: 1024,
              backend: 'cuda',
              inputMode: 'random',
              optimized: true,
              randomMin: 0,
              randomMax: 10,
              randomSeed: 42,
            },
            {
              size: 2,
              backend: 'cuda',
              inputMode: 'custom',
              optimized: false,
              matrixA: [
                [1, 2],
                [3, 4],
              ],
              matrixB: [
                [5, 6],
                [7, 8],
              ],
            },
            {
              size: 256,
              backend: 'cpu',
              inputMode: 'random',
              randomMin: -5,
              randomMax: 5,
            },
            {
              size: 2,
              backend: 'cpu',
              inputMode: 'custom',
              matrixA: [
                [2, 0],
                [1, 2],
              ],
              matrixB: [
                [3, 1],
                [4, 2],
              ],
            },
          ],
          properties: {
            size: { type: 'number', default: 512, description: 'NxN matrix size' },
            backend: {
              type: 'string',
              enum: ['webgpu', 'cuda', 'cpu'],
              default: 'webgpu',
              description: 'Silnik obliczeń: webgpu (WGSL), cuda (N-API addon), cpu (fallback/referencja).',
            },
            inputMode: {
              type: 'string',
              enum: ['random', 'custom'],
              default: 'random',
              description: 'random = generacja A/B (GPU dla webgpu/cuda, CPU dla cpu), custom = użyj matrixA i matrixB z requestu.',
            },
            optimized: {
              type: 'boolean',
              default: false,
              description: 'Dotyczy webgpu/cuda: true = tiled/shared memory, false = naive. Dla cpu parametr jest ignorowany.',
            },
            randomMin: { type: 'number', default: 0, description: 'Dolna granica losowania, używana tylko gdy inputMode=random.' },
            randomMax: { type: 'number', default: 9, description: 'Górna granica losowania, używana tylko gdy inputMode=random.' },
            randomSeed: {
              type: 'number',
              minimum: 0,
              maximum: U32_MAX,
              description: 'Opcjonalny seed deterministyczny dla random mode (webgpu/cuda/cpu).',
            },
            matrixA: {
              type: 'array',
              items: {
                type: 'array',
                items: { type: 'number' },
              },
              description: 'Wymagane tylko dla inputMode=custom. Format number[][] o rozmiarze size x size.',
            },
            matrixB: {
              type: 'array',
              items: {
                type: 'array',
                items: { type: 'number' },
              },
              description: 'Wymagane tylko dla inputMode=custom. Format number[][], ten sam rozmiar co matrixA.',
            },
          },
        },
        response: {
          200: {
            type: 'object',
            description:
              'Wynik mnożenia i metryki czasu/pamięci procesu. timingSource=gpu-timestamp dla backendów GPU (webgpu/cuda), timingSource=cpu-clock dla backendu cpu.',
            properties: {
              backend: { type: 'string' },
              size: { type: 'number' },
              inputMode: { type: 'string' },
              optimized: {
                type: 'boolean',
                description: 'Finalnie użyta ścieżka: true = tiled GPU, false = naive lub backend cpu.',
              },
              serverDurationMs: { type: 'number' },
              generationDurationMs: { type: 'number', nullable: true, description: 'Czas generowania danych wejściowych (random mode).' },
              multiplyDurationMs: { type: 'number', nullable: true, description: 'Czas samego mnożenia macierzy.' },
              totalDurationMs: { type: 'number', nullable: true, description: 'Suma generationDurationMs + multiplyDurationMs.' },
              timingSource: {
                type: 'string',
                enum: ['gpu-timestamp', 'cpu-clock'],
                description: 'Źródło pomiaru czasu: znaczniki GPU lub zegar CPU.',
              },
              processMemory: {
                type: 'object',
                description: 'Rzeczywisty snapshot pamięci procesu Node.js (before/after) dla bieżącego requestu.',
                properties: {
                  before: {
                    type: 'object',
                    properties: {
                      rss: { type: 'number' },
                      heapTotal: { type: 'number' },
                      heapUsed: { type: 'number' },
                      external: { type: 'number' },
                      arrayBuffers: { type: 'number' },
                    },
                  },
                  after: {
                    type: 'object',
                    properties: {
                      rss: { type: 'number' },
                      heapTotal: { type: 'number' },
                      heapUsed: { type: 'number' },
                      external: { type: 'number' },
                      arrayBuffers: { type: 'number' },
                    },
                  },
                },
              },
              gflops: { type: 'number' },
              matrixC: {
                type: 'array',
                items: {
                  type: 'array',
                  items: { type: 'number' },
                },
                description: `Included only when size <= ${MATRIX_C_RESPONSE_LIMIT}.`,
              },
            },
          },
          400: {
            type: 'object',
            properties: {
              error: { type: 'string' },
              message: { type: 'string' },
            },
          },
          500: {
            type: 'object',
            properties: {
              error: { type: 'string' },
              message: { type: 'string' },
            },
          },
        },
      },
    },
    async (req, reply) => {
      const serverStart = performance.now()
      const body = MatrixBodySchema.parse(req.body)

      let matrixA: Float32Array | null = null
      let matrixB: Float32Array | null = null

      if (body.inputMode === 'custom') {
        if (!body.matrixA || !body.matrixB) {
          return reply.code(400).send({
            error: 'invalid_input',
            message: 'Custom mode requires matrixA and matrixB.',
          })
        }

        try {
          matrixA = flattenMatrix(body.matrixA, body.size, 'matrixA')
          matrixB = flattenMatrix(body.matrixB, body.size, 'matrixB')
        } catch (err) {
          return reply.code(400).send({
            error: 'invalid_input',
            message: err instanceof Error ? err.message : 'Invalid custom matrices.',
          })
        }
      }

      const computeStart = performance.now()
      let matrixC: Float32Array
      let effectiveBackend: 'webgpu' | 'cuda' | 'cpu' = body.backend
      let generationDurationMs: number | null = null
      let multiplyDurationMs: number | null = null
      let totalDurationMs: number | null = null
      let timingSource: z.infer<typeof TimingSourceSchema> = 'cpu-clock'
      let processMemory: ProcessMemoryMetrics | null = null
      let effectiveOptimized = false
      const processMemoryBefore = captureProcessMemory()

      try {
        if (body.backend === 'webgpu') {
          const gpu = await getGpuDevice()
          const tooLargeForGpu = exceedsGpuMatrixLimits(gpu, body.size)

          if (tooLargeForGpu) {
            console.warn(`[Matrix] size=${body.size} exceeds GPU buffer limits, falling back to CPU for correctness.`)
            effectiveBackend = 'cpu'

            if (body.inputMode === 'random') {
              const cpuGenerationStart = performance.now()
              const baseSeed = getSeed(body.randomSeed)
              matrixA = randomMatrix(body.size, body.randomMin, body.randomMax, baseSeed)
              matrixB = randomMatrix(body.size, body.randomMin, body.randomMax, (baseSeed + 1) >>> 0)
              generationDurationMs = performance.now() - cpuGenerationStart
            }

            const cpuMulStart = performance.now()
            matrixC = multiplySquareMatricesCpu(body.size, matrixA!, matrixB!)
            multiplyDurationMs = performance.now() - cpuMulStart
            totalDurationMs = (generationDurationMs ?? 0) + multiplyDurationMs
            timingSource = 'cpu-clock'
          } else if (body.inputMode === 'random') {
            const result = await multiplyRandomSquareMatricesWebGpu(body.size, body.randomMin, body.randomMax, {
              readback: true,
              seed: body.randomSeed,
              optimized: body.optimized,
            })

            if (!result.output) {
              throw new Error('WebGPU returned null output despite readback: true')
            }

            matrixC = result.output
            generationDurationMs = result.generationDurationMs
            multiplyDurationMs = result.multiplyDurationMs
            totalDurationMs = result.totalDurationMs
            timingSource = result.timingSource
            effectiveOptimized = body.optimized
          } else {
            const result = await multiplySquareMatricesWebGpu(body.size, matrixA!, matrixB!, {
              readback: true,
              optimized: body.optimized,
            })
            if (!result.output) {
              throw new Error('WebGPU returned null output despite readback: true')
            }
            matrixC = result.output
            generationDurationMs = result.generationDurationMs
            multiplyDurationMs = result.multiplyDurationMs
            totalDurationMs = result.totalDurationMs
            timingSource = result.timingSource
            effectiveOptimized = body.optimized
          }
        } else if (body.backend === 'cuda') {
          const cudaState = getCudaRuntimeState()
          if (!cudaState.enabled) {
            return reply.code(400).send({
              error: 'cuda_unavailable',
              message: `CUDA backend is unavailable on this host: ${cudaState.reason}. Use backend=webgpu or backend=cpu.`,
            })
          }

          const result = await multiplyMatrixCuda({
            size: body.size,
            inputMode: body.inputMode,
            optimized: body.optimized,
            readback: true,
            randomMin: body.randomMin,
            randomMax: body.randomMax,
            randomSeed: body.randomSeed,
            matrixA: matrixA ?? undefined,
            matrixB: matrixB ?? undefined,
          })

          if (!result.output) {
            throw new Error('CUDA backend returned null output despite readback: true')
          }

          matrixC = result.output
          generationDurationMs = result.generationDurationMs
          multiplyDurationMs = result.multiplyDurationMs
          totalDurationMs = result.totalDurationMs
          timingSource = result.timingSource
          effectiveOptimized = body.optimized
        } else {
          if (body.inputMode === 'random') {
            const cpuGenerationStart = performance.now()
            const baseSeed = getSeed(body.randomSeed)
            matrixA = randomMatrix(body.size, body.randomMin, body.randomMax, baseSeed)
            matrixB = randomMatrix(body.size, body.randomMin, body.randomMax, (baseSeed + 1) >>> 0)
            generationDurationMs = performance.now() - cpuGenerationStart
          }

          const cpuMulStart = performance.now()
          matrixC = multiplySquareMatricesCpu(body.size, matrixA!, matrixB!)
          multiplyDurationMs = performance.now() - cpuMulStart
          totalDurationMs = (generationDurationMs ?? 0) + multiplyDurationMs
          timingSource = 'cpu-clock'
        }

        processMemory = buildProcessMemoryMetrics(processMemoryBefore, captureProcessMemory())
      } catch (err) {
        return reply.code(500).send({
          error: 'matrix_multiply_failed',
          message: err instanceof Error ? err.message : 'Matrix multiplication failed.',
        })
      }

      const computeDurationMs = performance.now() - computeStart
      const serverDurationMs = performance.now() - serverStart
      const ops = 2 * body.size ** 3
      const gflopsBaseMs = multiplyDurationMs ?? computeDurationMs
      const gflops = gflopsBaseMs > 0 ? ops / (gflopsBaseMs / 1000) / 1e9 : 0

      const response: {
        backend: 'webgpu' | 'cuda' | 'cpu'
        size: number
        inputMode: 'random' | 'custom'
        optimized: boolean
        serverDurationMs: number
        generationDurationMs: number | null
        multiplyDurationMs: number | null
        totalDurationMs: number | null
        timingSource: z.infer<typeof TimingSourceSchema>
        processMemory: ProcessMemoryMetrics
        gflops: number
        matrixC?: number[][]
      } = {
        backend: effectiveBackend,
        size: body.size,
        inputMode: body.inputMode,
        optimized: effectiveOptimized,
        serverDurationMs: Number(serverDurationMs.toFixed(3)),
        generationDurationMs: generationDurationMs === null ? null : Number(generationDurationMs.toFixed(3)),
        multiplyDurationMs: multiplyDurationMs === null ? null : Number(multiplyDurationMs.toFixed(3)),
        totalDurationMs: totalDurationMs === null ? null : Number(totalDurationMs.toFixed(3)),
        timingSource,
        processMemory: processMemory ?? buildProcessMemoryMetrics(processMemoryBefore, captureProcessMemory()),
        gflops: Number(gflops.toFixed(2)),
      }

      if (body.size <= MATRIX_C_RESPONSE_LIMIT) {
        response.matrixC = toMatrix2D(matrixC, body.size)
      }

      return reply.send(response)
    },
  )
}
