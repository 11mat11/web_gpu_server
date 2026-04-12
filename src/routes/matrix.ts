import type { FastifyInstance } from 'fastify'
import { z } from 'zod'

import {
  multiplyRandomSquareMatricesWebGpu,
  multiplySquareMatricesWebGpu,
  type MatrixMemoryEstimate,
} from '../gpu/matrixMul.js'
import { getGpuDevice } from '../gpu/device.js'

const BackendSchema = z.enum(['webgpu', 'cuda', 'cpu'])
const InputModeSchema = z.enum(['random', 'custom'])
const TimingSourceSchema = z.enum(['gpu-timestamp', 'cpu-clock'])
const MATRIX_C_RESPONSE_LIMIT = 100
const U32_MAX = 0xffffffff

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

function randomMatrix(size: number, min: number, max: number): Float32Array {
  const out = new Float32Array(size * size)
  const low = Math.min(min, max)
  const high = Math.max(min, max)

  for (let i = 0; i < out.length; i++) {
    out[i] = Math.random() * (high - low) + low
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

function toMiB(bytes: number): number {
  return Number((bytes / (1024 * 1024)).toFixed(3))
}

function createCpuMemoryEstimate(matrixA: Float32Array, matrixB: Float32Array, matrixC: Float32Array): MatrixMemoryEstimate {
  const hostAllocatedBytes = matrixA.byteLength + matrixB.byteLength + matrixC.byteLength
  return {
    gpuAllocatedBytes: 0,
    gpuAllocatedMiB: 0,
    hostAllocatedBytes,
    hostAllocatedMiB: toMiB(hostAllocatedBytes),
  }
}

export async function matrixRoute(server: FastifyInstance) {
  server.post(
    '/multiply',
    {
      schema: {
        tags: ['matrix'],
        summary: 'Matrix multiplication benchmark',
        body: {
          type: 'object',
          description: 'inputMode="random" with backend="webgpu" generates both matrices directly in WGSL. Set optimized=true to use tiled matrix multiplication (shared memory), or false to use naive multiplication.',
          examples: [
            {
              size: 3,
              backend: 'webgpu',
              inputMode: 'custom',
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
              size: 512,
              backend: 'webgpu',
              inputMode: 'random',
              optimized: true,
              randomMin: -1,
              randomMax: 1,
              randomSeed: 123456,
            },
          ],
          properties: {
            size: { type: 'number', default: 512, description: 'NxN matrix size' },
            backend: { type: 'string', enum: ['webgpu', 'cuda', 'cpu'], default: 'webgpu' },
            inputMode: {
              type: 'string',
              enum: ['random', 'custom'],
              default: 'random',
              description: 'random = data generation on selected backend, custom = use matrixA and matrixB from request.',
            },
            optimized: {
              type: 'boolean',
              default: false,
              description: 'If true, uses Tiled Matrix Multiplication (Shared Memory). If false, uses Naive approach.',
            },
            randomMin: { type: 'number', default: 0, description: 'Used in random mode.' },
            randomMax: { type: 'number', default: 9, description: 'Used in random mode.' },
            randomSeed: {
              type: 'number',
              minimum: 0,
              maximum: U32_MAX,
              description: 'Optional deterministic seed for random generation in GPU random mode.',
            },
            matrixA: {
              type: 'array',
              items: {
                type: 'array',
                items: { type: 'number' },
              },
              description: 'Required in custom mode. Format: number[][]. Must be size x size, e.g. [[1,2],[3,4]] for size=2.',
            },
            matrixB: {
              type: 'array',
              items: {
                type: 'array',
                items: { type: 'number' },
              },
              description: 'Required in custom mode. Format: number[][]. Must be size x size, same shape as matrixA.',
            },
          },
        },
        response: {
          200: {
            type: 'object',
            description: 'Response contains timing metadata for generation and multiplication, plus optional matrixC for small sizes.',
            examples: [
              {
                backend: 'webgpu',
                size: 3,
                inputMode: 'custom',
                optimized: true,
                serverDurationMs: 0.456,
                generationDurationMs: null,
                multiplyDurationMs: 0.123,
                totalDurationMs: 0.123,
                timingSource: 'gpu-timestamp',
                memoryEstimate: {
                  gpuAllocatedBytes: 112,
                  gpuAllocatedMiB: 0,
                  hostAllocatedBytes: 108,
                  hostAllocatedMiB: 0,
                },
                gflops: 0,
                matrixC: [
                  [30, 24, 18],
                  [84, 69, 54],
                  [138, 114, 90],
                ],
              },
              {
                backend: 'webgpu',
                size: 256,
                inputMode: 'random',
                optimized: false,
                serverDurationMs: 12.345,
                generationDurationMs: 2.111,
                multiplyDurationMs: 9.876,
                totalDurationMs: 11.987,
                timingSource: 'gpu-timestamp',
                memoryEstimate: {
                  gpuAllocatedBytes: 1048576,
                  gpuAllocatedMiB: 1,
                  hostAllocatedBytes: 262144,
                  hostAllocatedMiB: 0.25,
                },
                gflops: 2.7,
              },
            ],
            properties: {
              backend: { type: 'string' },
              size: { type: 'number' },
              inputMode: { type: 'string' },
              optimized: {
                type: 'boolean',
                description: 'True when tiled matrix multiplication was used. False for naive path or CPU fallback.',
              },
              serverDurationMs: { type: 'number' },
              generationDurationMs: { type: 'number', nullable: true },
              multiplyDurationMs: { type: 'number', nullable: true },
              totalDurationMs: { type: 'number', nullable: true },
              timingSource: { type: 'string', enum: ['gpu-timestamp', 'cpu-clock'] },
              memoryEstimate: {
                type: 'object',
                properties: {
                  gpuAllocatedBytes: { type: 'number' },
                  gpuAllocatedMiB: { type: 'number' },
                  hostAllocatedBytes: { type: 'number' },
                  hostAllocatedMiB: { type: 'number' },
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

      if (body.backend === 'cuda') {
        return reply.code(400).send({
          error: 'unsupported_backend',
          message: 'CUDA backend is not implemented yet. Use webgpu or cpu.',
        })
      }

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
      let effectiveBackend: 'webgpu' | 'cpu' = body.backend === 'cpu' ? 'cpu' : 'webgpu'
      let generationDurationMs: number | null = null
      let multiplyDurationMs: number | null = null
      let totalDurationMs: number | null = null
      let timingSource: z.infer<typeof TimingSourceSchema> = 'cpu-clock'
      let memoryEstimate: MatrixMemoryEstimate | null = null
      let effectiveOptimized = false

      try {
        if (body.backend === 'webgpu') {
          const gpu = await getGpuDevice()
          const tooLargeForGpu = exceedsGpuMatrixLimits(gpu, body.size)

          if (tooLargeForGpu) {
            console.warn(`[Matrix] size=${body.size} exceeds GPU buffer limits, falling back to CPU for correctness.`)
            effectiveBackend = 'cpu'

            if (body.inputMode === 'random') {
              const cpuGenerationStart = performance.now()
              matrixA = randomMatrix(body.size, body.randomMin, body.randomMax)
              matrixB = randomMatrix(body.size, body.randomMin, body.randomMax)
              generationDurationMs = performance.now() - cpuGenerationStart
            }

            const cpuMulStart = performance.now()
            matrixC = multiplySquareMatricesCpu(body.size, matrixA!, matrixB!)
            multiplyDurationMs = performance.now() - cpuMulStart
            totalDurationMs = (generationDurationMs ?? 0) + multiplyDurationMs
            timingSource = 'cpu-clock'
            memoryEstimate = createCpuMemoryEstimate(matrixA!, matrixB!, matrixC)
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
            memoryEstimate = result.memoryEstimate
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
            memoryEstimate = result.memoryEstimate
            effectiveOptimized = body.optimized
          }
        } else {
          if (body.inputMode === 'random') {
            const cpuGenerationStart = performance.now()
            matrixA = randomMatrix(body.size, body.randomMin, body.randomMax)
            matrixB = randomMatrix(body.size, body.randomMin, body.randomMax)
            generationDurationMs = performance.now() - cpuGenerationStart
          }

          const cpuMulStart = performance.now()
          matrixC = multiplySquareMatricesCpu(body.size, matrixA!, matrixB!)
          multiplyDurationMs = performance.now() - cpuMulStart
          totalDurationMs = (generationDurationMs ?? 0) + multiplyDurationMs
          timingSource = 'cpu-clock'
          memoryEstimate = createCpuMemoryEstimate(matrixA!, matrixB!, matrixC)
        }
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
        backend: 'webgpu' | 'cpu'
        size: number
        inputMode: 'random' | 'custom'
        optimized: boolean
        serverDurationMs: number
        generationDurationMs: number | null
        multiplyDurationMs: number | null
        totalDurationMs: number | null
        timingSource: z.infer<typeof TimingSourceSchema>
        memoryEstimate: MatrixMemoryEstimate
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
        memoryEstimate: memoryEstimate ?? createCpuMemoryEstimate(matrixA!, matrixB!, matrixC),
        gflops: Number(gflops.toFixed(2)),
      }

      if (body.size <= MATRIX_C_RESPONSE_LIMIT) {
        response.matrixC = toMatrix2D(matrixC, body.size)
      }

      return reply.send(response)
    },
  )
}
