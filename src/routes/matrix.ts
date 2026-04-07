import type { FastifyInstance } from 'fastify'
import { z } from 'zod'

import { multiplySquareMatricesWebGpu } from '../gpu/matrixMul.js'
import { getGpuDevice } from '../gpu/device.js'

const BackendSchema = z.enum(['webgpu', 'cuda', 'cpu'])
const InputModeSchema = z.enum(['random', 'custom'])
const MATRIX_C_RESPONSE_LIMIT = 100

const MatrixBodySchema = z.object({
  size: z.number().int().min(1).default(512),
  backend: BackendSchema.default('webgpu'),
  inputMode: InputModeSchema.default('random'),
  randomMin: z.number().default(0),
  randomMax: z.number().default(9),
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

export async function matrixRoute(server: FastifyInstance) {
  server.post(
    '/multiply',
    {
      schema: {
        tags: ['matrix'],
        summary: 'Matrix multiplication benchmark',
        body: {
          type: 'object',
          description: 'inputMode="random" generates both matrices with random values in range [randomMin, randomMax]. inputMode="custom" uses matrixA and matrixB provided as number[][] with exactly size rows and size columns in each row.',
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
          ],
          properties: {
            size: { type: 'number', default: 512, description: 'NxN matrix size' },
            backend: { type: 'string', enum: ['webgpu', 'cuda', 'cpu'], default: 'webgpu' },
            inputMode: {
              type: 'string',
              enum: ['random', 'custom'],
              default: 'random',
              description: 'random = generate values automatically, custom = use matrixA and matrixB from request.',
            },
            randomMin: { type: 'number', default: 0, description: 'Used only when inputMode=random.' },
            randomMax: { type: 'number', default: 9, description: 'Used only when inputMode=random.' },
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
            description: 'Response contains only the computed result matrixC plus timing metadata.',
            examples: [
              {
                backend: 'webgpu',
                size: 3,
                inputMode: 'custom',
                durationMs: 0.123,
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
                durationMs: 12.345,
                gflops: 2.7,
              },
            ],
            properties: {
              backend: { type: 'string' },
              size: { type: 'number' },
              inputMode: { type: 'string' },
              durationMs: { type: 'number' },
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
      } else {
        matrixA = randomMatrix(body.size, body.randomMin, body.randomMax)
        matrixB = randomMatrix(body.size, body.randomMin, body.randomMax)
      }

      const start = performance.now()
      let matrixC: Float32Array
      let effectiveBackend: 'webgpu' | 'cpu' = body.backend === 'cpu' ? 'cpu' : 'webgpu'

      try {
        if (body.backend === 'webgpu') {
          const gpu = await getGpuDevice()
          const byteLength = body.size * body.size * Float32Array.BYTES_PER_ELEMENT

          if (byteLength > gpu.limits.maxBufferSize || byteLength > gpu.limits.maxStorageBufferBindingSize) {
            console.warn(
              `[Matrix] size=${body.size} exceeds GPU buffer limits, falling back to CPU for correctness.`,
            )
            effectiveBackend = 'cpu'
            matrixC = multiplySquareMatricesCpu(body.size, matrixA!, matrixB!)
          } else {
            matrixC = (await multiplySquareMatricesWebGpu(body.size, matrixA!, matrixB!, { readback: true }))!
          }
        } else {
          matrixC =
            multiplySquareMatricesCpu(body.size, matrixA!, matrixB!)
        }
      } catch (err) {
        return reply.code(500).send({
          error: 'matrix_multiply_failed',
          message: err instanceof Error ? err.message : 'Matrix multiplication failed.',
        })
      }

      const durationMs = performance.now() - start
      const ops = 2 * body.size ** 3
      const gflops = ops / (durationMs / 1000) / 1e9

      const response: {
        backend: 'webgpu' | 'cpu'
        size: number
        inputMode: 'random' | 'custom'
        durationMs: number
        gflops: number
        matrixC?: number[][]
      } = {
        backend: effectiveBackend,
        size: body.size,
        inputMode: body.inputMode,
        durationMs: Number(durationMs.toFixed(3)),
        gflops: Number(gflops.toFixed(2)),
      }

      if (body.size <= MATRIX_C_RESPONSE_LIMIT) {
        response.matrixC = toMatrix2D(matrixC, body.size)
      }

      return reply.send(response)
    },
  )
}
