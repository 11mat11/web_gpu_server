import type { FastifyInstance } from 'fastify'
import { z } from 'zod'

import { getCudaRuntimeState, gaussianBlurCuda } from '../cuda/cudaBackend.js'
import {
  gaussianBlurCpu,
  gaussianBlurWebGpu,
  packRgbaBytesToU32,
  unpackU32ToRgbaBytes,
} from '../gpu/gaussianBlur.js'

const InputModeSchema = z.enum(['random', 'custom']).describe('Źródło danych wejściowych obrazu.').example('random')
const U32_MAX = 0xffffffff
const MemorySchema = z
  .object({
    gpuBytes: z.number().nullable().describe('Suma bajtów GPUBuffer utworzonych w żądaniu.').example(67108864),
    hostBytes: z.number().nullable().describe('Suma bajtów buforów Buffer/ArrayBuffer utworzonych w żądaniu.').example(33554432),
    serverRssBytes: z.number().describe('process.memoryUsage().rss po zakończeniu obliczeń.').example(123456789),
  })
  .describe('Ujednolicony raport pamięci dla żądania obliczeniowego.')
  .example({ gpuBytes: 67108864, hostBytes: 33554432, serverRssBytes: 123456789 })

const ImageBodySchema = z
  .object({
    filter: z
      .enum(['gaussian', 'sobel', 'grayscale'])
      .default('gaussian')
      .describe('Rodzaj filtra obrazu (aktualnie wspierany: gaussian).')
      .example('gaussian'),
    backend: z
      .enum(['webgpu', 'cuda', 'cpu'])
      .default('webgpu')
      .describe('Backend wykonania filtra (WebGPU/CUDA/CPU).')
      .example('webgpu'),
    inputMode: InputModeSchema.default('random')
      .describe('Tryb wejścia: random generuje obraz na serwerze, custom używa inputBase64.')
      .example('random'),
    seed: z
      .number()
      .int()
      .min(0)
      .max(U32_MAX)
      .optional()
      .describe('Opcjonalny seed dla deterministycznej generacji wejścia (random mode).')
      .example(123456),
    inputBase64: z
      .string()
      .min(1)
      .optional()
      .describe('Wejściowy obraz RGBA w Base64 (width*height*4 bajty).')
      .example('...base64...'),
    width: z
      .number()
      .int()
      .min(64)
      .max(8192)
      .default(1920)
      .describe('Szerokość obrazu w pikselach (RGBA).')
      .example(1920),
    height: z
      .number()
      .int()
      .min(64)
      .max(8192)
      .default(1080)
      .describe('Wysokość obrazu w pikselach (RGBA).')
      .example(1080),
  })
  .describe('Parametry filtrowania obrazu do benchmarku WebGPU/CUDA.')
  .example({ filter: 'gaussian', backend: 'webgpu', inputMode: 'random', width: 1920, height: 1080, seed: 123456 })

function sumByteLengths(
  ...buffers: Array<ArrayBuffer | ArrayBufferView | null | undefined>
): number {
  let total = 0
  for (const buffer of buffers) {
    if (!buffer) continue
    total += buffer.byteLength
  }
  return total
}

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

function createRandomRgbaBytes(width: number, height: number, seed?: number): Uint8Array {
  const totalBytes = width * height * 4
  const out = new Uint8Array(totalBytes)
  const nextRandom = createMulberry32(getSeed(seed))

  for (let i = 0; i < totalBytes; i += 4) {
    out[i] = Math.floor(nextRandom() * 256)
    out[i + 1] = Math.floor(nextRandom() * 256)
    out[i + 2] = Math.floor(nextRandom() * 256)
    out[i + 3] = 255
  }

  return out
}

function asBase64(buffer: Buffer): string {
  return buffer.toString('base64')
}

function decodeBase64Image(inputBase64: string, expectedBytes: number): Uint8Array {
  const buffer = Buffer.from(inputBase64, 'base64')
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(`Invalid input size. Expected ${expectedBytes} bytes, got ${buffer.byteLength}.`)
  }
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
}

/**
 * Benchmarkowe filtrowanie obrazu z metrykami GPU/serwera.
 */
export async function imageRoute(server: FastifyInstance) {
  server.post(
    '/filter',
    {
      schema: {
        tags: ['image'],
        summary: 'Apply image filter (benchmark)',
        body: {
          type: 'object',
          properties: {
            filter: { type: 'string', enum: ['gaussian', 'sobel', 'grayscale'], default: 'gaussian' },
            backend: { type: 'string', enum: ['webgpu', 'cuda', 'cpu'], default: 'webgpu' },
            inputMode: { type: 'string', enum: ['random', 'custom'], default: 'random' },
            seed: { type: 'number', minimum: 0, maximum: U32_MAX },
            inputBase64: { type: 'string' },
            width: { type: 'number', default: 1920 },
            height: { type: 'number', default: 1080 },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              filter: { type: 'string' },
              backend: { type: 'string' },
              resolution: { type: 'string' },
              serverDurationMs: { type: 'number' },
              generationDurationMs: { type: 'number', nullable: true },
              gpuDurationMs: { type: 'number' },
              backendDurationMs: { type: 'number' },
              timingSource: { type: 'string', enum: ['gpu-timestamp', 'cpu-clock'] },
              pixelsPerSecond: { type: 'number' },
              imageBase64: { type: 'string' },
              memory: {
                type: 'object',
                description: 'Ujednolicony raport pamięci dla żądania obliczeniowego.',
                properties: {
                  gpuBytes: { type: ['number', 'null'] },
                  hostBytes: { type: ['number', 'null'] },
                  serverRssBytes: { type: 'number' },
                },
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
      const body = ImageBodySchema.parse(req.body)

      if (body.filter !== 'gaussian') {
        return reply.code(400).send({
          error: 'filter_not_supported',
          message: `Filter "${body.filter}" is not implemented yet.`,
        })
      }

      const expectedBytes = body.width * body.height * 4
      let inputBytes: Uint8Array
      let generationDurationMs: number | null = null

      if (body.inputMode === 'custom') {
        if (!body.inputBase64) {
          return reply.code(400).send({
            error: 'invalid_input',
            message: 'Custom mode requires inputBase64 (RGBA, width*height*4 bytes).',
          })
        }
        try {
          inputBytes = decodeBase64Image(body.inputBase64, expectedBytes)
        } catch (error) {
          return reply.code(400).send({
            error: 'invalid_input',
            message: error instanceof Error ? error.message : 'Invalid inputBase64 payload.',
          })
        }
      } else {
        const generationStart = performance.now()
        inputBytes = createRandomRgbaBytes(body.width, body.height, body.seed)
        generationDurationMs = performance.now() - generationStart
      }

      const inputPacked = packRgbaBytesToU32(inputBytes)

      let gpuDurationMs = 0
      let backendDurationMs = 0
      let timingSource: 'gpu-timestamp' | 'cpu-clock' = 'cpu-clock'
      let imageBase64 = ''
      let gpuBytes: number | null = null
      let hostBytes = 0

      try {
        if (body.backend === 'webgpu') {
          const result = await gaussianBlurWebGpu(body.width, body.height, inputPacked, { readback: true })
          if (!result.output) {
            throw new Error('WebGPU returned null output despite readback: true')
          }
          gpuDurationMs = result.gpuDurationMs
          backendDurationMs = result.backendDurationMs
          timingSource = result.timingSource
          gpuBytes = result.gpuMemoryBytes
          const rgba = unpackU32ToRgbaBytes(result.output)
          const outputBuffer = Buffer.from(rgba)
          imageBase64 = asBase64(outputBuffer)
          hostBytes = sumByteLengths(inputBytes, inputPacked, rgba, outputBuffer)
        } else if (body.backend === 'cuda') {
          const cudaState = getCudaRuntimeState()
          if (!cudaState.enabled) {
            return reply.code(400).send({
              error: 'cuda_unavailable',
              message: `CUDA backend is unavailable on this host: ${cudaState.reason}. Use backend=webgpu or backend=cpu.`,
            })
          }

          const result = await gaussianBlurCuda({
            width: body.width,
            height: body.height,
            input: inputBytes,
            readback: true,
          })
          if (!result.output) {
            throw new Error('CUDA returned null output despite readback: true')
          }
          gpuDurationMs = result.gpuDurationMs
          backendDurationMs = result.backendDurationMs
          timingSource = result.timingSource
          gpuBytes = result.memory.gpuAllocatedBytes
          imageBase64 = asBase64(result.output)
          hostBytes = sumByteLengths(inputBytes, inputPacked, result.output)
        } else {
          const cpuStart = performance.now()
          const cpuOutput = gaussianBlurCpu(inputPacked, body.width, body.height)
          gpuDurationMs = performance.now() - cpuStart
          backendDurationMs = gpuDurationMs
          timingSource = 'cpu-clock'
          const rgba = unpackU32ToRgbaBytes(cpuOutput)
          const outputBuffer = Buffer.from(rgba)
          imageBase64 = asBase64(outputBuffer)
          hostBytes = sumByteLengths(inputBytes, inputPacked, cpuOutput, rgba, outputBuffer)
        }
      } catch (error) {
        return reply.code(500).send({
          error: 'filter_failed',
          message: error instanceof Error ? error.message : 'Image filter failed.',
        })
      }

      const serverDurationMs = performance.now() - serverStart
      const pixels = body.width * body.height
      const pixelsPerSecond = gpuDurationMs > 0 ? Math.round(pixels / (gpuDurationMs / 1000)) : 0
      const memory: z.infer<typeof MemorySchema> = {
        gpuBytes,
        hostBytes,
        serverRssBytes: process.memoryUsage().rss,
      }

      return reply.send({
        filter: body.filter,
        backend: body.backend,
        resolution: `${body.width}x${body.height}`,
        serverDurationMs: Number(serverDurationMs.toFixed(3)),
        generationDurationMs: generationDurationMs === null ? null : Number(generationDurationMs.toFixed(3)),
        gpuDurationMs: Number(gpuDurationMs.toFixed(3)),
        backendDurationMs: Number(backendDurationMs.toFixed(3)),
        timingSource,
        pixelsPerSecond,
        imageBase64,
        memory,
      })
    },
  )
}
