import type { FastifyInstance } from 'fastify'
import { z } from 'zod'

import { renderSceneWebGpuCompute, renderSceneWebGpuRender } from '../gpu/render-runner.js'
import { generateScene } from '../render/scene.js'
import { getCudaRuntimeState, renderSceneCuda } from '../cuda/cudaBackend.js'

const BackendSchema = z.enum(['webgpu-render', 'webgpu-compute', 'cuda'])

const RenderBodySchema = z.object({
  seed: z.number().int().nonnegative().default(0),
  count: z.number().int().min(1),
  backend: BackendSchema.default('webgpu-render'),
})

function asBase64(buffer: Buffer): string {
  return buffer.toString('base64')
}

export async function renderRoute(server: FastifyInstance) {
  server.post(
    '/',
    {
      schema: {
        tags: ['render'],
        summary: 'Procedural SDF scene renderer (WebGPU render/compute or CUDA)',
        body: {
          type: 'object',
          required: ['seed', 'count', 'backend'],
          properties: {
            seed: { type: 'number', description: 'Deterministic scene seed (LCG)' },
            count: { type: 'number', description: 'Number of shapes to generate' },
            backend: { type: 'string', enum: ['webgpu-render', 'webgpu-compute', 'cuda'] },
          },
          examples: [
            { seed: 1234, count: 2000, backend: 'webgpu-render' },
            { seed: 1234, count: 2000, backend: 'webgpu-compute' },
            { seed: 1234, count: 2000, backend: 'cuda' },
          ],
        },
        response: {
          200: {
            type: 'object',
            properties: {
              backend: { type: 'string', enum: ['webgpu-render', 'webgpu-compute', 'cuda'] },
              width: { type: 'number' },
              height: { type: 'number' },
              format: { type: 'string', enum: ['rgba'] },
              imageBase64: { type: 'string' },
              gpuTimeMs: { type: 'number' },
              serverTimeMs: { type: 'number' },
              timingSource: { type: 'string', enum: ['gpu-timestamp', 'cpu-clock'] },
              gpuMemoryBytes: { type: 'number' },
              serverMemoryBytes: { type: 'number' },
            },
          },
          400: {
            type: 'object',
            properties: { error: { type: 'string' }, message: { type: 'string' } },
          },
          500: {
            type: 'object',
            properties: { error: { type: 'string' }, message: { type: 'string' } },
          },
        },
      },
    },
    async (req, reply) => {
      const parsed = RenderBodySchema.safeParse(req.body)
      if (!parsed.success) {
        return reply.code(400).send({
          error: 'invalid_input',
          message: 'Body must contain seed, count (>=1) and backend (webgpu-render|webgpu-compute|cuda).',
        })
      }

      const startedAt = performance.now()

      try {
        const { seed, count, backend } = parsed.data
        const shapes = generateScene(seed, count)

        let result: { rgba: Buffer; width: number; height: number; gpuTimeMs: number; timingSource: 'gpu-timestamp' | 'cpu-clock'; gpuMemoryBytes: number }

        if (backend === 'webgpu-render') {
          result = await renderSceneWebGpuRender(shapes, count)
        } else if (backend === 'webgpu-compute') {
          result = await renderSceneWebGpuCompute(shapes, count)
        } else {
          const cudaState = getCudaRuntimeState()
          if (!cudaState.enabled) {
            return reply.code(400).send({
              error: 'cuda_unavailable',
              message: `CUDA backend is unavailable on this host: ${cudaState.reason}. Use backend=webgpu-render or backend=webgpu-compute.`,
            })
          }
          result = await renderSceneCuda(shapes, count)
        }

        const serverTimeMs = performance.now() - startedAt
        const serverMemoryBytes = process.memoryUsage().rss

        return reply.send({
          backend,
          width: result.width,
          height: result.height,
          format: 'rgba',
          imageBase64: asBase64(result.rgba),
          gpuTimeMs: Number(result.gpuTimeMs.toFixed(3)),
          serverTimeMs: Number(serverTimeMs.toFixed(3)),
          timingSource: result.timingSource,
          gpuMemoryBytes: result.gpuMemoryBytes,
          serverMemoryBytes,
        })
      } catch (error) {
        return reply.code(500).send({
          error: 'render_failed',
          message: error instanceof Error ? error.message : 'Failed to render scene.',
        })
      }
    },
  )
}

