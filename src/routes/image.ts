import type { FastifyInstance } from 'fastify'
import { z } from 'zod'

const ImageBodySchema = z.object({
  filter: z.enum(['gaussian', 'sobel', 'grayscale']).default('gaussian'),
  backend: z.enum(['webgpu', 'cuda', 'cpu']).default('webgpu'),
  width: z.number().int().min(64).max(8192).default(1920),
  height: z.number().int().min(64).max(8192).default(1080),
})

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
              durationMs: { type: 'number' },
              pixelsPerSecond: { type: 'number' },
            },
          },
        },
      },
    },
    async (req, reply) => {
      const body = ImageBodySchema.parse(req.body)
      const start = performance.now()

      // TODO: call gpu/imageFilter.ts
      await new Promise((r) => setTimeout(r, 5))

      const durationMs = performance.now() - start
      const pixels = body.width * body.height
      const pixelsPerSecond = Math.round(pixels / (durationMs / 1000))

      return reply.send({
        filter: body.filter,
        backend: body.backend,
        resolution: `${body.width}x${body.height}`,
        durationMs: Number(durationMs.toFixed(3)),
        pixelsPerSecond,
      })
    },
  )
}
