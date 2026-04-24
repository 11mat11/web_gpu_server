import type { FastifyInstance } from 'fastify'
import { getCudaRuntimeState } from '../cuda/cudaBackend.js'

export async function healthRoute(server: FastifyInstance) {
  server.get(
    '/health',
    {
      schema: {
        tags: ['system'],
        summary: 'Server health check',
        response: {
          200: {
            type: 'object',
            properties: {
              status: { type: 'string' },
              uptime: { type: 'number' },
              timestamp: { type: 'string' },
              cuda: {
                type: 'object',
                properties: {
                  enabled: { type: 'boolean' },
                  reason: { type: 'string' },
                },
              },
            },
          },
        },
      },
    },
    async (_req, reply) => {
      const cuda = getCudaRuntimeState()
      return reply.send({
        status: 'ok',
        uptime: process.uptime(),
        timestamp: new Date().toISOString(),
        cuda,
      })
    },
  )
}
