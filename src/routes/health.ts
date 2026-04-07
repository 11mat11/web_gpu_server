import type { FastifyInstance } from 'fastify'

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
            },
          },
        },
      },
    },
    async (_req, reply) => {
      return reply.send({
        status: 'ok',
        uptime: process.uptime(),
        timestamp: new Date().toISOString(),
      })
    },
  )
}
