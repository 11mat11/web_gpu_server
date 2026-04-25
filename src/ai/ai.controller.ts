import type { FastifyReply, FastifyRequest } from 'fastify'
import { z } from 'zod'

import { AiManager, AiManagerError } from './AiManager.js'

const LifecycleBodySchema = z
  .object({
    model: z.enum(['mlp', 'cnn']).optional(),
    webgpu: z.boolean().optional(),
    cuda: z.boolean().optional(),
  })
  .partial()

const PredictMlpBodySchema = z.object({
  backend: z.enum(['cuda', 'webgpu']),
  input: z.array(z.number().finite()).length(16384),
})

const PredictCnnBodySchema = z.object({
  backend: z.enum(['cuda', 'webgpu']),
  input: z.array(z.number().finite()).length(49152),
})

function handleAiError(reply: FastifyReply, error: unknown): FastifyReply {
  if (error instanceof AiManagerError) {
    return reply.code(error.statusCode).send({
      error: error.code,
      message: error.message,
    })
  }

  return reply.code(500).send({
    error: 'ai_internal_error',
    message: error instanceof Error ? error.message : 'Unknown AI pipeline error.',
  })
}

export async function loadAiModelHandler(req: FastifyRequest, reply: FastifyReply): Promise<FastifyReply> {
  const manager = AiManager.getInstance()

  let parsedBody: z.infer<typeof LifecycleBodySchema> | undefined
  try {
    const body = req.body === undefined ? {} : req.body
    parsedBody = LifecycleBodySchema.parse(body)
  } catch {
    return reply.code(400).send({
      error: 'invalid_input',
      message: 'Request body may contain optional boolean flags: webgpu and cuda.',
    })
  }

  try {
    const result = await manager.loadModel(parsedBody)
    return reply.send(result)
  } catch (error) {
    return handleAiError(reply, error)
  }
}

export async function predictMlpHandler(req: FastifyRequest, reply: FastifyReply): Promise<FastifyReply> {
  const manager = AiManager.getInstance()

  let parsed: z.infer<typeof PredictMlpBodySchema>
  try {
    parsed = PredictMlpBodySchema.parse(req.body)
  } catch {
    return reply.code(400).send({
      error: 'invalid_input',
      message: 'Request body must contain backend and exactly 16384 finite float values in input.',
    })
  }

  const inputVector = new Float32Array(parsed.input)

  try {
    const result = await manager.predictMlp(parsed.backend, inputVector)
    return reply.send(result)
  } catch (error) {
    return handleAiError(reply, error)
  }
}

export async function predictCnnHandler(req: FastifyRequest, reply: FastifyReply): Promise<FastifyReply> {
  const manager = AiManager.getInstance()

  let parsed: z.infer<typeof PredictCnnBodySchema>
  try {
    parsed = PredictCnnBodySchema.parse(req.body)
  } catch {
    return reply.code(400).send({
      error: 'invalid_input',
      message: 'Request body must contain backend and exactly 49152 finite float values in input.',
    })
  }

  const inputVector = new Float32Array(parsed.input)

  try {
    const result = await manager.predictCnn(parsed.backend, inputVector)
    return reply.send(result)
  } catch (error) {
    return handleAiError(reply, error)
  }
}

export async function unloadAiModelHandler(req: FastifyRequest, reply: FastifyReply): Promise<FastifyReply> {
  const manager = AiManager.getInstance()

  let parsedBody: z.infer<typeof LifecycleBodySchema> | undefined
  try {
    const body = req.body === undefined ? {} : req.body
    parsedBody = LifecycleBodySchema.parse(body)
  } catch {
    return reply.code(400).send({
      error: 'invalid_input',
      message: 'Request body may contain optional boolean flags: webgpu and cuda.',
    })
  }

  try {
    const result = await manager.unloadModel(parsedBody)
    return reply.send(result)
  } catch (error) {
    return handleAiError(reply, error)
  }
}

export async function getAiStatusHandler(_req: FastifyRequest, reply: FastifyReply): Promise<FastifyReply> {
  const manager = AiManager.getInstance()

  try {
    return reply.send(manager.getStatus())
  } catch (error) {
    return handleAiError(reply, error)
  }
}

