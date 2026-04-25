import type { FastifyInstance } from 'fastify'

import {
  getAiStatusHandler,
  loadAiModelHandler,
  predictCnnHandler,
  predictMlpHandler,
  unloadAiModelHandler,
} from '../ai/aicontroller'

const aiErrorSchema = {
  type: 'object',
  properties: {
    error: { type: 'string' },
    message: { type: 'string' },
  },
} as const

const backendStatusSchema = {
  type: 'object',
  properties: {
    loaded: { type: 'boolean' },
    available: { type: 'boolean' },
    reason: { type: 'string' },
  },
} as const

const memoryNodeSchema = {
  type: 'object',
  properties: {
    hostAllocatedBytes: { type: 'number' },
    hostAllocatedMiB: { type: 'number' },
    totalGpuAllocatedBytes: { type: 'number' },
    totalGpuAllocatedMiB: { type: 'number' },
    webgpu: {
      type: ['object', 'null'],
      properties: {
        gpuAllocatedBytes: { type: 'number' },
        gpuAllocatedMiB: { type: 'number' },
        hostAllocatedBytes: { type: 'number' },
        hostAllocatedMiB: { type: 'number' },
      },
    },
    cuda: {
      type: ['object', 'null'],
      properties: {
        gpuAllocatedBytes: { type: 'number' },
        gpuAllocatedMiB: { type: 'number' },
        hostAllocatedBytes: { type: 'number' },
        hostAllocatedMiB: { type: 'number' },
      },
    },
  },
} as const

const modelStatusSchema = {
  type: 'object',
  properties: {
    loaded: { type: 'boolean' },
    loadedBackends: {
      type: 'array',
      items: { type: 'string', enum: ['webgpu', 'cuda'] },
    },
    backends: {
      type: 'object',
      properties: {
        webgpu: backendStatusSchema,
        cuda: backendStatusSchema,
      },
    },
    memoryEstimate: memoryNodeSchema,
  },
} as const

export async function ai(server: FastifyInstance) {
  server.get(
    '/status',
    {
      schema: {
        tags: ['ai'],
        summary: 'Pobierz status modeli AI (MLP + CNN)',
        response: {
          200: {
            type: 'object',
            properties: {
              state: { type: 'string', enum: ['idle', 'loading', 'unloading'] },
              loaded: { type: 'boolean' },
              loadedModels: {
                type: 'array',
                items: { type: 'string', enum: ['mlp', 'cnn'] },
              },
              models: {
                type: 'object',
                properties: {
                  mlp: modelStatusSchema,
                  cnn: modelStatusSchema,
                },
              },
              memoryEstimate: {
                type: 'object',
                properties: {
                  hostAllocatedBytes: { type: 'number' },
                  hostAllocatedMiB: { type: 'number' },
                  totalGpuAllocatedBytes: { type: 'number' },
                  totalGpuAllocatedMiB: { type: 'number' },
                  models: {
                    type: 'object',
                    properties: {
                      mlp: memoryNodeSchema,
                      cnn: memoryNodeSchema,
                    },
                  },
                },
              },
            },
          },
          500: aiErrorSchema,
        },
      },
    },
    getAiStatusHandler,
  )

  server.post(
    '/load',
    {
      schema: {
        tags: ['ai'],
        summary: 'Załaduj model(e) AI do backendów WebGPU/CUDA',
        description: 'Jeśli pole `model` jest pominięte, ładowane są oba modele: `mlp` i `cnn`.',
        body: {
          type: 'object',
          additionalProperties: false,
          properties: {
            model: { type: 'string', enum: ['mlp', 'cnn'] },
            webgpu: { type: 'boolean' },
            cuda: { type: 'boolean' },
          },
          examples: [
            { model: 'mlp', webgpu: true, cuda: true },
            { model: 'cnn', webgpu: true, cuda: false },
            { webgpu: true, cuda: true },
          ],
        },
        response: {
          200: {
            type: 'object',
            properties: {
              status: { type: 'string', enum: ['loaded'] },
              loadedModels: { type: 'array', items: { type: 'string', enum: ['mlp', 'cnn'] } },
              models: {
                type: 'object',
                properties: {
                  mlp: modelStatusSchema,
                  cnn: modelStatusSchema,
                },
              },
              memoryEstimate: {
                type: 'object',
                properties: {
                  hostAllocatedBytes: { type: 'number' },
                  hostAllocatedMiB: { type: 'number' },
                  totalGpuAllocatedBytes: { type: 'number' },
                  totalGpuAllocatedMiB: { type: 'number' },
                  models: {
                    type: 'object',
                    properties: {
                      mlp: memoryNodeSchema,
                      cnn: memoryNodeSchema,
                    },
                  },
                },
              },
            },
          },
          400: aiErrorSchema,
          409: aiErrorSchema,
          500: aiErrorSchema,
        },
      },
    },
    loadAiModelHandler,
  )

  server.post(
    '/predict/mlp',
    {
      schema: {
        tags: ['ai'],
        summary: 'Inferencja MLP (MNIST 128x128)',
        body: {
          type: 'object',
          required: ['backend', 'input'],
          properties: {
            backend: { type: 'string', enum: ['cuda', 'webgpu'] },
            input: {
              type: 'array',
              minItems: 16384,
              maxItems: 16384,
              items: { type: 'number' },
            },
          },
          examples: [
            {
              backend: 'cuda',
              input: [0.0, 0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 0.0],
              note: 'Przyklad skrocony; wymagane jest 16384 elementy.',
            },
            {
              backend: 'webgpu',
              input: [0.0, 0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 0.0],
              note: 'Przyklad skrocony; wymagane jest 16384 elementy.',
            },
          ],
        },
        response: {
          200: {
            type: 'object',
            properties: {
              prediction: { type: 'number' },
              probabilities: { type: 'array', items: { type: 'number' }, minItems: 10, maxItems: 10 },
              gpuDurationMs: { type: 'number' },
              totalDurationMs: { type: 'number' },
              timingSource: { type: 'string', enum: ['gpu-timestamp', 'cpu-clock'] },
            },
          },
          400: aiErrorSchema,
          409: aiErrorSchema,
          500: aiErrorSchema,
        },
      },
    },
    predictMlpHandler,
  )

  server.post(
    '/predict/cnn',
    {
      schema: {
        tags: ['ai'],
        summary: 'Inferencja CNN Mini-VGG (CIFAR-10, CHW 3x128x128)',
        body: {
          type: 'object',
          required: ['backend', 'input'],
          properties: {
            backend: { type: 'string', enum: ['cuda', 'webgpu'] },
            input: {
              type: 'array',
              minItems: 49152,
              maxItems: 49152,
              items: { type: 'number' },
            },
          },
          examples: [
            {
              backend: 'cuda',
              input: [0.0, 0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5],
              note: 'Przyklad skrocony; wymagane jest 49152 elementy.',
            },
            {
              backend: 'webgpu',
              input: [0.0, 0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5],
              note: 'Przyklad skrocony; wymagane jest 49152 elementy.',
            },
          ],
        },
        response: {
          200: {
            type: 'object',
            properties: {
              prediction: { type: 'number' },
              predictionLabel: { type: 'string' },
              probabilities: { type: 'array', items: { type: 'number' }, minItems: 10, maxItems: 10 },
              gpuDurationMs: { type: 'number' },
              totalDurationMs: { type: 'number' },
              timingSource: { type: 'string', enum: ['gpu-timestamp', 'cpu-clock'] },
              memoryEstimate: memoryNodeSchema,
            },
          },
          400: aiErrorSchema,
          409: aiErrorSchema,
          500: aiErrorSchema,
        },
      },
    },
    predictCnnHandler,
  )

  server.post(
    '/unload',
    {
      schema: {
        tags: ['ai'],
        summary: 'Zwolnij model(e) AI z backendów WebGPU/CUDA',
        description: 'Jeśli pole `model` jest pominięte, zwalniane są oba modele: `mlp` i `cnn`.',
        body: {
          type: 'object',
          additionalProperties: false,
          properties: {
            model: { type: 'string', enum: ['mlp', 'cnn'] },
            webgpu: { type: 'boolean' },
            cuda: { type: 'boolean' },
          },
          examples: [{ model: 'cnn', cuda: true }, { model: 'mlp', webgpu: true }, { webgpu: true, cuda: true }],
        },
        response: {
          200: {
            type: 'object',
            properties: {
              status: { type: 'string', enum: ['unloaded'] },
              loadedModels: { type: 'array', items: { type: 'string', enum: ['mlp', 'cnn'] } },
              models: {
                type: 'object',
                properties: {
                  mlp: modelStatusSchema,
                  cnn: modelStatusSchema,
                },
              },
              memoryEstimate: {
                type: 'object',
                properties: {
                  hostAllocatedBytes: { type: 'number' },
                  hostAllocatedMiB: { type: 'number' },
                  totalGpuAllocatedBytes: { type: 'number' },
                  totalGpuAllocatedMiB: { type: 'number' },
                  models: {
                    type: 'object',
                    properties: {
                      mlp: memoryNodeSchema,
                      cnn: memoryNodeSchema,
                    },
                  },
                },
              },
            },
          },
          400: aiErrorSchema,
          409: aiErrorSchema,
          500: aiErrorSchema,
        },
      },
    },
    unloadAiModelHandler,
  )
}

