import type { FastifyInstance } from 'fastify'
import * as z from 'zod'

const results: BenchmarkResult[] = []
let runningJob: string | null = null

interface BenchmarkResult {
  id: string
  name: string
  backend: string
  status: 'running' | 'done' | 'error'
  startedAt: string
  finishedAt?: string
  measurements: number[]
  avgMs?: number
  minMs?: number
  maxMs?: number
  error?: string
}

const StartBenchSchema = z.object({
  name: z.enum(['matrix-multiply', 'image-filter', 'video-transcode']),
  backend: z.enum(['webgpu', 'cuda', 'cpu']),
  runs: z.number().int().min(1).max(100).default(10),
  params: z.record(z.string(), z.unknown()).default({}),
})

export async function benchmarkRoute(server: FastifyInstance) {
  // POST /benchmark/start  — start a benchmark job
  server.post(
      '/start',
      {
        schema: {
          tags: ['benchmark'],
          summary: 'Start a benchmark job',
          body: {
            type: 'object',
            required: ['name', 'backend'],
            properties: {
              name: { type: 'string', enum: ['matrix-multiply', 'image-filter', 'video-transcode'] },
              backend: { type: 'string', enum: ['webgpu', 'cuda', 'cpu'] },
              runs: { type: 'number', default: 10 },
              params: { type: 'object' },
            },
          },
        },
      },
      async (req, reply) => {
        if (runningJob) {
          return reply.status(409).send({ error: 'Another benchmark is already running', jobId: runningJob })
        }

        const body = StartBenchSchema.parse(req.body)
        const id = `bench_${Date.now()}`
        runningJob = id

        const result: BenchmarkResult = {
          id,
          name: body.name,
          backend: body.backend,
          status: 'running',
          startedAt: new Date().toISOString(),
          measurements: [],
        }
        results.push(result)

        // Run benchmark in background
        runBenchmarkJob(result, body.runs).finally(() => {
          runningJob = null
        })

        return reply.status(202).send({ jobId: id, status: 'running' })
      },
  )

  // GET /benchmark/status/:id  — poll job status
  server.get(
      '/status/:id',
      {
        schema: {
          tags: ['benchmark'],
          summary: 'Get benchmark job status',
          params: { type: 'object', properties: { id: { type: 'string' } } },
        },
      },
      async (req, reply) => {
        const { id } = req.params as { id: string }
        const job = results.find((r) => r.id === id)
        if (!job) return reply.status(404).send({ error: 'Job not found' })
        return reply.send(job)
      },
  )

  // GET /benchmark/results  — list all past results
  server.get(
      '/results',
      {
        schema: {
          tags: ['benchmark'],
          summary: 'List all benchmark results',
        },
      },
      async (_req, reply) => {
        return reply.send(results)
      },
  )

  // DELETE /benchmark/results  — clear results
  server.delete(
      '/results',
      {
        schema: {
          tags: ['benchmark'],
          summary: 'Clear all benchmark results',
        },
      },
      async (_req, reply) => {
        results.length = 0
        return reply.send({ cleared: true })
      },
  )
}

// ─── Background job runner (placeholder logic) ────────────────────────────────
async function runBenchmarkJob(job: BenchmarkResult, runs: number) {
  try {
    for (let i = 0; i < runs; i++) {
      const start = performance.now()
      // TODO: call the actual implementation module here
      await new Promise((r) => setTimeout(r, 20 + Math.random() * 10))
      job.measurements.push(Number((performance.now() - start).toFixed(3)))
    }

    job.status = 'done'
    job.finishedAt = new Date().toISOString()
    job.avgMs = avg(job.measurements)
    job.minMs = Math.min(...job.measurements)
    job.maxMs = Math.max(...job.measurements)
  } catch (err) {
    job.status = 'error'
    job.error = String(err)
  }
}

function avg(nums: number[]) {
  return Number((nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(3))
}