import type { FastifyInstance } from 'fastify'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { z } from 'zod'
import { createDedicatedGpuDevice } from '../gpu/device.js'

// ─── Job store ────────────────────────────────────────────────────────────────

interface StressJob {
  id:          string
  requestedMb: number
  actualMb:    number
  bufferCount: number
  durationSec: number
  startedAt:   string
  expiresAt:   string
  status:      'active' | 'released' | 'error'
  error?:      string
}

const activeBuffers = new Map<string, GPUBuffer[]>()
const activeJobs    = new Map<string, StressJob>()
const activeDevices = new Map<string, GPUDevice>()
const activeTimers  = new Map<string, ReturnType<typeof setTimeout>>()

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Dawn max per single GPUBuffer — hard limit from Dawn source */
const CHUNK_BYTES  = 256 * 1024 * 1024   // 256 MB
/** Stable default target when client does not provide explicit value. */
const DEFAULT_MB   = 1024  // 1 GB
/** One touch per page gives a good balance between realism and cost. */
const TOUCH_STEP_BYTES = 4096
const TOUCH_WORKGROUP_SIZE = 256
const STRESS_SHADER_PATH = resolve(process.cwd(), 'src/gpu/shaders/stressTouch.wgsl')

const FALLBACK_STRESS_TOUCH_WGSL = `
struct TouchParams {
  stepWords:  u32,
  totalWords: u32,
  seed:       u32,
  _pad:       u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: TouchParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lane = gid.x;
  let wordIndex = lane * params.stepWords;

  if (wordIndex < params.totalWords) {
    data[wordIndex] = data[wordIndex] ^ (params.seed + lane);
  }
}
`

let stressTouchShaderSourcePromise: Promise<string> | null = null

function getStressTouchShaderSource(): Promise<string> {
  if (!stressTouchShaderSourcePromise) {
    stressTouchShaderSourcePromise = readFile(STRESS_SHADER_PATH, 'utf8')
      .catch((err) => {
        console.warn('[GPU Stress] Could not read stressTouch.wgsl, using inline fallback:', err)
        return FALLBACK_STRESS_TOUCH_WGSL
      })
  }

  return stressTouchShaderSourcePromise
}

function destroyDeviceSafe(device: GPUDevice): void {
  try {
    ;(device as GPUDevice & { destroy?: () => void }).destroy?.()
  } catch {
    /* ignore */
  }
}

function releaseJob(id: string): boolean {
  const timer = activeTimers.get(id)
  if (timer) {
    clearTimeout(timer)
    activeTimers.delete(id)
  }

  const bufs = activeBuffers.get(id)
  const device = activeDevices.get(id)
  if (!bufs && !device) return false

  if (bufs) {
    bufs.forEach(b => { try { b.destroy() } catch { /* already destroyed */ } })
    activeBuffers.delete(id)
  }

  if (device) {
    destroyDeviceSafe(device)
    activeDevices.delete(id)
  }

  const job = activeJobs.get(id)
  if (job) job.status = 'released'
  return true
}

/**
 * Allocates GPU buffers in 256 MB chunks up to targetBytes.
 * Returns actually allocated bytes.
 * Never throws — returns partial result on OOM.
 */
async function allocateVram(device: GPUDevice, targetBytes: number): Promise<GPUBuffer[]> {
  const buffers: GPUBuffer[] = []
  let allocated = 0
  const touchWord = new Uint32Array([0xABABABAB])

  while (allocated < targetBytes) {
    const chunkBytes = Math.min(targetBytes - allocated, CHUNK_BYTES)
    let buf: GPUBuffer | null = null

    try {
      buf = device.createBuffer({
        label: `stress-${buffers.length}`,
        size:  chunkBytes,
        // STORAGE | COPY_DST forces driver to actually back the VRAM pages
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })

      // WebGPU writeBuffer requires write size aligned to 4 bytes.
      // Touch one dword every page to encourage physical backing.
      for (let offset = 0; offset + 4 <= chunkBytes; offset += TOUCH_STEP_BYTES) {
        device.queue.writeBuffer(buf, offset, touchWord)
      }
      await device.queue.onSubmittedWorkDone()

      buffers.push(buf)
      allocated += chunkBytes

      console.log(`[GPU Stress]  chunk ${buffers.length}: +${Math.round(chunkBytes / 1024 / 1024)} MB  (total ${Math.round(allocated / 1024 / 1024)} MB)`)
    } catch (err) {
      if (buf) {
        try { buf.destroy() } catch { /* ignore */ }
      }
      console.warn(`[GPU Stress]  OOM after ${Math.round(allocated / 1024 / 1024)} MB:`, err)
      break
    }
  }

  return buffers
}

async function touchAllocatedBuffers(device: GPUDevice, buffers: GPUBuffer[]): Promise<void> {
  if (buffers.length === 0) return

  const shaderCode = await getStressTouchShaderSource()
  const shaderModule = device.createShaderModule({
    label: 'stress-touch-shader',
    code: shaderCode,
  })

  const pipeline = device.createComputePipeline({
    label: 'stress-touch-pipeline',
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  })

  const paramsBuffer = device.createBuffer({
    label: 'stress-touch-params',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  try {
    const commandBuffers: GPUCommandBuffer[] = []

    for (let i = 0; i < buffers.length; i++) {
      const buffer = buffers[i]
      const totalWords = Math.floor(buffer.size / 4)
      const stepWords = Math.max(1, Math.floor(TOUCH_STEP_BYTES / 4))
      const invocationCount = Math.ceil(totalWords / stepWords)
      const workgroupCount = Math.ceil(invocationCount / TOUCH_WORKGROUP_SIZE)

      const maxWorkgroups = device.limits.maxComputeWorkgroupsPerDimension
      if (workgroupCount > maxWorkgroups) {
        throw new Error(
          `touch dispatch too large for buffer #${i}: ${workgroupCount} > ${maxWorkgroups}`,
        )
      }

      const params = new Uint32Array([
        stepWords,
        totalWords,
        ((i + 1) * 2654435761) >>> 0,
        0,
      ])
      device.queue.writeBuffer(paramsBuffer, 0, params)

      const bindGroup = device.createBindGroup({
        label: `stress-touch-bind-group-${i}`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: paramsBuffer } },
        ],
      })

      const encoder = device.createCommandEncoder({
        label: `stress-touch-encoder-${i}`,
      })
      const pass = encoder.beginComputePass({
        label: `stress-touch-pass-${i}`,
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(workgroupCount)
      pass.end()

      commandBuffers.push(encoder.finish())
    }

    device.queue.submit(commandBuffers)
    await device.queue.onSubmittedWorkDone()
  } finally {
    paramsBuffer.destroy()
  }
}

// ─── Schema ───────────────────────────────────────────────────────────────────

const StressBodySchema = z.object({
  durationSec: z.number().int().min(1).max(300).default(30),
  /** Explicit target in MB (256 – 24576), default 1 GB */
  targetMb: z.number().int().min(256).max(24576).default(DEFAULT_MB),
})

// ─── Route ────────────────────────────────────────────────────────────────────

export async function gpuStressRoute(server: FastifyInstance) {

  // POST /gpu/stress/start ───────────────────────────────────────────────────
  server.post(
      '/start',
      {
        schema: {
          tags: ['system'],
          summary: 'Allocate GPU VRAM for a fixed duration',
          body: {
            type: 'object',
            properties: {
              durationSec: { type: 'number', default: 30,     description: 'Seconds to hold memory (1–300)' },
              targetMb:    { type: 'number', default: DEFAULT_MB, minimum: 256, maximum: 24576, description: 'Explicit MB to allocate (default 1024)' },
            },
          },
          response: {
            200: {
              type: 'object',
              properties: {
                id:          { type: 'string' },
                requestedMb: { type: 'number' },
                actualMb:    { type: 'number' },
                bufferCount: { type: 'number' },
                durationSec: { type: 'number' },
                startedAt:   { type: 'string' },
                expiresAt:   { type: 'string' },
                status:      { type: 'string' },
              },
            },
            500: {
              type: 'object',
              properties: { error: { type: 'string' }, message: { type: 'string' } },
            },
          },
        },
      },
      async (req, reply) => {
        const body = StressBodySchema.parse(req.body ?? {})
        const id = `stress_${Date.now()}`

        let device: GPUDevice
        try {
          device = await createDedicatedGpuDevice(`stress-device-${id}`)
        } catch (err) {
          reply.statusCode = 500
          return reply.send({ error: 'no_gpu', message: String(err) })
        }

        // Total VRAM is not reliably exposed by WebGPU in Node, so use explicit MB.
        const targetMb = body.targetMb
        const targetBytes = targetMb * 1024 * 1024

        // ── Allocate ─────────────────────────────────────────────────────────
        console.log(`[GPU Stress] 🔴 Starting allocation: ${targetMb} MB`)
        let buffers   = await allocateVram(device, targetBytes)

        // If we couldn't allocate anything, try smaller fallbacks
        if (buffers.length === 0) {
          for (const fallbackMb of [512, 256]) {
            if (buffers.length === 0) {
              console.log(`[GPU Stress] ⚠️  Requested ${targetMb} MB failed, trying fallback ${fallbackMb} MB`)
              buffers = await allocateVram(device, fallbackMb * 1024 * 1024)
            }
          }
        }

        try {
          if (buffers.length > 0) {
            await touchAllocatedBuffers(device, buffers)
          }
        } catch (err) {
          buffers.forEach((b) => { try { b.destroy() } catch { /* already destroyed */ } })
          destroyDeviceSafe(device)
          reply.statusCode = 500
          return reply.send({
            error: 'touch_failed',
            message: `Allocated memory but WGSL touch pass failed: ${String(err)}`,
          })
        }

        const actualMb  = Math.round(buffers.reduce((s, b) => s + b.size, 0) / 1024 / 1024)

        const now     = new Date()
        const expires = new Date(now.getTime() + body.durationSec * 1000)

        const job: StressJob = {
          id,
          requestedMb: targetMb,
          actualMb,
          bufferCount: buffers.length,
          durationSec: body.durationSec,
          startedAt:   now.toISOString(),
          expiresAt:   expires.toISOString(),
          status:      buffers.length > 0 ? 'active' : 'error',
          error:       buffers.length === 0 ? 'No buffers could be allocated' : undefined,
        }

        activeJobs.set(id, job)

        if (buffers.length > 0) {
          activeBuffers.set(id, buffers)
          activeDevices.set(id, device)

          const timer = setTimeout(() => {
            const released = releaseJob(id)
            if (released) {
              console.log(`[GPU Stress] ⏱️  Auto-released ${id} (${actualMb} MB freed)`)
            }
          }, body.durationSec * 1000)
          activeTimers.set(id, timer)
        } else {
          destroyDeviceSafe(device)
        }

        if (buffers.length > 0) {
          console.log(`[GPU Stress] ✅ Holding ${actualMb} MB in ${buffers.length} buffer(s) for ${body.durationSec}s (WGSL touch complete)`)
        } else {
          console.log(`[GPU Stress] ❌ Failed to allocate any buffers`)
        }

        return reply.send(job)
      },
  )

  // DELETE /gpu/stress/:id ──────────────────────────────────────────────────
  server.delete(
      '/:id',
      {
        schema: {
          tags: ['system'],
          summary: 'Release stress allocation early',
          params: { type: 'object', properties: { id: { type: 'string' } } },
          response: {
            200: { type: 'object', properties: { released: { type: 'boolean' }, id: { type: 'string' } } },
            404: { type: 'object', properties: { error: { type: 'string' } } },
          },
        },
      },
      async (req, reply) => {
        const { id } = req.params as { id: string }
        const ok = releaseJob(id)
        if (!ok) {
          reply.statusCode = 404
          return reply.send({ error: `Job ${id} not found or already released` })
        }
        console.log(`[GPU Stress] 🟢 Manually released ${id}`)
        return reply.send({ released: true, id })
      },
  )

  // GET /gpu/stress ─────────────────────────────────────────────────────────
  server.get(
      '/',
      {
        schema: {
          tags: ['system'],
          summary: 'List all stress jobs',
          response: {
            200: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  id:          { type: 'string' },
                  requestedMb: { type: 'number' },
                  actualMb:    { type: 'number' },
                  bufferCount: { type: 'number' },
                  durationSec: { type: 'number' },
                  startedAt:   { type: 'string' },
                  expiresAt:   { type: 'string' },
                  status:      { type: 'string' },
                  error:       { type: 'string' },
                },
              },
            },
          },
        },
      },
      async (_req, reply) => reply.send([...activeJobs.values()]),
  )
}