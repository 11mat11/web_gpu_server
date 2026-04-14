import type { FastifyInstance } from 'fastify'
import {
  getGpuAdapter,
  getGpuDevice,
  getAdapterInfo,
  serializeGpuLimits,
} from '../gpu/device.js'

export async function gpuInfoRoute(server: FastifyInstance) {
  // ─── GET /gpu/info ────────────────────────────────────────────────────────
  server.get(
      '/info',
      {
        schema: {
          tags: ['system'],
          summary: 'GPU adapter information',
          response: {
            200: {
              type: 'object',
              properties: {
                available:     { type: 'boolean' },
                vendor:        { type: 'string' },
                architecture:  { type: 'string' },
                description:   { type: 'string' },
                deviceId:      { type: 'number' },
                backend:       { type: 'string' },
                deviceType:    { type: 'string' },
                driver:        { type: 'string' },
                features:      { type: 'array', items: { type: 'string' } },
                limits:        { type: 'object', additionalProperties: { type: 'number' } },
              },
            },
          },
        },
      },
      async (_req, reply) => {
        const adapter = await getGpuAdapter()

        if (!adapter) {
          return reply.send({
            available: false,
            vendor: 'none',
            architecture: 'none',
            description: 'No GPU adapter found on this machine',
            deviceId: 0,
            backend: 'none',
            deviceType: 'none',
            driver: 'none',
            features: [],
            limits: {},
          })
        }

        const info     = await getAdapterInfo(adapter)
        const features = [...adapter.features].map(String)

        const device = await getGpuDevice()
        const limits = serializeGpuLimits(device.limits)

        return reply.send({
          available:    true,
          vendor:       info.vendor,
          architecture: info.architecture,
          description:  info.description,
          deviceId:     info.deviceId,
          backend:      info.backendType,
          deviceType:   info.deviceType,
          driver:       info.driver,
          features,
          limits,
        })
      },
  )

  // ─── GET /gpu/test ────────────────────────────────────────────────────────
  // Runs a tiny compute shader (vector add) to prove the GPU actually works.
  server.get(
      '/test',
      {
        schema: {
          tags: ['system'],
          summary: 'Run a minimal compute shader to verify GPU is functional',
          response: {
            200: {
              type: 'object',
              properties: {
                ok:          { type: 'boolean' },
                durationMs:  { type: 'number' },
                inputA:      { type: 'array', items: { type: 'number' } },
                inputB:      { type: 'array', items: { type: 'number' } },
                result:      { type: 'array', items: { type: 'number' } },
                message:     { type: 'string' },
              },
            },
            500: {
              type: 'object',
              properties: {
                ok:         { type: 'boolean' },
                durationMs: { type: 'number' },
                inputA:     { type: 'array', items: { type: 'number' } },
                inputB:     { type: 'array', items: { type: 'number' } },
                result:     { type: 'array', items: { type: 'number' } },
                message:    { type: 'string' },
              },
            },
          },
        },
      },
      async (_req, reply) => {
        const start = performance.now()

        try {
          const device = await getGpuDevice()

          // Simple vector add: C[i] = A[i] + B[i]
          const inputA = [1, 2, 3, 4, 5, 6, 7, 8]
          const inputB = [10, 20, 30, 40, 50, 60, 70, 80]
          const N = inputA.length

          const wgsl = /* wgsl */ `
          @group(0) @binding(0) var<storage, read>       a      : array<f32>;
          @group(0) @binding(1) var<storage, read>       b      : array<f32>;
          @group(0) @binding(2) var<storage, read_write> result : array<f32>;

          @compute @workgroup_size(8)
          fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let i = gid.x;
            if (i < arrayLength(&a)) {
              result[i] = a[i] + b[i];
            }
          }
        `

          const pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
              module: device.createShaderModule({ code: wgsl }),
              entryPoint: 'main',
            },
          })

          const byteSize = N * Float32Array.BYTES_PER_ELEMENT

          const bufA = makeStorageBuffer(device, new Float32Array(inputA))
          const bufB = makeStorageBuffer(device, new Float32Array(inputB))
          const bufResult = device.createBuffer({
            size: byteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          })
          const bufRead = device.createBuffer({
            size: byteSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          })

          const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: bufA } },
              { binding: 1, resource: { buffer: bufB } },
              { binding: 2, resource: { buffer: bufResult } },
            ],
          })

          const encoder = device.createCommandEncoder()
          const pass = encoder.beginComputePass()
          pass.setPipeline(pipeline)
          pass.setBindGroup(0, bindGroup)
          pass.dispatchWorkgroups(1)
          pass.end()
          encoder.copyBufferToBuffer(bufResult, 0, bufRead, 0, byteSize)
          device.queue.submit([encoder.finish()])

          await bufRead.mapAsync(GPUMapMode.READ)
          const resultArr = Array.from(new Float32Array(bufRead.getMappedRange()))
          bufRead.unmap()

          // Cleanup
          bufA.destroy(); bufB.destroy(); bufResult.destroy(); bufRead.destroy()

          const durationMs = Number((performance.now() - start).toFixed(2))
          const expected = inputA.map((v, i) => v + inputB[i])
          const ok = resultArr.every((v, i) => Math.abs(v - expected[i]) < 0.001)

          return reply.send({
            ok,
            durationMs,
            inputA,
            inputB,
            result: resultArr,
            message: ok
                ? `✅ GPU compute works correctly in ${durationMs} ms`
                : '❌ Result mismatch — GPU compute may be broken',
          })
        } catch (err) {
          const durationMs = Number((performance.now() - start).toFixed(2))
          reply.statusCode = 500
          return reply.send({
            ok: false,
            durationMs,
            inputA: [],
            inputB: [],
            result: [],
            message: `GPU test failed: ${String(err)}`,
          })
        }
      },
  )
}

function makeStorageBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Float32Array(buf.getMappedRange()).set(data)
  buf.unmap()
  return buf
}