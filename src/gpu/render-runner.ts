import { readFileSync } from 'node:fs'
import { getGpuDevice } from './device.js'
import { RENDER_HEIGHT, RENDER_WIDTH, SHAPE_FLOATS, getShapeBufferBytes } from '../render/scene.js'

const renderShaderCode = readFileSync(new URL('./shaders/renderSceneRender.wgsl', import.meta.url), 'utf8')
const computeShaderCode = readFileSync(new URL('./shaders/renderSceneCompute.wgsl', import.meta.url), 'utf8')

type TimingSource = 'gpu-timestamp' | 'cpu-clock'

export interface RenderSceneResult {
  rgba: Buffer
  width: number
  height: number
  gpuDurationMs: number
  backendDurationMs: number
  timingSource: TimingSource
  gpuMemoryBytes: number
}

const renderPipelineCache = new WeakMap<GPUDevice, GPURenderPipeline>()
const computePipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>()
const quadBufferCache = new WeakMap<GPUDevice, GPUBuffer>()

const QUAD_VERTICES = new Float32Array([
  -1.0, -1.0,
  1.0, -1.0,
  -1.0, 1.0,
  -1.0, 1.0,
  1.0, -1.0,
  1.0, 1.0,
])

function getQuadBuffer(device: GPUDevice): GPUBuffer {
  const cached = quadBufferCache.get(device)
  if (cached) return cached

  const buffer = device.createBuffer({
    label: 'render-quad-buffer',
    size: QUAD_VERTICES.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buffer, 0, QUAD_VERTICES)
  quadBufferCache.set(device, buffer)
  return buffer
}

function getRenderPipeline(device: GPUDevice): GPURenderPipeline {
  const cached = renderPipelineCache.get(device)
  if (cached) return cached

  const module = device.createShaderModule({
    label: 'render-scene-render-shader',
    code: renderShaderCode,
  })

  const pipeline = device.createRenderPipeline({
    label: 'render-scene-render-pipeline',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vsMain',
      buffers: [
        {
          arrayStride: 8,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: 'fsMain',
      targets: [{ format: 'rgba8unorm' }],
    },
    primitive: {
      topology: 'triangle-list',
    },
    depthStencil: {
      format: 'depth32float',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
  })

  renderPipelineCache.set(device, pipeline)
  return pipeline
}

function getComputePipeline(device: GPUDevice): GPUComputePipeline {
  const cached = computePipelineCache.get(device)
  if (cached) return cached

  const module = device.createShaderModule({
    label: 'render-scene-compute-shader',
    code: computeShaderCode,
  })

  const pipeline = device.createComputePipeline({
    label: 'render-scene-compute-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  })

  computePipelineCache.set(device, pipeline)
  return pipeline
}

function readDurationMsFromTimestampBuffer(buffer: GPUBuffer): number | null {
  const mapped = buffer.getMappedRange()
  const timestamps = new BigUint64Array(mapped.slice(0))
  buffer.unmap()

  if (timestamps.length < 2) return null
  const start = timestamps[0]
  const end = timestamps[1]
  const deltaTicks = end >= start ? end - start : (1n << 64n) - start + end
  return deltaTicks > 0n ? Number(deltaTicks) / 1e6 : null
}

function validateShapes(shapes: Float32Array, count: number): void {
  const expected = Math.max(0, Math.floor(count)) * SHAPE_FLOATS
  if (shapes.length !== expected) {
    throw new Error(`Shape buffer length mismatch. Expected ${expected} floats, got ${shapes.length}.`)
  }
}

export async function renderSceneWebGpuRender(
  shapes: Float32Array,
  count: number,
  width = RENDER_WIDTH,
  height = RENDER_HEIGHT,
): Promise<RenderSceneResult> {
  validateShapes(shapes, count)
  const device = await getGpuDevice()

  const pipeline = getRenderPipeline(device)
  const quadBuffer = getQuadBuffer(device)

  const shapeBuffer = device.createBuffer({
    label: 'render-shapes',
    size: shapes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  const paramsBuffer = device.createBuffer({
    label: 'render-params',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const bytesPerRow = width * 4
  if (bytesPerRow % 256 !== 0) {
    throw new Error(`bytesPerRow must be 256-byte aligned. Got ${bytesPerRow}.`)
  }
  const outputBytes = bytesPerRow * height

  const colorTexture = device.createTexture({
    label: 'render-color',
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  })

  const depthTexture = device.createTexture({
    label: 'render-depth',
    size: { width, height, depthOrArrayLayers: 1 },
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  const readback = device.createBuffer({
    label: 'render-readback',
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })

  const supportsTimestamp = device.features.has('timestamp-query' as GPUFeatureName)
  const querySet = supportsTimestamp ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null
  const queryResolve = querySet
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null
  const queryReadback = querySet
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null

  try {
    device.queue.writeBuffer(shapeBuffer, 0, shapes)
    device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([width, height, 0, 0]))

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: shapeBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    })

    const encoder = device.createCommandEncoder({ label: 'render-scene-render-encoder' })
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: colorTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      ...(querySet
        ? {
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          }
        : {}),
    })

    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.setVertexBuffer(0, quadBuffer)
    pass.draw(6, Math.max(0, Math.floor(count)), 0, 0)
    pass.end()

    encoder.copyTextureToBuffer(
      { texture: colorTexture },
      { buffer: readback, bytesPerRow, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    )

    if (querySet && queryResolve && queryReadback) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0)
      encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, 16)
    }

    const cpuStart = performance.now()
    device.queue.submit([encoder.finish()])

    const waits: Promise<void>[] = [readback.mapAsync(GPUMapMode.READ)]
    if (queryReadback) {
      waits.push(queryReadback.mapAsync(GPUMapMode.READ))
    }
    await Promise.all(waits)

    const mapped = readback.getMappedRange()
    const output = Buffer.from(mapped.slice(0))
    readback.unmap()

    const cpuMs = performance.now() - cpuStart
    let gpuMs = cpuMs
    let timingSource: TimingSource = 'cpu-clock'

    if (queryReadback) {
      const timestampMs = readDurationMsFromTimestampBuffer(queryReadback)
      if (timestampMs !== null) {
        gpuMs = timestampMs
        timingSource = 'gpu-timestamp'
      }
    }

    const gpuMemoryBytes =
      getShapeBufferBytes(count) +
      16 +
      QUAD_VERTICES.byteLength +
      outputBytes +
      width * height * 4 +
      width * height * 4

    return {
      rgba: output,
      width,
      height,
      gpuDurationMs: Number(gpuMs.toFixed(3)),
      backendDurationMs: Number(cpuMs.toFixed(3)),
      timingSource,
      gpuMemoryBytes,
    }
  } finally {
    shapeBuffer.destroy()
    paramsBuffer.destroy()
    colorTexture.destroy()
    depthTexture.destroy()
    readback.destroy()
    if (queryResolve) queryResolve.destroy()
    if (queryReadback) queryReadback.destroy()
    if (querySet) querySet.destroy()
  }
}

export async function renderSceneWebGpuCompute(
  shapes: Float32Array,
  count: number,
  width = RENDER_WIDTH,
  height = RENDER_HEIGHT,
): Promise<RenderSceneResult> {
  validateShapes(shapes, count)
  const device = await getGpuDevice()
  const pipeline = getComputePipeline(device)

  const shapeBuffer = device.createBuffer({
    label: 'render-compute-shapes',
    size: shapes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  const paramsBuffer = device.createBuffer({
    label: 'render-compute-params',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const outputBytes = width * height * 4
  const outputBuffer = device.createBuffer({
    label: 'render-compute-output',
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const readback = device.createBuffer({
    label: 'render-compute-readback',
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })

  const supportsTimestamp = device.features.has('timestamp-query' as GPUFeatureName)
  const querySet = supportsTimestamp ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null
  const queryResolve = querySet
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null
  const queryReadback = querySet
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null

  try {
    device.queue.writeBuffer(shapeBuffer, 0, shapes)
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([width, height, Math.max(0, Math.floor(count)), 0]))

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: shapeBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
      ],
    })

    const encoder = device.createCommandEncoder({ label: 'render-scene-compute-encoder' })
    const pass = encoder.beginComputePass(
      querySet
        ? {
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          }
        : {},
    )

    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    pass.end()

    encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, outputBytes)

    if (querySet && queryResolve && queryReadback) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0)
      encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, 16)
    }

    const cpuStart = performance.now()
    device.queue.submit([encoder.finish()])

    const waits: Promise<void>[] = [readback.mapAsync(GPUMapMode.READ)]
    if (queryReadback) {
      waits.push(queryReadback.mapAsync(GPUMapMode.READ))
    }
    await Promise.all(waits)

    const mapped = readback.getMappedRange()
    const output = Buffer.from(mapped.slice(0))
    readback.unmap()

    const cpuMs = performance.now() - cpuStart
    let gpuMs = cpuMs
    let timingSource: TimingSource = 'cpu-clock'

    if (queryReadback) {
      const timestampMs = readDurationMsFromTimestampBuffer(queryReadback)
      if (timestampMs !== null) {
        gpuMs = timestampMs
        timingSource = 'gpu-timestamp'
      }
    }

    const gpuMemoryBytes = getShapeBufferBytes(count) + 16 + outputBytes + outputBytes

    return {
      rgba: output,
      width,
      height,
      gpuDurationMs: Number(gpuMs.toFixed(3)),
      backendDurationMs: Number(cpuMs.toFixed(3)),
      timingSource,
      gpuMemoryBytes,
    }
  } finally {
    shapeBuffer.destroy()
    paramsBuffer.destroy()
    outputBuffer.destroy()
    readback.destroy()
    if (queryResolve) queryResolve.destroy()
    if (queryReadback) queryReadback.destroy()
    if (querySet) querySet.destroy()
  }
}

