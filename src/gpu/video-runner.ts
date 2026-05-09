import { readFileSync } from 'node:fs'
import { getGpuDevice } from './device.js'

export type VideoQuality = '1080p' | '720p' | '480p' | '160p'

const SRC_WIDTH = 1920
const SRC_HEIGHT = 1080
const DST_WIDTH_720 = 1280
const DST_HEIGHT_720 = 720
const DST_WIDTH_480 = 854
const DST_HEIGHT_480 = 480
const DST_WIDTH_160 = 284
const DST_HEIGHT_160 = 160
const CHANNELS = 4

const SRC_FRAME_BYTES = SRC_WIDTH * SRC_HEIGHT * CHANNELS
const DST_FRAME_BYTES_720 = DST_WIDTH_720 * DST_HEIGHT_720 * CHANNELS
const DST_FRAME_BYTES_480 = DST_WIDTH_480 * DST_HEIGHT_480 * CHANNELS
const DST_FRAME_BYTES_160 = DST_WIDTH_160 * DST_HEIGHT_160 * CHANNELS
const MAX_DOWNSCALE_BYTES = DST_FRAME_BYTES_720

const PARAM_BYTES = 16

type TimingSource = 'gpu-timestamp' | 'cpu-clock'

export interface VideoRunResult {
  rgba: Buffer
  width: number
  height: number
  gpuDurationMs: number
  backendDurationMs: number
  timingSource: TimingSource
}

export interface LoadedWebGpuVideoPipeline {
  device: GPUDevice
  srcBuffer: GPUBuffer
  dstBuffer: GPUBuffer
  paramsBuffer: GPUBuffer
  readback: GPUBuffer
  pipeline: GPUComputePipeline
  gpuMemoryBytes: number
}

const shaderCode = readFileSync(new URL('./shaders/videoBilinear.wgsl', import.meta.url), 'utf8')
const histogramShaderCode = readFileSync(new URL('../gpu/shaders/histogram.wgsl', import.meta.url), 'utf8')

const HISTOGRAM_BINS_PER_CHANNEL = 256
const HISTOGRAM_CHANNELS = 3
const HISTOGRAM_TOTAL_BINS = HISTOGRAM_BINS_PER_CHANNEL * HISTOGRAM_CHANNELS
const HISTOGRAM_BYTES = HISTOGRAM_TOTAL_BINS * Uint32Array.BYTES_PER_ELEMENT

let histogramPipelineCache: WeakMap<GPUDevice, GPUComputePipeline> | null = null

export const videoLayout = {
  srcWidth: SRC_WIDTH,
  srcHeight: SRC_HEIGHT,
  dstWidth720: DST_WIDTH_720,
  dstHeight720: DST_HEIGHT_720,
  dstWidth480: DST_WIDTH_480,
  dstHeight480: DST_HEIGHT_480,
  dstWidth160: DST_WIDTH_160,
  dstHeight160: DST_HEIGHT_160,
  srcFrameBytes: SRC_FRAME_BYTES,
  dstFrameBytes720: DST_FRAME_BYTES_720,
  dstFrameBytes480: DST_FRAME_BYTES_480,
  dstFrameBytes160: DST_FRAME_BYTES_160,
} as const

function resolveTargetSize(quality: Exclude<VideoQuality, '1080p'>): { width: number; height: number; bytes: number } {
  if (quality === '720p') {
    return { width: DST_WIDTH_720, height: DST_HEIGHT_720, bytes: DST_FRAME_BYTES_720 }
  }
  if (quality === '480p') {
    return { width: DST_WIDTH_480, height: DST_HEIGHT_480, bytes: DST_FRAME_BYTES_480 }
  }
  return { width: DST_WIDTH_160, height: DST_HEIGHT_160, bytes: DST_FRAME_BYTES_160 }
}

function createStorageBuffer(device: GPUDevice, size: number, copySrc = false): GPUBuffer {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (copySrc ? GPUBufferUsage.COPY_SRC : 0),
  })
}

export async function initWebGpuVideoPipeline(): Promise<LoadedWebGpuVideoPipeline> {
  const device = await getGpuDevice()

  const module = device.createShaderModule({
    label: 'video-bilinear-shader',
    code: shaderCode,
  })

  const pipeline = device.createComputePipeline({
    label: 'video-bilinear-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  })

  const srcBuffer = createStorageBuffer(device, SRC_FRAME_BYTES)
  const dstBuffer = createStorageBuffer(device, MAX_DOWNSCALE_BYTES, true)
  const paramsBuffer = device.createBuffer({
    size: PARAM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  const params = new Uint32Array([SRC_WIDTH, SRC_HEIGHT, DST_WIDTH_720, DST_HEIGHT_720])
  device.queue.writeBuffer(paramsBuffer, 0, params)

  const readback = device.createBuffer({
    size: MAX_DOWNSCALE_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })

  const gpuMemoryBytes = SRC_FRAME_BYTES + MAX_DOWNSCALE_BYTES + PARAM_BYTES

  await device.queue.onSubmittedWorkDone()

  return {
    device,
    srcBuffer,
    dstBuffer,
    paramsBuffer,
    readback,
    pipeline,
    gpuMemoryBytes,
  }
}

export async function processVideoFrameWebGpu(
  pipeline: LoadedWebGpuVideoPipeline,
  frameRgba: Uint8Array,
  quality: VideoQuality,
): Promise<VideoRunResult> {
  if (frameRgba.byteLength !== SRC_FRAME_BYTES) {
    throw new Error(`Invalid frame size: expected ${SRC_FRAME_BYTES} bytes.`)
  }

  if (quality === '1080p') {
    return {
      rgba: Buffer.from(frameRgba),
      width: SRC_WIDTH,
      height: SRC_HEIGHT,
      gpuDurationMs: 0,
      backendDurationMs: 0,
      timingSource: 'cpu-clock',
    }
  }

  const target = resolveTargetSize(quality)

  const bindGroup = pipeline.device.createBindGroup({
    layout: pipeline.pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pipeline.paramsBuffer } },
      { binding: 1, resource: { buffer: pipeline.srcBuffer } },
      { binding: 2, resource: { buffer: pipeline.dstBuffer } },
    ],
  })

  pipeline.device.queue.writeBuffer(pipeline.srcBuffer, 0, frameRgba)
  pipeline.device.queue.writeBuffer(
    pipeline.paramsBuffer,
    0,
    new Uint32Array([SRC_WIDTH, SRC_HEIGHT, target.width, target.height]),
  )

  const supportsTimestamp = pipeline.device.features.has('timestamp-query' as GPUFeatureName)
  const querySet = supportsTimestamp
    ? pipeline.device.createQuerySet({ type: 'timestamp', count: 2 })
    : null
  const queryResolve = querySet
    ? pipeline.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : null
  const queryReadback = querySet
    ? pipeline.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    : null

  try {
    const encoder = pipeline.device.createCommandEncoder({ label: 'video-downscale-encoder' })
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

    pass.setPipeline(pipeline.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(target.width / 16), Math.ceil(target.height / 16), 1)
    pass.end()

    encoder.copyBufferToBuffer(pipeline.dstBuffer, 0, pipeline.readback, 0, target.bytes)

    if (querySet && queryResolve && queryReadback) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0)
      encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, 16)
    }

    const cpuStart = performance.now()
    pipeline.device.queue.submit([encoder.finish()])

    const waits: Promise<void>[] = [
      pipeline.device.queue.onSubmittedWorkDone(),
      pipeline.readback.mapAsync(GPUMapMode.READ),
    ]
    if (queryReadback) {
      waits.push(queryReadback.mapAsync(GPUMapMode.READ))
    }
    await Promise.all(waits)

    const mapped = pipeline.readback.getMappedRange(0, target.bytes)
    const out = Buffer.from(mapped.slice(0))
    pipeline.readback.unmap()

    const cpuMs = performance.now() - cpuStart
    let gpuMs = cpuMs
    let timingSource: TimingSource = 'cpu-clock'

    if (queryReadback) {
      const raw = new BigUint64Array(queryReadback.getMappedRange().slice(0))
      queryReadback.unmap()
      if (raw.length >= 2 && raw[1] >= raw[0]) {
        gpuMs = Number(raw[1] - raw[0]) / 1e6
        timingSource = 'gpu-timestamp'
      }
    }

    return {
      rgba: out,
      width: target.width,
      height: target.height,
      gpuDurationMs: Number(gpuMs.toFixed(3)),
      backendDurationMs: Number(cpuMs.toFixed(3)),
      timingSource,
    }
  } finally {
    if (queryResolve) queryResolve.destroy()
    if (queryReadback) queryReadback.destroy()
    if (querySet) querySet.destroy()
  }
}

export function unloadWebGpuVideoPipeline(pipeline: LoadedWebGpuVideoPipeline): void {
  pipeline.srcBuffer.destroy()
  pipeline.dstBuffer.destroy()
  pipeline.paramsBuffer.destroy()
  pipeline.readback.destroy()
}

function getHistogramPipeline(device: GPUDevice): GPUComputePipeline {
  if (!histogramPipelineCache) {
    histogramPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>()
  }

  const cached = histogramPipelineCache.get(device)
  if (cached) return cached

  const module = device.createShaderModule({
    label: 'video-histogram-shader',
    code: histogramShaderCode,
  })

  const pipeline = device.createComputePipeline({
    label: 'video-histogram-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  })

  histogramPipelineCache.set(device, pipeline)
  return pipeline
}

export interface VideoHistogramResult {
  histogram: number[]
  gpuDurationMs: number
  backendDurationMs: number
  timingSource: TimingSource
}

export async function computeHistogramWebGpu(frameRgba: Uint8Array): Promise<VideoHistogramResult> {
  if (frameRgba.byteLength !== SRC_FRAME_BYTES) {
    throw new Error(`Invalid frame size: expected ${SRC_FRAME_BYTES} bytes.`)
  }

  const device = await getGpuDevice()
  const pipeline = getHistogramPipeline(device)

  const inputBuffer = device.createBuffer({
    size: SRC_FRAME_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

  const histogramBuffer = device.createBuffer({
    size: HISTOGRAM_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })

  const histogramReadback = device.createBuffer({
    size: HISTOGRAM_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })

  const supportsTimestamp = device.features.has('timestamp-query' as GPUFeatureName)
  const querySet = supportsTimestamp
    ? device.createQuerySet({ type: 'timestamp', count: 2 })
    : null
  const queryResolve = querySet
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null
  const queryReadback = querySet
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null

  try {
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: histogramBuffer } },
      ],
    })

    const zeros = new Uint32Array(HISTOGRAM_TOTAL_BINS)
    device.queue.writeBuffer(inputBuffer, 0, frameRgba)
    // Reset histogram before each dispatch so calls are independent.
    device.queue.writeBuffer(histogramBuffer, 0, zeros)

    const encoder = device.createCommandEncoder({ label: 'video-histogram-encoder' })
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
    pass.dispatchWorkgroups(Math.ceil((SRC_WIDTH * SRC_HEIGHT) / 256))
    pass.end()

    encoder.copyBufferToBuffer(histogramBuffer, 0, histogramReadback, 0, HISTOGRAM_BYTES)

    if (querySet && queryResolve && queryReadback) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0)
      encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, 16)
    }

    const cpuStart = performance.now()
    device.queue.submit([encoder.finish()])

    const waits: Promise<void>[] = [device.queue.onSubmittedWorkDone(), histogramReadback.mapAsync(GPUMapMode.READ)]
    if (queryReadback) {
      waits.push(queryReadback.mapAsync(GPUMapMode.READ))
    }
    await Promise.all(waits)

    const histogramMapped = histogramReadback.getMappedRange()
    const histogram = Array.from(new Uint32Array(histogramMapped.slice(0)))
    histogramReadback.unmap()

    const cpuMs = performance.now() - cpuStart
    let gpuMs = cpuMs
    let timingSource: TimingSource = 'cpu-clock'

    if (queryReadback) {
      const raw = new BigUint64Array(queryReadback.getMappedRange().slice(0))
      queryReadback.unmap()
      if (raw.length >= 2 && raw[1] >= raw[0]) {
        gpuMs = Number(raw[1] - raw[0]) / 1e6
        timingSource = 'gpu-timestamp'
      }
    }

    return {
      histogram,
      gpuDurationMs: Number(gpuMs.toFixed(3)),
      backendDurationMs: Number(cpuMs.toFixed(3)),
      timingSource,
    }
  } finally {
    inputBuffer.destroy()
    histogramBuffer.destroy()
    histogramReadback.destroy()
    if (queryResolve) queryResolve.destroy()
    if (queryReadback) queryReadback.destroy()
    if (querySet) querySet.destroy()
  }
}

