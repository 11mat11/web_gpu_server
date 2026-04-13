import { readFileSync } from 'node:fs'
import { getGpuDevice } from './device.js'

const WORKGROUP_SIZE = 16
const RANDOM_FILL_WORKGROUP_SIZE = 256
const OPTIMIZED_TILE_SIZE = 32

const matrixMulShaderSource = readFileSync(new URL('./shaders/matrixMul.wgsl', import.meta.url), 'utf8')
const matrixMulTiledShaderSource = readFileSync(new URL('./shaders/matrixMulTiled.wgsl', import.meta.url), 'utf8')
const randomFillShaderSource = readFileSync(new URL('./shaders/randomFill.wgsl', import.meta.url), 'utf8')

type TimingSource = 'gpu-timestamp' | 'cpu-clock'

export interface MatrixMulWebGpuOptions {
  readback?: boolean
  optimized?: boolean
}

export interface RandomGpuMatrixMulOptions {
  readback?: boolean
  seed?: number
  optimized?: boolean
}

export interface MatrixGpuTimings {
  generationDurationMs: number | null
  multiplyDurationMs: number | null
  totalDurationMs: number | null
  timingSource: TimingSource
}

export interface MatrixMemoryEstimate {
  gpuAllocatedBytes: number
  gpuAllocatedMiB: number
  hostAllocatedBytes: number
  hostAllocatedMiB: number
}

export interface MatrixMulWebGpuResult extends MatrixGpuTimings {
  output: Float32Array | null
  memoryEstimate: MatrixMemoryEstimate
}

const matrixMulPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>()
const matrixMulTiledPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>()
const randomFillPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>()

function getMatrixMulPipeline(device: GPUDevice): GPUComputePipeline {
  const cached = matrixMulPipelineCache.get(device)
  if (cached) return cached

  const module = device.createShaderModule({
    label: 'matrix-mul-naive-shader',
    code: matrixMulShaderSource,
  })

  const pipeline = device.createComputePipeline({
    label: 'matrix-mul-naive-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  })

  matrixMulPipelineCache.set(device, pipeline)
  return pipeline
}

function getRandomFillPipeline(device: GPUDevice): GPUComputePipeline {
  const cached = randomFillPipelineCache.get(device)
  if (cached) return cached

  const module = device.createShaderModule({
    label: 'matrix-random-fill-shader',
    code: randomFillShaderSource,
  })

  const pipeline = device.createComputePipeline({
    label: 'matrix-random-fill-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  })

  randomFillPipelineCache.set(device, pipeline)
  return pipeline
}

function getMatrixMulTiledPipeline(device: GPUDevice): GPUComputePipeline {
  const cached = matrixMulTiledPipelineCache.get(device)
  if (cached) return cached

  const module = device.createShaderModule({
    label: 'matrix-mul-tiled-shader',
    code: matrixMulTiledShaderSource,
  })

  const pipeline = device.createComputePipeline({
    label: 'matrix-mul-tiled-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  })

  matrixMulTiledPipelineCache.set(device, pipeline)
  return pipeline
}

function readDurationMsFromTimestampBuffer(buffer: GPUBuffer): number | null {
  const mapped = buffer.getMappedRange()
  const timestamps = new BigUint64Array(mapped.slice(0))
  buffer.unmap()

  const start = timestamps[0]
  const end = timestamps[1]

  // Timestampy są 64-bit unsigned; signed odczyt może dać fałszywie ujemną deltę.
  const deltaTicks = end >= start ? end - start : (1n << 64n) - start + end
  return deltaTicks > 0n ? Number(deltaTicks) / 1e6 : null
}

function readDurationMsFromRawTimestamps(start: bigint, end: bigint): number | null {
  const deltaTicks = end >= start ? end - start : (1n << 64n) - start + end
  return deltaTicks > 0n ? Number(deltaTicks) / 1e6 : null
}

function getSeed(seed?: number): number {
  if (typeof seed === 'number' && Number.isFinite(seed)) {
    return seed >>> 0
  }
  const coarse = Date.now() >>> 0
  const fine = Number(process.hrtime.bigint() & 0xffffffffn) >>> 0
  return (coarse ^ fine) >>> 0
}

function createDimsData(size: number): Uint32Array {
  return new Uint32Array([size, 0, 0, 0])
}

function createRandomParamsData(size: number, seed: number, minValue: number, maxValue: number): ArrayBuffer {
  const data = new ArrayBuffer(16)
  const view = new DataView(data)
  view.setUint32(0, size, true)
  view.setUint32(4, seed >>> 0, true)
  view.setFloat32(8, minValue, true)
  view.setFloat32(12, maxValue, true)
  return data
}

function toMiB(bytes: number): number {
  return bytes / (1024 * 1024)
}

function createMemoryEstimate(gpuAllocatedBytes: number, hostAllocatedBytes: number): MatrixMemoryEstimate {
  return {
    gpuAllocatedBytes,
    gpuAllocatedMiB: Number(toMiB(gpuAllocatedBytes).toFixed(3)),
    hostAllocatedBytes,
    hostAllocatedMiB: Number(toMiB(hostAllocatedBytes).toFixed(3)),
  }
}

function computeDispatchGrid(totalWorkgroups: number, maxPerDimension: number): [number, number, number] {
  const x = Math.min(totalWorkgroups, maxPerDimension)
  const remainingAfterX = Math.ceil(totalWorkgroups / x)

  if (remainingAfterX <= maxPerDimension) {
    return [x, remainingAfterX, 1]
  }

  const y = maxPerDimension
  const z = Math.ceil(remainingAfterX / y)
  if (z > maxPerDimension) {
    throw new Error(
      `Workgroup grid exceeds device limits. Required workgroups=${totalWorkgroups}, maxPerDimension=${maxPerDimension}.`,
    )
  }

  return [x, y, z]
}

export async function multiplySquareMatricesWebGpu(
  size: number,
  matrixA: Float32Array,
  matrixB: Float32Array,
  options: MatrixMulWebGpuOptions = {},
): Promise<MatrixMulWebGpuResult> {
  const expectedLength = size * size
  if (matrixA.length !== expectedLength || matrixB.length !== expectedLength) {
    throw new Error(`Invalid matrix length. Expected ${expectedLength}`)
  }

  const shouldReadback = options.readback ?? true
  const device = await getGpuDevice()
  const pipeline = options.optimized ? getMatrixMulTiledPipeline(device) : getMatrixMulPipeline(device)
  const outputByteLength = expectedLength * Float32Array.BYTES_PER_ELEMENT

  if (outputByteLength > device.limits.maxStorageBufferBindingSize) {
    throw new Error('Matrix buffer size exceeds device limits.')
  }

  const dims = createDimsData(size)
  const dimsBuffer = device.createBuffer({
    label: 'matrix-dims',
    size: dims.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const matrixABuffer = device.createBuffer({
    label: 'matrix-a',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const matrixBBuffer = device.createBuffer({
    label: 'matrix-b',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const matrixCBuffer = device.createBuffer({
    label: 'matrix-c',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const readbackBuffer = shouldReadback
    ? device.createBuffer({
        label: 'matrix-c-readback',
        size: outputByteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    : null

  const supportsTimestampQuery = device.features.has('timestamp-query' as GPUFeatureName)
  const querySet = supportsTimestampQuery
    ? device.createQuerySet({
        label: 'matrix-mul-timestamps',
        type: 'timestamp',
        count: 2,
      })
    : null
  const queryResolveBuffer = querySet
    ? device.createBuffer({
        label: 'matrix-mul-query-resolve',
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : null
  const queryReadbackBuffer = querySet
    ? device.createBuffer({
        label: 'matrix-mul-query-readback',
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
    : null

  const gpuAllocatedBytes =
    dims.byteLength +
    outputByteLength * 3 +
    (readbackBuffer ? outputByteLength : 0) +
    (queryResolveBuffer ? 16 : 0) +
    (queryReadbackBuffer ? 16 : 0)
  const hostAllocatedBytes = matrixA.byteLength + matrixB.byteLength + (readbackBuffer ? outputByteLength : 0)
  const memoryEstimate = createMemoryEstimate(gpuAllocatedBytes, hostAllocatedBytes)

  try {
    device.queue.writeBuffer(dimsBuffer, 0, dims)
    device.queue.writeBuffer(matrixABuffer, 0, matrixA)
    device.queue.writeBuffer(matrixBBuffer, 0, matrixB)

    const bindGroup = device.createBindGroup({
      label: 'matrix-mul-bind-group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dimsBuffer } },
        { binding: 1, resource: { buffer: matrixABuffer } },
        { binding: 2, resource: { buffer: matrixBBuffer } },
        { binding: 3, resource: { buffer: matrixCBuffer } },
      ],
    })

    const encoder = device.createCommandEncoder({ label: 'matrix-mul-encoder' })
    const pass = encoder.beginComputePass({
      label: 'matrix-mul-pass',
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

    const tileSize = options.optimized ? OPTIMIZED_TILE_SIZE : WORKGROUP_SIZE
    const gridDim = Math.ceil(size / tileSize)
    pass.dispatchWorkgroups(gridDim, gridDim, 1)
    pass.end()

    if (readbackBuffer) {
      encoder.copyBufferToBuffer(matrixCBuffer, 0, readbackBuffer, 0, outputByteLength)
    }

    if (querySet && queryResolveBuffer && queryReadbackBuffer) {
      encoder.resolveQuerySet(querySet, 0, 2, queryResolveBuffer, 0)
      encoder.copyBufferToBuffer(queryResolveBuffer, 0, queryReadbackBuffer, 0, 16)
    }

    const cpuStartMs = performance.now()
    device.queue.submit([encoder.finish()])

    const pendingMaps: Promise<void>[] = []
    if (readbackBuffer) pendingMaps.push(readbackBuffer.mapAsync(GPUMapMode.READ))
    if (queryReadbackBuffer) pendingMaps.push(queryReadbackBuffer.mapAsync(GPUMapMode.READ))

    if (pendingMaps.length > 0) {
      await Promise.all(pendingMaps)
    } else {
      await device.queue.onSubmittedWorkDone()
    }

    let output: Float32Array | null = null
    if (readbackBuffer) {
      const mapped = readbackBuffer.getMappedRange()
      output = new Float32Array(mapped.slice(0))
      readbackBuffer.unmap()
    }

    const cpuDurationMs = performance.now() - cpuStartMs
    let multiplyDurationMs: number | null = null
    let timingSource: TimingSource = 'cpu-clock'

    if (queryReadbackBuffer) {
      const timestampDurationMs = readDurationMsFromTimestampBuffer(queryReadbackBuffer)
      if (timestampDurationMs !== null) {
        multiplyDurationMs = timestampDurationMs
        timingSource = 'gpu-timestamp'
      }
    }

    if (multiplyDurationMs === null) {
      multiplyDurationMs = cpuDurationMs
      timingSource = 'cpu-clock'
    }

    return {
      output,
      generationDurationMs: null,
      multiplyDurationMs,
      totalDurationMs: multiplyDurationMs,
      timingSource,
      memoryEstimate,
    }
  } finally {
    dimsBuffer.destroy()
    matrixABuffer.destroy()
    matrixBBuffer.destroy()
    matrixCBuffer.destroy()
    if (readbackBuffer) readbackBuffer.destroy()
    if (queryResolveBuffer) queryResolveBuffer.destroy()
    if (queryReadbackBuffer) queryReadbackBuffer.destroy()
    if (querySet) querySet.destroy()
  }
}

export async function multiplyRandomSquareMatricesWebGpu(
  size: number,
  minValue: number,
  maxValue: number,
  options: RandomGpuMatrixMulOptions = {},
): Promise<MatrixMulWebGpuResult> {
  const expectedLength = size * size
  const outputByteLength = expectedLength * Float32Array.BYTES_PER_ELEMENT
  const shouldReadback = options.readback ?? true

  const device = await getGpuDevice()
  const fillPipeline = getRandomFillPipeline(device)
  const mulPipeline = options.optimized ? getMatrixMulTiledPipeline(device) : getMatrixMulPipeline(device)

  if (outputByteLength > device.limits.maxStorageBufferBindingSize) {
    throw new Error('Matrix buffer size exceeds device limits.')
  }

  const low = Math.min(minValue, maxValue)
  const high = Math.max(minValue, maxValue)
  const seed = getSeed(options.seed)
  const totalFillWorkgroups = Math.ceil(expectedLength / RANDOM_FILL_WORKGROUP_SIZE)
  const maxWorkgroupsPerDimension = device.limits.maxComputeWorkgroupsPerDimension
  const [fillDispatchX, fillDispatchY, fillDispatchZ] = computeDispatchGrid(totalFillWorkgroups, maxWorkgroupsPerDimension)

  const randomParams = createRandomParamsData(size, seed, low, high)
  const dims = createDimsData(size)

  const randomParamsBuffer = device.createBuffer({
    label: 'matrix-random-params',
    size: randomParams.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const dimsBuffer = device.createBuffer({
    label: 'matrix-dims',
    size: dims.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const matrixABuffer = device.createBuffer({
    label: 'matrix-a',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const matrixBBuffer = device.createBuffer({
    label: 'matrix-b',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE,
  })
  const matrixCBuffer = device.createBuffer({
    label: 'matrix-c',
    size: outputByteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })
  const readbackBuffer = shouldReadback
    ? device.createBuffer({
        label: 'matrix-c-readback',
        size: outputByteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    : null

  const supportsTimestampQuery = device.features.has('timestamp-query' as GPUFeatureName)

  const querySet = supportsTimestampQuery
    ? device.createQuerySet({
        label: 'matrix-full-pipeline-timestamps',
        type: 'timestamp',
        count: 4,
      })
    : null
  const queryResolveBuffer = querySet
    ? device.createBuffer({
        label: 'matrix-full-pipeline-query-resolve',
        size: 32,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : null
  const queryReadbackBuffer = querySet
    ? device.createBuffer({
        label: 'matrix-full-pipeline-query-readback',
        size: 32,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
    : null

  const gpuAllocatedBytes =
    randomParams.byteLength +
    dims.byteLength +
    outputByteLength * 3 +
    (readbackBuffer ? outputByteLength : 0) +
    (queryResolveBuffer ? 32 : 0) +
    (queryReadbackBuffer ? 32 : 0)
  const hostAllocatedBytes = randomParams.byteLength + dims.byteLength + (readbackBuffer ? outputByteLength : 0)
  const memoryEstimate = createMemoryEstimate(gpuAllocatedBytes, hostAllocatedBytes)

  try {
    device.queue.writeBuffer(randomParamsBuffer, 0, randomParams)
    device.queue.writeBuffer(dimsBuffer, 0, dims)

    const fillBindGroup = device.createBindGroup({
      label: 'matrix-random-fill-bind-group',
      layout: fillPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: randomParamsBuffer } },
        { binding: 1, resource: { buffer: matrixABuffer } },
        { binding: 2, resource: { buffer: matrixBBuffer } },
      ],
    })

    const mulBindGroup = device.createBindGroup({
      label: 'matrix-mul-bind-group',
      layout: mulPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dimsBuffer } },
        { binding: 1, resource: { buffer: matrixABuffer } },
        { binding: 2, resource: { buffer: matrixBBuffer } },
        { binding: 3, resource: { buffer: matrixCBuffer } },
      ],
    })

    const encoder = device.createCommandEncoder({ label: 'matrix-full-pipeline-encoder' })
    const fillPass = encoder.beginComputePass({
      label: 'matrix-random-fill-pass',
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
    fillPass.setPipeline(fillPipeline)
    fillPass.setBindGroup(0, fillBindGroup)
    fillPass.dispatchWorkgroups(fillDispatchX, fillDispatchY, fillDispatchZ)
    fillPass.end()

    const mulPass = encoder.beginComputePass({
      label: 'matrix-mul-pass',
      ...(querySet
        ? {
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: 2,
              endOfPassWriteIndex: 3,
            },
          }
        : {}),
    })
    mulPass.setPipeline(mulPipeline)
    mulPass.setBindGroup(0, mulBindGroup)
    const tileSize = options.optimized ? OPTIMIZED_TILE_SIZE : WORKGROUP_SIZE
    const gridDim = Math.ceil(size / tileSize)
    mulPass.dispatchWorkgroups(gridDim, gridDim, 1)
    mulPass.end()

    if (readbackBuffer) {
      encoder.copyBufferToBuffer(matrixCBuffer, 0, readbackBuffer, 0, outputByteLength)
    }

    if (querySet && queryResolveBuffer && queryReadbackBuffer) {
      encoder.resolveQuerySet(querySet, 0, 4, queryResolveBuffer, 0)
      encoder.copyBufferToBuffer(queryResolveBuffer, 0, queryReadbackBuffer, 0, 32)
    }

    const cpuPipelineStartMs = performance.now()
    device.queue.submit([encoder.finish()])

    const pendingMaps: Promise<void>[] = []
    pendingMaps.push(device.queue.onSubmittedWorkDone())
    if (readbackBuffer) pendingMaps.push(readbackBuffer.mapAsync(GPUMapMode.READ))
    if (queryReadbackBuffer) pendingMaps.push(queryReadbackBuffer.mapAsync(GPUMapMode.READ))
    await Promise.all(pendingMaps)

    let output: Float32Array | null = null
    if (readbackBuffer) {
      const mapped = readbackBuffer.getMappedRange()
      output = new Float32Array(mapped.slice(0))
      readbackBuffer.unmap()
    }

    const cpuPipelineDurationMs = performance.now() - cpuPipelineStartMs
    let generationTimestampMs: number | null = null
    let multiplyTimestampMs: number | null = null

    if (queryReadbackBuffer) {
      const mapped = queryReadbackBuffer.getMappedRange()
      const timestamps = new BigUint64Array(mapped.slice(0))
      queryReadbackBuffer.unmap()

      if (timestamps.length >= 4) {
        generationTimestampMs = readDurationMsFromRawTimestamps(timestamps[0], timestamps[1])
        multiplyTimestampMs = readDurationMsFromRawTimestamps(timestamps[2], timestamps[3])
      }
    }

    const hasValidGenerationGpuTimestamp = generationTimestampMs !== null && generationTimestampMs > 0
    const hasValidMultiplyGpuTimestamp = multiplyTimestampMs !== null && multiplyTimestampMs > 0

    const generationDurationMs = hasValidGenerationGpuTimestamp ? generationTimestampMs : null
    const multiplyDurationMs = hasValidMultiplyGpuTimestamp ? multiplyTimestampMs : cpuPipelineDurationMs
    const totalDurationMs =
      hasValidGenerationGpuTimestamp && hasValidMultiplyGpuTimestamp
        ? generationTimestampMs! + multiplyTimestampMs!
        : cpuPipelineDurationMs
    const timingSource: TimingSource = hasValidMultiplyGpuTimestamp ? 'gpu-timestamp' : 'cpu-clock'

    return {
      output,
      generationDurationMs,
      multiplyDurationMs,
      totalDurationMs,
      timingSource,
      memoryEstimate,
    }
  } finally {
    randomParamsBuffer.destroy()
    dimsBuffer.destroy()
    matrixABuffer.destroy()
    matrixBBuffer.destroy()
    matrixCBuffer.destroy()
    if (readbackBuffer) readbackBuffer.destroy()

    if (queryResolveBuffer) queryResolveBuffer.destroy()
    if (queryReadbackBuffer) queryReadbackBuffer.destroy()
    if (querySet) querySet.destroy()
  }
}


