import { readFileSync } from 'node:fs'
import { getGpuDevice } from './device.js'

const CNN_INPUT_CHANNELS = 3
const CNN_INPUT_HEIGHT = 128
const CNN_INPUT_WIDTH = 128
const CNN_CONV1_OUT_CHANNELS = 16
const CNN_CONV2_OUT_CHANNELS = 32
const CNN_DENSE1_OUT = 128
const CNN_OUTPUT_SIZE = 10

const CNN_POOL1_HEIGHT = 64
const CNN_POOL1_WIDTH = 64
const CNN_POOL2_HEIGHT = 32
const CNN_POOL2_WIDTH = 32
const CNN_FLATTEN_SIZE = CNN_CONV2_OUT_CHANNELS * CNN_POOL2_HEIGHT * CNN_POOL2_WIDTH

const CONV1_W_COUNT = CNN_CONV1_OUT_CHANNELS * CNN_INPUT_CHANNELS * 3 * 3
const CONV1_B_COUNT = CNN_CONV1_OUT_CHANNELS
const CONV2_W_COUNT = CNN_CONV2_OUT_CHANNELS * CNN_CONV1_OUT_CHANNELS * 3 * 3
const CONV2_B_COUNT = CNN_CONV2_OUT_CHANNELS
const DENSE1_W_COUNT = CNN_FLATTEN_SIZE * CNN_DENSE1_OUT
const DENSE1_B_COUNT = CNN_DENSE1_OUT
const DENSE2_W_COUNT = CNN_DENSE1_OUT * CNN_OUTPUT_SIZE
const DENSE2_B_COUNT = CNN_OUTPUT_SIZE

const TOTAL_WEIGHT_COUNT =
  CONV1_W_COUNT +
  CONV1_B_COUNT +
  CONV2_W_COUNT +
  CONV2_B_COUNT +
  DENSE1_W_COUNT +
  DENSE1_B_COUNT +
  DENSE2_W_COUNT +
  DENSE2_B_COUNT

const PARAM_BYTES = 32
const TIMESTAMP_COUNT = 12
const TOTAL_TIMESTAMP_BYTES = TIMESTAMP_COUNT * 8

type TimingSource = 'gpu-timestamp' | 'cpu-clock'

export interface CnnMemoryEstimate {
  gpuAllocatedBytes: number
  gpuAllocatedMiB: number
  hostAllocatedBytes: number
  hostAllocatedMiB: number
}

export interface LoadedWebGpuCnnModel {
  device: GPUDevice
  conv1W: GPUBuffer
  conv1B: GPUBuffer
  conv2W: GPUBuffer
  conv2B: GPUBuffer
  dense1W: GPUBuffer
  dense1B: GPUBuffer
  dense2W: GPUBuffer
  dense2B: GPUBuffer
  input: GPUBuffer
  conv1Out: GPUBuffer
  pool1Out: GPUBuffer
  conv2Out: GPUBuffer
  pool2Out: GPUBuffer
  dense1Out: GPUBuffer
  output: GPUBuffer
  memoryEstimate: CnnMemoryEstimate
}

export interface WebGpuCnnPredictResult {
  logits: Float32Array
  gpuDurationMs: number
  totalDurationMs: number
  timingSource: TimingSource
  memoryEstimate: CnnMemoryEstimate
}

interface PipelineSet {
  conv: GPUComputePipeline
  pool: GPUComputePipeline
  dense: GPUComputePipeline
}

const shaderSource = readFileSync(new URL('./shaders/cnn.wgsl', import.meta.url), 'utf8')
const pipelineCache = new WeakMap<GPUDevice, PipelineSet>()

function toMiB(bytes: number): number {
  return Math.round((bytes / (1024 * 1024)) * 1000) / 1000
}

function readDurationMsFromRaw(start: bigint, end: bigint): number | null {
  const deltaTicks = end >= start ? end - start : (1n << 64n) - start + end
  return deltaTicks > 0n ? Number(deltaTicks) / 1e6 : null
}

function createStorageBuffer(device: GPUDevice, label: string, size: number, includeCopySrc = false): GPUBuffer {
  return device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (includeCopySrc ? GPUBufferUsage.COPY_SRC : 0),
  })
}

function getPipelines(device: GPUDevice): PipelineSet {
  const cached = pipelineCache.get(device)
  if (cached) return cached

  const module = device.createShaderModule({
    label: 'cnn-shader',
    code: shaderSource,
  })

  const next: PipelineSet = {
    conv: device.createComputePipeline({
      label: 'cnn-conv-pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'conv2dRelu' },
    }),
    pool: device.createComputePipeline({
      label: 'cnn-pool-pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'maxPool2x2' },
    }),
    dense: device.createComputePipeline({
      label: 'cnn-dense-pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'denseGemv' },
    }),
  }

  pipelineCache.set(device, next)
  return next
}

function makeParams(...values: number[]): Uint32Array {
  const params = new Uint32Array(8)
  for (let i = 0; i < values.length && i < 8; i++) {
    params[i] = values[i] >>> 0
  }
  return params
}

function createParamsBuffer(device: GPUDevice, label: string, params: Uint32Array): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: PARAM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buffer, 0, params)
  return buffer
}

export async function loadCnnModelToWebGpu(weights: Float32Array): Promise<LoadedWebGpuCnnModel> {
  if (weights.length !== TOTAL_WEIGHT_COUNT) {
    throw new Error(`Invalid CNN weights length. Expected ${TOTAL_WEIGHT_COUNT}, got ${weights.length}.`)
  }

  const device = await getGpuDevice()
  void getPipelines(device)

  const conv1WBytes = CONV1_W_COUNT * Float32Array.BYTES_PER_ELEMENT
  const conv1BBytes = CONV1_B_COUNT * Float32Array.BYTES_PER_ELEMENT
  const conv2WBytes = CONV2_W_COUNT * Float32Array.BYTES_PER_ELEMENT
  const conv2BBytes = CONV2_B_COUNT * Float32Array.BYTES_PER_ELEMENT
  const dense1WBytes = DENSE1_W_COUNT * Float32Array.BYTES_PER_ELEMENT
  const dense1BBytes = DENSE1_B_COUNT * Float32Array.BYTES_PER_ELEMENT
  const dense2WBytes = DENSE2_W_COUNT * Float32Array.BYTES_PER_ELEMENT
  const dense2BBytes = DENSE2_B_COUNT * Float32Array.BYTES_PER_ELEMENT

  const inputBytes = CNN_INPUT_CHANNELS * CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH * Float32Array.BYTES_PER_ELEMENT
  const conv1OutBytes = CNN_CONV1_OUT_CHANNELS * CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH * Float32Array.BYTES_PER_ELEMENT
  const pool1OutBytes = CNN_CONV1_OUT_CHANNELS * CNN_POOL1_HEIGHT * CNN_POOL1_WIDTH * Float32Array.BYTES_PER_ELEMENT
  const conv2OutBytes = CNN_CONV2_OUT_CHANNELS * CNN_POOL1_HEIGHT * CNN_POOL1_WIDTH * Float32Array.BYTES_PER_ELEMENT
  const pool2OutBytes = CNN_CONV2_OUT_CHANNELS * CNN_POOL2_HEIGHT * CNN_POOL2_WIDTH * Float32Array.BYTES_PER_ELEMENT
  const dense1OutBytes = CNN_DENSE1_OUT * Float32Array.BYTES_PER_ELEMENT
  const outputBytes = CNN_OUTPUT_SIZE * Float32Array.BYTES_PER_ELEMENT

  const conv1W = createStorageBuffer(device, 'cnn-conv1-w', conv1WBytes)
  const conv1B = createStorageBuffer(device, 'cnn-conv1-b', conv1BBytes)
  const conv2W = createStorageBuffer(device, 'cnn-conv2-w', conv2WBytes)
  const conv2B = createStorageBuffer(device, 'cnn-conv2-b', conv2BBytes)
  const dense1W = createStorageBuffer(device, 'cnn-dense1-w', dense1WBytes)
  const dense1B = createStorageBuffer(device, 'cnn-dense1-b', dense1BBytes)
  const dense2W = createStorageBuffer(device, 'cnn-dense2-w', dense2WBytes)
  const dense2B = createStorageBuffer(device, 'cnn-dense2-b', dense2BBytes)

  const input = createStorageBuffer(device, 'cnn-input', inputBytes)
  const conv1Out = createStorageBuffer(device, 'cnn-conv1-out', conv1OutBytes)
  const pool1Out = createStorageBuffer(device, 'cnn-pool1-out', pool1OutBytes)
  const conv2Out = createStorageBuffer(device, 'cnn-conv2-out', conv2OutBytes)
  const pool2Out = createStorageBuffer(device, 'cnn-pool2-out', pool2OutBytes)
  const dense1Out = createStorageBuffer(device, 'cnn-dense1-out', dense1OutBytes)
  const output = createStorageBuffer(device, 'cnn-output', outputBytes, true)

  try {
    let offset = 0
    device.queue.writeBuffer(conv1W, 0, weights, offset, CONV1_W_COUNT)
    offset += CONV1_W_COUNT
    device.queue.writeBuffer(conv1B, 0, weights, offset, CONV1_B_COUNT)
    offset += CONV1_B_COUNT

    device.queue.writeBuffer(conv2W, 0, weights, offset, CONV2_W_COUNT)
    offset += CONV2_W_COUNT
    device.queue.writeBuffer(conv2B, 0, weights, offset, CONV2_B_COUNT)
    offset += CONV2_B_COUNT

    device.queue.writeBuffer(dense1W, 0, weights, offset, DENSE1_W_COUNT)
    offset += DENSE1_W_COUNT
    device.queue.writeBuffer(dense1B, 0, weights, offset, DENSE1_B_COUNT)
    offset += DENSE1_B_COUNT

    device.queue.writeBuffer(dense2W, 0, weights, offset, DENSE2_W_COUNT)
    offset += DENSE2_W_COUNT
    device.queue.writeBuffer(dense2B, 0, weights, offset, DENSE2_B_COUNT)

    await device.queue.onSubmittedWorkDone()

    const gpuAllocatedBytes =
      conv1WBytes +
      conv1BBytes +
      conv2WBytes +
      conv2BBytes +
      dense1WBytes +
      dense1BBytes +
      dense2WBytes +
      dense2BBytes +
      inputBytes +
      conv1OutBytes +
      pool1OutBytes +
      conv2OutBytes +
      pool2OutBytes +
      dense1OutBytes +
      outputBytes

    return {
      device,
      conv1W,
      conv1B,
      conv2W,
      conv2B,
      dense1W,
      dense1B,
      dense2W,
      dense2B,
      input,
      conv1Out,
      pool1Out,
      conv2Out,
      pool2Out,
      dense1Out,
      output,
      memoryEstimate: {
        gpuAllocatedBytes,
        gpuAllocatedMiB: toMiB(gpuAllocatedBytes),
        hostAllocatedBytes: weights.byteLength,
        hostAllocatedMiB: toMiB(weights.byteLength),
      },
    }
  } catch (error) {
    conv1W.destroy(); conv1B.destroy(); conv2W.destroy(); conv2B.destroy()
    dense1W.destroy(); dense1B.destroy(); dense2W.destroy(); dense2B.destroy()
    input.destroy(); conv1Out.destroy(); pool1Out.destroy(); conv2Out.destroy()
    pool2Out.destroy(); dense1Out.destroy(); output.destroy()
    throw error
  }
}

export async function predictWithWebGpuCnn(
  model: LoadedWebGpuCnnModel,
  input: Float32Array,
): Promise<WebGpuCnnPredictResult> {
  const expectedInput = CNN_INPUT_CHANNELS * CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH
  if (input.length !== expectedInput) {
    throw new Error(`Invalid input length. Expected ${expectedInput}.`)
  }

  const pipelines = getPipelines(model.device)
  const device = model.device

  const conv1Params = createParamsBuffer(
    device,
    'cnn-conv1-params',
    makeParams(CNN_INPUT_CHANNELS, CNN_CONV1_OUT_CHANNELS, CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH, 1),
  )
  const pool1Params = createParamsBuffer(device, 'cnn-pool1-params', makeParams(CNN_CONV1_OUT_CHANNELS, CNN_INPUT_HEIGHT, CNN_INPUT_WIDTH))
  const conv2Params = createParamsBuffer(
    device,
    'cnn-conv2-params',
    makeParams(CNN_CONV1_OUT_CHANNELS, CNN_CONV2_OUT_CHANNELS, CNN_POOL1_HEIGHT, CNN_POOL1_WIDTH, 1),
  )
  const pool2Params = createParamsBuffer(device, 'cnn-pool2-params', makeParams(CNN_CONV2_OUT_CHANNELS, CNN_POOL1_HEIGHT, CNN_POOL1_WIDTH))
  const dense1Params = createParamsBuffer(device, 'cnn-dense1-params', makeParams(CNN_FLATTEN_SIZE, CNN_DENSE1_OUT, 1))
  const dense2Params = createParamsBuffer(device, 'cnn-dense2-params', makeParams(CNN_DENSE1_OUT, CNN_OUTPUT_SIZE, 0))

  const outputBytes = CNN_OUTPUT_SIZE * Float32Array.BYTES_PER_ELEMENT
  const readback = device.createBuffer({
    label: 'cnn-logits-readback',
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })

  const supportsTimestampQuery = device.features.has('timestamp-query' as GPUFeatureName)
  const querySet = supportsTimestampQuery
    ? device.createQuerySet({
        label: 'cnn-query-set',
        type: 'timestamp',
        count: TIMESTAMP_COUNT,
      })
    : null
  const queryResolve = querySet
    ? device.createBuffer({
        label: 'cnn-query-resolve',
        size: TOTAL_TIMESTAMP_BYTES,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : null
  const queryReadback = querySet
    ? device.createBuffer({
        label: 'cnn-query-readback',
        size: TOTAL_TIMESTAMP_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    : null

  try {
    device.queue.writeBuffer(model.input, 0, input)

    const conv1Bind = device.createBindGroup({
      layout: pipelines.conv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: conv1Params } },
        { binding: 1, resource: { buffer: model.input } },
        { binding: 2, resource: { buffer: model.conv1W } },
        { binding: 3, resource: { buffer: model.conv1B } },
        { binding: 4, resource: { buffer: model.conv1Out } },
      ],
    })

    const pool1Bind = device.createBindGroup({
      layout: pipelines.pool.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pool1Params } },
        { binding: 1, resource: { buffer: model.conv1Out } },
        { binding: 2, resource: { buffer: model.conv1W } },
        { binding: 3, resource: { buffer: model.conv1B } },
        { binding: 4, resource: { buffer: model.pool1Out } },
      ],
    })

    const conv2Bind = device.createBindGroup({
      layout: pipelines.conv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: conv2Params } },
        { binding: 1, resource: { buffer: model.pool1Out } },
        { binding: 2, resource: { buffer: model.conv2W } },
        { binding: 3, resource: { buffer: model.conv2B } },
        { binding: 4, resource: { buffer: model.conv2Out } },
      ],
    })

    const pool2Bind = device.createBindGroup({
      layout: pipelines.pool.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pool2Params } },
        { binding: 1, resource: { buffer: model.conv2Out } },
        { binding: 2, resource: { buffer: model.conv2W } },
        { binding: 3, resource: { buffer: model.conv2B } },
        { binding: 4, resource: { buffer: model.pool2Out } },
      ],
    })

    const dense1Bind = device.createBindGroup({
      layout: pipelines.dense.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dense1Params } },
        { binding: 1, resource: { buffer: model.pool2Out } },
        { binding: 2, resource: { buffer: model.dense1W } },
        { binding: 3, resource: { buffer: model.dense1B } },
        { binding: 4, resource: { buffer: model.dense1Out } },
      ],
    })

    const dense2Bind = device.createBindGroup({
      layout: pipelines.dense.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dense2Params } },
        { binding: 1, resource: { buffer: model.dense1Out } },
        { binding: 2, resource: { buffer: model.dense2W } },
        { binding: 3, resource: { buffer: model.dense2B } },
        { binding: 4, resource: { buffer: model.output } },
      ],
    })

    const encoder = device.createCommandEncoder({ label: 'cnn-inference-encoder' })

    const pass0 = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {})
    pass0.setPipeline(pipelines.conv)
    pass0.setBindGroup(0, conv1Bind)
    pass0.dispatchWorkgroups(Math.ceil(CNN_INPUT_WIDTH / 8), Math.ceil(CNN_INPUT_HEIGHT / 8), CNN_CONV1_OUT_CHANNELS)
    pass0.end()

    const pass1 = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 } } : {})
    pass1.setPipeline(pipelines.pool)
    pass1.setBindGroup(0, pool1Bind)
    pass1.dispatchWorkgroups(Math.ceil(CNN_POOL1_WIDTH / 8), Math.ceil(CNN_POOL1_HEIGHT / 8), CNN_CONV1_OUT_CHANNELS)
    pass1.end()

    const pass2 = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 } } : {})
    pass2.setPipeline(pipelines.conv)
    pass2.setBindGroup(0, conv2Bind)
    pass2.dispatchWorkgroups(Math.ceil(CNN_POOL1_WIDTH / 8), Math.ceil(CNN_POOL1_HEIGHT / 8), CNN_CONV2_OUT_CHANNELS)
    pass2.end()

    const pass3 = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 6, endOfPassWriteIndex: 7 } } : {})
    pass3.setPipeline(pipelines.pool)
    pass3.setBindGroup(0, pool2Bind)
    pass3.dispatchWorkgroups(Math.ceil(CNN_POOL2_WIDTH / 8), Math.ceil(CNN_POOL2_HEIGHT / 8), CNN_CONV2_OUT_CHANNELS)
    pass3.end()

    const pass4 = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 8, endOfPassWriteIndex: 9 } } : {})
    pass4.setPipeline(pipelines.dense)
    pass4.setBindGroup(0, dense1Bind)
    pass4.dispatchWorkgroups(CNN_DENSE1_OUT)
    pass4.end()

    const pass5 = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 10, endOfPassWriteIndex: 11 } } : {})
    pass5.setPipeline(pipelines.dense)
    pass5.setBindGroup(0, dense2Bind)
    pass5.dispatchWorkgroups(CNN_OUTPUT_SIZE)
    pass5.end()

    encoder.copyBufferToBuffer(model.output, 0, readback, 0, outputBytes)

    if (querySet && queryResolve && queryReadback) {
      encoder.resolveQuerySet(querySet, 0, TIMESTAMP_COUNT, queryResolve, 0)
      encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, TOTAL_TIMESTAMP_BYTES)
    }

    const cpuStart = performance.now()
    device.queue.submit([encoder.finish()])

    const waitTasks: Promise<void>[] = [
      device.queue.onSubmittedWorkDone(),
      readback.mapAsync(GPUMapMode.READ),
    ]
    if (queryReadback) {
      waitTasks.push(queryReadback.mapAsync(GPUMapMode.READ))
    }
    await Promise.all(waitTasks)

    const logitsData = readback.getMappedRange()
    const logits = new Float32Array(CNN_OUTPUT_SIZE)
    logits.set(new Float32Array(logitsData))
    readback.unmap()

    const cpuDurationMs = performance.now() - cpuStart
    let gpuDurationMs = cpuDurationMs
    let timingSource: TimingSource = 'cpu-clock'

    if (queryReadback) {
      const mapped = queryReadback.getMappedRange()
      const timestamps = new BigUint64Array(mapped.slice(0))
      queryReadback.unmap()

      if (timestamps.length >= TIMESTAMP_COUNT) {
        let sum = 0
        let ok = true
        for (let i = 0; i < TIMESTAMP_COUNT; i += 2) {
          const part = readDurationMsFromRaw(timestamps[i], timestamps[i + 1])
          if (part === null) {
            ok = false
            break
          }
          sum += part
        }
        if (ok) {
          gpuDurationMs = sum
          timingSource = 'gpu-timestamp'
        }
      }
    }

    return {
      logits,
      gpuDurationMs,
      totalDurationMs: cpuDurationMs,
      timingSource,
      memoryEstimate: model.memoryEstimate,
    }
  } finally {
    conv1Params.destroy()
    pool1Params.destroy()
    conv2Params.destroy()
    pool2Params.destroy()
    dense1Params.destroy()
    dense2Params.destroy()
    readback.destroy()
    if (queryResolve) queryResolve.destroy()
    if (queryReadback) queryReadback.destroy()
    if (querySet) querySet.destroy()
  }
}

export function unloadWebGpuCnnModel(model: LoadedWebGpuCnnModel): void {
  model.conv1W.destroy(); model.conv1B.destroy(); model.conv2W.destroy(); model.conv2B.destroy()
  model.dense1W.destroy(); model.dense1B.destroy(); model.dense2W.destroy(); model.dense2B.destroy()
  model.input.destroy(); model.conv1Out.destroy(); model.pool1Out.destroy(); model.conv2Out.destroy()
  model.pool2Out.destroy(); model.dense1Out.destroy(); model.output.destroy()
}

export const cnnLayout = {
  inputChannels: CNN_INPUT_CHANNELS,
  inputHeight: CNN_INPUT_HEIGHT,
  inputWidth: CNN_INPUT_WIDTH,
  flattenSize: CNN_FLATTEN_SIZE,
  outputSize: CNN_OUTPUT_SIZE,
  totalWeightCount: TOTAL_WEIGHT_COUNT,
} as const


