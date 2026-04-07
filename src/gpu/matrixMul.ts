import { readFileSync } from 'node:fs'

import { getGpuDevice } from './device.js'

const WORKGROUP_SIZE = 8
const shaderSource = readFileSync(new URL('./shaders/matrixMul.wgsl', import.meta.url), 'utf8')

export interface MatrixMulWebGpuOptions {
  readback?: boolean
}

let cachedPipeline: { device: GPUDevice; pipeline: GPUComputePipeline } | null = null

function getPipeline(device: GPUDevice): GPUComputePipeline {
  if (cachedPipeline?.device === device) return cachedPipeline.pipeline

  const module = device.createShaderModule({
    label: 'matrix-mul-shader',
    code: shaderSource,
  })

  const pipeline = device.createComputePipeline({
    label: 'matrix-mul-pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  })

  cachedPipeline = { device, pipeline }
  return pipeline
}


export async function multiplySquareMatricesWebGpu(
  size: number,
  matrixA: Float32Array,
  matrixB: Float32Array,
  options: MatrixMulWebGpuOptions = {},
): Promise<Float32Array | null> {
  const expectedLength = size * size
  if (matrixA.length !== expectedLength || matrixB.length !== expectedLength) {
    throw new Error(`Invalid matrix length. Expected ${expectedLength}, got A=${matrixA.length}, B=${matrixB.length}`)
  }

  const shouldReadback = options.readback ?? true
  const device = await getGpuDevice()
  const pipeline = getPipeline(device)

  const outputByteLength = expectedLength * Float32Array.BYTES_PER_ELEMENT
  if (outputByteLength > device.limits.maxBufferSize) {
    throw new Error(
      `Matrix buffer size (${outputByteLength}) exceeds device maxBufferSize (${device.limits.maxBufferSize}).`,
    )
  }
  if (outputByteLength > device.limits.maxStorageBufferBindingSize) {
    throw new Error(
      `Matrix buffer size (${outputByteLength}) exceeds device maxStorageBufferBindingSize (${device.limits.maxStorageBufferBindingSize}).`,
    )
  }

  const dims = new Uint32Array([size, size, size, 0])

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
    const pass = encoder.beginComputePass({ label: 'matrix-mul-pass' })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(size / WORKGROUP_SIZE), Math.ceil(size / WORKGROUP_SIZE), 1)
    pass.end()

    if (readbackBuffer) {
      encoder.copyBufferToBuffer(matrixCBuffer, 0, readbackBuffer, 0, outputByteLength)
    }

    device.queue.submit([encoder.finish()])

    if (!readbackBuffer) {
      await device.queue.onSubmittedWorkDone()
      return null
    }

    await readbackBuffer.mapAsync(GPUMapMode.READ)
    const mapped = readbackBuffer.getMappedRange()
    const output = new Float32Array(mapped.slice(0))
    readbackBuffer.unmap()

    return output
  } finally {
    dimsBuffer.destroy()
    matrixABuffer.destroy()
    matrixBBuffer.destroy()
    matrixCBuffer.destroy()
    if (readbackBuffer) readbackBuffer.destroy()
  }
}


