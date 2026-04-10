import { readFileSync } from 'node:fs'
import { getGpuDevice } from './device.js'

// Zmieniamy na 16, musi pasować do wgsl!
const WORKGROUP_SIZE = 16
const shaderSource = readFileSync(new URL('./shaders/matrixMul.wgsl', import.meta.url), 'utf8')

export interface MatrixMulWebGpuOptions {
    readback?: boolean
}

// Używamy WeakMap, aby uniknąć wycieków pamięci. Jeśli urządzenie zostanie zniszczone,
// pipeline również automatycznie wyparuje z pamięci RAM.
const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>()

function getPipeline(device: GPUDevice): GPUComputePipeline {
    if (pipelineCache.has(device)) {
        return pipelineCache.get(device)!
    }

    const module = device.createShaderModule({
        label: 'matrix-mul-naive-shader',
        code: shaderSource,
    })

    const pipeline = device.createComputePipeline({
        label: 'matrix-mul-naive-pipeline',
        layout: 'auto',
        compute: {
            module,
            entryPoint: 'main',
        },
    })

    pipelineCache.set(device, pipeline)
    return pipeline
}

export async function multiplySquareMatricesWebGpu(
    size: number,
    matrixA: Float32Array,
    matrixB: Float32Array,
    options: MatrixMulWebGpuOptions = {},
): Promise<{ output: Float32Array | null }> {

    const expectedLength = size * size
    if (matrixA.length !== expectedLength || matrixB.length !== expectedLength) {
        throw new Error(`Invalid matrix length. Expected ${expectedLength}`)
    }

    const shouldReadback = options.readback ?? true
    const device = await getGpuDevice()
    const pipeline = getPipeline(device)

    const outputByteLength = expectedLength * Float32Array.BYTES_PER_ELEMENT

    // Weryfikacja limitów urządzenia
    if (outputByteLength > device.limits.maxStorageBufferBindingSize) {
        throw new Error(`Matrix buffer size exceeds device limits.`)
    }

    // Wymiary wyrównane do 16 bajtów (4 x u32) - zgodność z uniform buffer
    const dims = new Uint32Array([size, 0, 0, 0])

    // Alokacja buforów
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
        // 1. Transfer danych na GPU (Host -> Device)
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

        // Dispatch grid bazujący na nowym workgroup size (16)
        const gridDim = Math.ceil(size / WORKGROUP_SIZE)
        pass.dispatchWorkgroups(gridDim, gridDim, 1)
        pass.end()

        if (readbackBuffer) {
            encoder.copyBufferToBuffer(matrixCBuffer, 0, readbackBuffer, 0, outputByteLength)
        }

        // 2. Wykonanie i synchronizacja
        device.queue.submit([encoder.finish()])

        if (!readbackBuffer) {
            await device.queue.onSubmittedWorkDone()
            return { output: null }
        }

        // 3. Transfer danych z powrotem (Device -> Host)
        await readbackBuffer.mapAsync(GPUMapMode.READ)
        const mapped = readbackBuffer.getMappedRange()
        const output = new Float32Array(mapped.slice(0))
        readbackBuffer.unmap()

        return { output }

    } finally {
        // Rygorystyczne czyszczenie pamięci
        dimsBuffer.destroy()
        matrixABuffer.destroy()
        matrixBBuffer.destroy()
        matrixCBuffer.destroy()
        if (readbackBuffer) readbackBuffer.destroy()
    }
}