import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  getCudaRuntimeState,
  loadCnnModelCuda,
  loadModelCuda,
  predictCnnCuda,
  predictMlpCuda,
  type CudaMlpMemoryEstimate,
  unloadCnnModelCuda,
  unloadModelCuda,
} from '../cuda/cudaBackend.js'
import {
  cnnLayout,
  loadCnnModelToWebGpu,
  predictWithWebGpuCnn,
  type CnnMemoryEstimate,
  type LoadedWebGpuCnnModel,
  unloadWebGpuCnnModel,
} from '../gpu/cnn-runner.js'
import {
  loadMlpModelToWebGpu,
  mlpLayout,
  predictWithWebGpuMlp,
  type LoadedWebGpuMlpModel,
  type MlpMemoryEstimate,
  unloadWebGpuMlpModel,
} from '../gpu/mlp-runner.js'

export type AiBackend = 'cuda' | 'webgpu'
export type AiModel = 'mlp' | 'cnn'

export interface AiLifecycleOptions {
  model?: AiModel
  webgpu?: boolean
  cuda?: boolean
}

export interface AiBackendStatus {
  loaded: boolean
  available: boolean
  reason?: string
}

export interface AiSingleModelMemoryEstimate {
  hostAllocatedBytes: number
  hostAllocatedMiB: number
  totalGpuAllocatedBytes: number
  totalGpuAllocatedMiB: number
  webgpu: MlpMemoryEstimate | CnnMemoryEstimate | null
  cuda: CudaMlpMemoryEstimate | null
}

export interface AiPipelineMemoryEstimate {
  hostAllocatedBytes: number
  hostAllocatedMiB: number
  totalGpuAllocatedBytes: number
  totalGpuAllocatedMiB: number
  models: {
    mlp: AiSingleModelMemoryEstimate
    cnn: AiSingleModelMemoryEstimate
  }
}

export interface AiModelStatus {
  loaded: boolean
  loadedBackends: AiBackend[]
  backends: {
    webgpu: AiBackendStatus
    cuda: AiBackendStatus
  }
  memoryEstimate: AiSingleModelMemoryEstimate
}

export interface AiLoadResult {
  status: 'loaded'
  loadedModels: AiModel[]
  models: {
    mlp: AiModelStatus
    cnn: AiModelStatus
  }
  memoryEstimate: AiPipelineMemoryEstimate
}

export interface AiPredictResult {
  prediction: number
  probabilities: number[]
  gpuDurationMs: number
  totalDurationMs: number
  timingSource: 'gpu-timestamp' | 'cpu-clock'
}

export interface AiCnnPredictResult extends AiPredictResult {
  predictionLabel: string
  memoryEstimate: AiSingleModelMemoryEstimate
}

export interface AiUnloadResult {
  status: 'unloaded'
  loadedModels: AiModel[]
  models: {
    mlp: AiModelStatus
    cnn: AiModelStatus
  }
  memoryEstimate: AiPipelineMemoryEstimate
}

export interface AiStatusResult {
  state: ManagerState
  loaded: boolean
  loadedModels: AiModel[]
  models: {
    mlp: AiModelStatus
    cnn: AiModelStatus
  }
  memoryEstimate: AiPipelineMemoryEstimate
}

export class AiManagerError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number,
  ) {
    super(message)
  }
}

type ManagerState = 'idle' | 'loading' | 'unloading'

const CIFAR10_LABELS = [
  'airplane',
  'automobile',
  'bird',
  'cat',
  'deer',
  'dog',
  'frog',
  'horse',
  'ship',
  'truck',
] as const

function toMiB(bytes: number): number {
  return Math.round((bytes / (1024 * 1024)) * 1000) / 1000
}

function softmax(logits: Float32Array): number[] {
  let maxLogit = Number.NEGATIVE_INFINITY
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i]
  }

  const exps = new Array<number>(logits.length)
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp(logits[i] - maxLogit)
    exps[i] = value
    sum += value
  }

  if (sum <= 0 || !Number.isFinite(sum)) {
    throw new AiManagerError('Softmax failed due to invalid logits.', 'softmax_failed', 500)
  }

  for (let i = 0; i < exps.length; i++) {
    exps[i] = exps[i] / sum
  }

  return exps
}

function argmax(values: number[]): number {
  let bestIndex = 0
  let bestValue = values[0] ?? Number.NEGATIVE_INFINITY
  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i]
      bestIndex = i
    }
  }
  return bestIndex
}

export class AiManager {
  private static instance: AiManager | null = null

  private state: ManagerState = 'idle'

  private mlpWebgpuModel: LoadedWebGpuMlpModel | null = null
  private mlpCudaReady = false
  private mlpWebgpuMemory: MlpMemoryEstimate | null = null
  private mlpCudaMemory: CudaMlpMemoryEstimate | null = null
  private mlpHostWeightsBytes = 0

  private cnnWebgpuModel: LoadedWebGpuCnnModel | null = null
  private cnnCudaReady = false
  private cnnWebgpuMemory: CnnMemoryEstimate | null = null
  private cnnCudaMemory: CudaMlpMemoryEstimate | null = null
  private cnnHostWeightsBytes = 0

  private cudaUnavailableReason: string | null = null

  private readonly mlpWeightsPath: string
  private readonly cnnWeightsPath: string

  private constructor() {
    const currentDir = fileURLToPath(new URL('.', import.meta.url))
    this.mlpWeightsPath = resolve(currentDir, './mega_mnist_weights.bin')
    this.cnnWeightsPath = resolve(currentDir, './cifar10_mini_vgg_weights.bin')
  }

  static getInstance(): AiManager {
    if (!AiManager.instance) {
      AiManager.instance = new AiManager()
    }
    return AiManager.instance
  }

  getState(): ManagerState {
    return this.state
  }

  isLoaded(): boolean {
    return this.getLoadedModels().length > 0
  }

  getStatus(): AiStatusResult {
    return {
      state: this.state,
      loaded: this.isLoaded(),
      loadedModels: this.getLoadedModels(),
      models: {
        mlp: this.buildModelStatus('mlp'),
        cnn: this.buildModelStatus('cnn'),
      },
      memoryEstimate: this.buildPipelineMemoryEstimate(),
    }
  }

  private resolveModels(model?: AiModel): AiModel[] {
    if (model) return [model]
    return ['mlp', 'cnn']
  }

  private resolveTargets(options?: AiLifecycleOptions): AiBackend[] {
    const hasAnyFlag = typeof options?.webgpu === 'boolean' || typeof options?.cuda === 'boolean'
    if (!hasAnyFlag) return ['webgpu', 'cuda']

    const targets: AiBackend[] = []
    if (options?.webgpu === true) targets.push('webgpu')
    if (options?.cuda === true) targets.push('cuda')
    return targets
  }

  private getLoadedModels(): AiModel[] {
    const loaded: AiModel[] = []
    if (this.mlpWebgpuModel || this.mlpCudaReady) loaded.push('mlp')
    if (this.cnnWebgpuModel || this.cnnCudaReady) loaded.push('cnn')
    return loaded
  }

  private getLoadedBackends(model: AiModel): AiBackend[] {
    const backends: AiBackend[] = []
    if (model === 'mlp') {
      if (this.mlpWebgpuModel) backends.push('webgpu')
      if (this.mlpCudaReady) backends.push('cuda')
      return backends
    }

    if (this.cnnWebgpuModel) backends.push('webgpu')
    if (this.cnnCudaReady) backends.push('cuda')
    return backends
  }

  private buildSingleModelMemoryEstimate(model: AiModel): AiSingleModelMemoryEstimate {
    const hostAllocatedBytes = model === 'mlp' ? this.mlpHostWeightsBytes : this.cnnHostWeightsBytes
    const webgpu = model === 'mlp' ? this.mlpWebgpuMemory : this.cnnWebgpuMemory
    const cuda = model === 'mlp' ? this.mlpCudaMemory : this.cnnCudaMemory
    const totalGpuAllocatedBytes = (webgpu?.gpuAllocatedBytes ?? 0) + (cuda?.gpuAllocatedBytes ?? 0)

    return {
      hostAllocatedBytes,
      hostAllocatedMiB: toMiB(hostAllocatedBytes),
      totalGpuAllocatedBytes,
      totalGpuAllocatedMiB: toMiB(totalGpuAllocatedBytes),
      webgpu,
      cuda,
    }
  }

  private buildPipelineMemoryEstimate(): AiPipelineMemoryEstimate {
    const mlp = this.buildSingleModelMemoryEstimate('mlp')
    const cnn = this.buildSingleModelMemoryEstimate('cnn')
    const hostAllocatedBytes = mlp.hostAllocatedBytes + cnn.hostAllocatedBytes
    const totalGpuAllocatedBytes = mlp.totalGpuAllocatedBytes + cnn.totalGpuAllocatedBytes

    return {
      hostAllocatedBytes,
      hostAllocatedMiB: toMiB(hostAllocatedBytes),
      totalGpuAllocatedBytes,
      totalGpuAllocatedMiB: toMiB(totalGpuAllocatedBytes),
      models: {
        mlp,
        cnn,
      },
    }
  }

  private buildBackendStatuses(model: AiModel): { webgpu: AiBackendStatus; cuda: AiBackendStatus } {
    const cudaRuntime = getCudaRuntimeState()
    const loadedBackends = this.getLoadedBackends(model)

    return {
      webgpu: {
        loaded: loadedBackends.includes('webgpu'),
        available: true,
      },
      cuda: {
        loaded: loadedBackends.includes('cuda'),
        available: loadedBackends.includes('cuda') || cudaRuntime.enabled,
        ...(loadedBackends.includes('cuda')
          ? {}
          : {
              reason: this.cudaUnavailableReason ?? (cudaRuntime.enabled ? undefined : cudaRuntime.reason),
            }),
      },
    }
  }

  private buildModelStatus(model: AiModel): AiModelStatus {
    const loadedBackends = this.getLoadedBackends(model)
    return {
      loaded: loadedBackends.length > 0,
      loadedBackends,
      backends: this.buildBackendStatuses(model),
      memoryEstimate: this.buildSingleModelMemoryEstimate(model),
    }
  }

  private async readModelWeights(model: AiModel): Promise<Float32Array> {
    const path = model === 'mlp' ? this.mlpWeightsPath : this.cnnWeightsPath
    const expectedFloats = model === 'mlp' ? mlpLayout.totalWeightCount : cnnLayout.totalWeightCount
    const fileBuffer = await readFile(path)
    const expectedBytes = expectedFloats * Float32Array.BYTES_PER_ELEMENT

    if (fileBuffer.byteLength !== expectedBytes) {
      throw new AiManagerError(
        `Invalid weight file size for ${model}. Expected ${expectedBytes} bytes, got ${fileBuffer.byteLength}.`,
        'invalid_weights_file',
        500,
      )
    }

    if (model === 'mlp') this.mlpHostWeightsBytes = fileBuffer.byteLength
    if (model === 'cnn') this.cnnHostWeightsBytes = fileBuffer.byteLength

    const source = new Float32Array(fileBuffer.buffer, fileBuffer.byteOffset, expectedFloats)
    return new Float32Array(source)
  }

  async loadModel(options?: AiLifecycleOptions): Promise<AiLoadResult> {
    if (this.state !== 'idle') {
      throw new AiManagerError('Model lifecycle operation is already in progress.', 'ai_busy', 409)
    }

    const models = this.resolveModels(options?.model)
    const targets = this.resolveTargets(options)
    if (targets.length === 0) {
      throw new AiManagerError(
        'At least one backend flag must be true (`webgpu` or `cuda`).',
        'invalid_load_targets',
        400,
      )
    }

    this.state = 'loading'
    const loadErrors: string[] = []

    try {
      for (const model of models) {
        const needsWebgpuLoad = targets.includes('webgpu') && !this.getLoadedBackends(model).includes('webgpu')
        const needsCudaLoad = targets.includes('cuda') && !this.getLoadedBackends(model).includes('cuda')
        if (!needsWebgpuLoad && !needsCudaLoad) continue

        const weights = await this.readModelWeights(model)

        if (needsWebgpuLoad) {
          try {
            if (model === 'mlp') {
              const loaded = await loadMlpModelToWebGpu(weights)
              this.mlpWebgpuModel = loaded
              this.mlpWebgpuMemory = loaded.memoryEstimate
            } else {
              const loaded = await loadCnnModelToWebGpu(weights)
              this.cnnWebgpuModel = loaded
              this.cnnWebgpuMemory = loaded.memoryEstimate
            }
          } catch (error) {
            loadErrors.push(`${model}: ${error instanceof Error ? error.message : 'WebGPU load failed.'}`)
          }
        }

        if (needsCudaLoad) {
          const runtimeState = getCudaRuntimeState()
          if (!runtimeState.enabled) {
            this.cudaUnavailableReason = runtimeState.reason
            loadErrors.push(`${model}: ${runtimeState.reason}`)
          } else {
            try {
              if (model === 'mlp') {
                const loaded = await loadModelCuda(weights)
                this.mlpCudaReady = true
                this.mlpCudaMemory = loaded.memoryEstimate
              } else {
                const loaded = await loadCnnModelCuda(weights)
                this.cnnCudaReady = true
                this.cnnCudaMemory = loaded.memoryEstimate
              }
              this.cudaUnavailableReason = null
            } catch (error) {
              if (model === 'mlp') {
                this.mlpCudaReady = false
                this.mlpCudaMemory = null
              } else {
                this.cnnCudaReady = false
                this.cnnCudaMemory = null
              }
              const message = error instanceof Error ? error.message : 'CUDA load failed.'
              this.cudaUnavailableReason = message
              loadErrors.push(`${model}: ${message}`)
            }
          }
        }
      }

      if (this.getLoadedModels().length === 0) {
        throw new AiManagerError(loadErrors[0] ?? 'No backend is loaded.', 'backend_unavailable', 400)
      }

      this.state = 'idle'
      return {
        status: 'loaded',
        loadedModels: this.getLoadedModels(),
        models: {
          mlp: this.buildModelStatus('mlp'),
          cnn: this.buildModelStatus('cnn'),
        },
        memoryEstimate: this.buildPipelineMemoryEstimate(),
      }
    } catch (error) {
      this.state = 'idle'
      if (error instanceof AiManagerError) throw error
      throw new AiManagerError(error instanceof Error ? error.message : 'Failed to load AI model.', 'ai_load_failed', 500)
    }
  }

  async predictMlp(backend: AiBackend, input: Float32Array): Promise<AiPredictResult> {
    if (input.length !== mlpLayout.inputSize) {
      throw new AiManagerError(`Invalid input length. Expected ${mlpLayout.inputSize} float values.`, 'invalid_input', 400)
    }

    if (backend === 'webgpu' && !this.mlpWebgpuModel) {
      throw new AiManagerError('MLP model is not loaded on WebGPU.', 'model_not_loaded', 409)
    }
    if (backend === 'cuda' && !this.mlpCudaReady) {
      throw new AiManagerError('MLP model is not loaded on CUDA.', 'model_not_loaded', 409)
    }

    const totalStart = performance.now()

    try {
      const backendResult =
        backend === 'webgpu'
          ? await predictWithWebGpuMlp(this.mlpWebgpuModel as LoadedWebGpuMlpModel, input)
          : await predictMlpCuda(input)

      const probabilities = softmax(backendResult.logits)
      const prediction = argmax(probabilities)
      const totalDurationMs = performance.now() - totalStart

      return {
        prediction,
        probabilities,
        gpuDurationMs: Number(backendResult.gpuDurationMs.toFixed(3)),
        totalDurationMs: Number(totalDurationMs.toFixed(3)),
        timingSource: backendResult.timingSource,
      }
    } catch (error) {
      throw new AiManagerError(error instanceof Error ? error.message : 'MLP prediction failed.', 'ai_predict_failed', 500)
    }
  }

  async predictCnn(backend: AiBackend, input: Float32Array): Promise<AiCnnPredictResult> {
    const expectedInput = cnnLayout.inputChannels * cnnLayout.inputHeight * cnnLayout.inputWidth
    if (input.length !== expectedInput) {
      throw new AiManagerError(`Invalid input length. Expected ${expectedInput} float values.`, 'invalid_input', 400)
    }

    if (backend === 'webgpu' && !this.cnnWebgpuModel) {
      throw new AiManagerError('CNN model is not loaded on WebGPU.', 'model_not_loaded', 409)
    }
    if (backend === 'cuda' && !this.cnnCudaReady) {
      throw new AiManagerError('CNN model is not loaded on CUDA.', 'model_not_loaded', 409)
    }

    const totalStart = performance.now()

    try {
      const backendResult =
        backend === 'webgpu'
          ? await predictWithWebGpuCnn(this.cnnWebgpuModel as LoadedWebGpuCnnModel, input)
          : await predictCnnCuda(input)

      const probabilities = softmax(backendResult.logits)
      const prediction = argmax(probabilities)
      const totalDurationMs = performance.now() - totalStart

      return {
        prediction,
        predictionLabel: CIFAR10_LABELS[prediction] ?? 'unknown',
        probabilities,
        gpuDurationMs: Number(backendResult.gpuDurationMs.toFixed(3)),
        totalDurationMs: Number(totalDurationMs.toFixed(3)),
        timingSource: backendResult.timingSource,
        memoryEstimate: this.buildSingleModelMemoryEstimate('cnn'),
      }
    } catch (error) {
      throw new AiManagerError(error instanceof Error ? error.message : 'CNN prediction failed.', 'ai_predict_failed', 500)
    }
  }

  async unloadModel(options?: AiLifecycleOptions): Promise<AiUnloadResult> {
    if (this.state !== 'idle') {
      throw new AiManagerError('Model lifecycle operation is already in progress.', 'ai_busy', 409)
    }

    const models = this.resolveModels(options?.model)
    const targets = this.resolveTargets(options)
    if (targets.length === 0) {
      throw new AiManagerError(
        'At least one backend flag must be true (`webgpu` or `cuda`).',
        'invalid_unload_targets',
        400,
      )
    }

    this.state = 'unloading'

    try {
      for (const model of models) {
        if (model === 'mlp') {
          if (targets.includes('webgpu') && this.mlpWebgpuModel) {
            unloadWebGpuMlpModel(this.mlpWebgpuModel)
            this.mlpWebgpuModel = null
            this.mlpWebgpuMemory = null
          }
          if (targets.includes('cuda') && this.mlpCudaReady) {
            await unloadModelCuda()
            this.mlpCudaReady = false
            this.mlpCudaMemory = null
          }
          if (!this.mlpWebgpuModel && !this.mlpCudaReady) {
            this.mlpHostWeightsBytes = 0
          }
        } else {
          if (targets.includes('webgpu') && this.cnnWebgpuModel) {
            unloadWebGpuCnnModel(this.cnnWebgpuModel)
            this.cnnWebgpuModel = null
            this.cnnWebgpuMemory = null
          }
          if (targets.includes('cuda') && this.cnnCudaReady) {
            await unloadCnnModelCuda()
            this.cnnCudaReady = false
            this.cnnCudaMemory = null
          }
          if (!this.cnnWebgpuModel && !this.cnnCudaReady) {
            this.cnnHostWeightsBytes = 0
          }
        }
      }

      this.state = 'idle'
      return {
        status: 'unloaded',
        loadedModels: this.getLoadedModels(),
        models: {
          mlp: this.buildModelStatus('mlp'),
          cnn: this.buildModelStatus('cnn'),
        },
        memoryEstimate: this.buildPipelineMemoryEstimate(),
      }
    } catch (error) {
      this.state = 'idle'
      throw new AiManagerError(error instanceof Error ? error.message : 'Failed to unload model.', 'ai_unload_failed', 500)
    }
  }
}

