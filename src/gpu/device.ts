/**
 * webgpu 0.3.x (Dawn/Node.js) — correct initialization:
 *
 *   const { create, globals } = await import('webgpu')
 *   Object.assign(globalThis, globals)   ← injects GPUBuffer, GPUDevice, etc.
 *   const gpu = create()                 ← GPU entry point (= navigator.gpu)
 */

// ─── Singletons ───────────────────────────────────────────────────────────────

let _gpu:        GPU        | null = null
let _adapter:    GPUAdapter | null = null
let _device:     GPUDevice  | null = null
let _initialized               = false

// ─── Bootstrap ────────────────────────────────────────────────────────────────

interface WebGpuModule {
  create:  (flags?: string[]) => GPU
  globals: Record<string, unknown>
}

function getRequiredDeviceLimits(adapter: GPUAdapter): GPUDeviceDescriptor['requiredLimits'] {
  return {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  }
}

async function bootstrap(): Promise<GPU> {
  if (_gpu) return _gpu

  // Dynamic import works in both ESM and tsx (no import.meta needed)
  const mod = await import('webgpu') as unknown as WebGpuModule

  // Inject GPUBuffer, GPUDevice, GPUTextureFormat… into globalThis
  Object.assign(globalThis, mod.globals)

  _gpu = mod.create([])
  return _gpu
}

// ─── AdapterInfo ──────────────────────────────────────────────────────────────

export interface AdapterInfo {
  vendor:       string
  architecture: string
  description:  string
  deviceId:     number
}

/**
 * Version-safe adapter info reader.
 * Handles both old spec (requestAdapterInfo async method)
 * and new spec (adapter.info sync property).
 */
export async function getAdapterInfo(adapter: GPUAdapter): Promise<AdapterInfo> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const a   = adapter as any
  const raw = typeof a.requestAdapterInfo === 'function'
      ? await a.requestAdapterInfo()
      : (a.info ?? {})

  return {
    vendor:       String(raw.vendor       ?? 'unknown'),
    architecture: String(raw.architecture ?? 'unknown'),
    description:  String(raw.description  ?? 'unknown'),
    deviceId:     Number(raw.deviceId     ?? 0),
  }
}

// ─── Adapter ──────────────────────────────────────────────────────────────────

/**
 * Returns the (cached) GPUAdapter.
 * Returns null on soft failure — /gpu/info uses this to avoid 500.
 */
export async function getGpuAdapter(): Promise<GPUAdapter | null> {
  if (_initialized) return _adapter

  try {
    const gpu = await bootstrap()
    _adapter = await gpu.requestAdapter({
      powerPreference:
          (process.env.GPU_POWER_PREFERENCE as GPUPowerPreference) ?? 'high-performance',
    })
  } catch (err) {
    console.warn('[GPU] Adapter request failed:', err)
    _adapter = null
  }

  _initialized = true
  return _adapter
}

// ─── Device ───────────────────────────────────────────────────────────────────

/**
 * Returns the (cached) GPUDevice.
 * Throws when no adapter / device is available.
 */
export async function getGpuDevice(): Promise<GPUDevice> {
  if (_device) return _device

  const adapter = await getGpuAdapter()
  if (!adapter) throw new Error('No WebGPU adapter found on this machine.')

  _device = await adapter.requestDevice({
    label: 'thesis-device',
    requiredLimits: getRequiredDeviceLimits(adapter),
  })

  _device.lost.then((info) => {
    console.warn(`[GPU] Device lost (reason: ${info.reason}): ${info.message}`)
    _device = null; _adapter = null; _gpu = null; _initialized = false
  })

  return _device
}

/**
 * Creates a dedicated GPUDevice instance (not cached singleton).
 * Useful for workloads that should fully release memory after completion.
 */
export async function createDedicatedGpuDevice(label = 'thesis-dedicated-device'): Promise<GPUDevice> {
  const adapter = await getGpuAdapter()
  if (!adapter) throw new Error('No WebGPU adapter found on this machine.')

  try {
    console.log(`[GPU] Creating dedicated device: "${label}"...`)
    const device = await adapter.requestDevice({
      label,
      requiredLimits: getRequiredDeviceLimits(adapter),
    })

    device.lost.then((info) => {
      console.warn(`[GPU] Dedicated device lost (reason: ${info.reason}): ${info.message}`)
    })

    console.log(`[GPU] ✓ Dedicated device created: "${label}"`)
    return device
  } catch (err) {
    console.error(`[GPU] ✗ Failed to create dedicated device:`, err)
    // Reset adapter on failure — it may be in bad state
    if (String(err).includes('consumed')) {
      console.warn(`[GPU] Adapter appears consumed, resetting singletons...`)
      _adapter = null
      _device = null
      _gpu = null
      _initialized = false
    }
    throw err
  }
}

// ─── Warmup ───────────────────────────────────────────────────────────────────

/**
 * Eagerly init adapter + device at startup.
 * Call once in index.ts — first HTTP request pays zero init cost.
 */
export async function warmupGpu(): Promise<void> {
  try {
    const device  = await getGpuDevice()
    const adapter = (await getGpuAdapter())!
    const info    = await getAdapterInfo(adapter)

    console.log(`[GPU] ✅ Ready — ${info.vendor} / ${info.architecture} (${info.description})`)

    const encoder = device.createCommandEncoder({ label: 'warmup' })
    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()
    console.log('[GPU] ✅ Command queue verified')
  } catch (err) {
    console.warn('[GPU] ⚠️  Warmup failed (server will still start):', err)
  }
}