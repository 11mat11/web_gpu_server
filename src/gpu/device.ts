/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-base-to-string */

let _gpu: GPU | null = null;
let _adapter: GPUAdapter | null = null;
let _device: GPUDevice | null = null;
let _initialized = false;
let _recoveryPromise: Promise<void> | null = null;
interface WebGpuModule {
	create: (flags?: string[]) => GPU;
	globals: Record<string, unknown>;
}

export function serializeGpuLimits(limits: GPUSupportedLimits): Record<string, number> {
	const source = limits as unknown as Record<string, unknown>;
	const keys = new Set<string>(Object.keys(source));
	const proto = Object.getPrototypeOf(source);

	if (proto) {
		for (const key of Object.getOwnPropertyNames(proto)) {
			if (key !== 'constructor') keys.add(key);
		}
	}

	const serialized: Record<string, number> = {};
	for (const key of keys) {
		const value = source[key];
		if (typeof value === 'number') serialized[key] = value;
	}

	return serialized;
}
async function bootstrap(): Promise<GPU> {
	if (_gpu) return _gpu;

	const mod = (await import('webgpu')) as unknown as WebGpuModule;

	Object.assign(globalThis, mod.globals);

	_gpu = mod.create([]);
	return _gpu;
}

// ─── AdapterInfo ──────────────────────────────────────────────────────────────

export interface AdapterInfo {
	vendor: string;
	architecture: string;
	description: string;
	deviceId: number;
	backend: string;
	driver: string;
}

function normalizeString(value: unknown): string | null {
	if (typeof value !== 'string') return null;
	const trimmed = value.trim();
	if (!trimmed) return null;
	if (trimmed.toLowerCase() === 'unknown') return null;
	return trimmed;
}

function firstKnownString(...values: unknown[]): string | null {
	for (const value of values) {
		const normalized = normalizeString(value);
		if (normalized) return normalized;
	}
	return null;
}

function inferBackendFromDescription(description: string): string | null {
	const lower = description.toLowerCase();
	if (lower.includes('d3d12') || lower.includes('direct3d 12')) return 'd3d12';
	if (lower.includes('d3d11') || lower.includes('direct3d 11')) return 'd3d11';
	if (lower.includes('vulkan')) return 'vulkan';
	if (lower.includes('metal')) return 'metal';
	if (lower.includes('opengl')) return 'opengl';
	return null;
}

function inferDriverFromDescription(description: string): string | null {
	const match = description.match(/driver\s+version\s+([0-9.]+)/i);
	if (!match) return null;
	return normalizeString(match[1]);
}

export async function getAdapterInfo(adapter: GPUAdapter): Promise<AdapterInfo> {
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const a = adapter as any;
	const raw =
		typeof a.requestAdapterInfo === 'function' ? await a.requestAdapterInfo() : (a.info ?? {});

	const rawRecord = raw as Record<string, unknown>;
	const description = String(rawRecord.description ?? 'unknown');

	const backend = firstKnownString(
		rawRecord.backendType,
		rawRecord.backend,
		inferBackendFromDescription(description),
	);

	const driver = firstKnownString(
		rawRecord.driver,
		rawRecord.driverInfo,
		rawRecord.driverVersion,
		inferDriverFromDescription(description),
	);

	return {
		vendor: String(rawRecord.vendor ?? 'unknown'),
		architecture: String(rawRecord.architecture ?? 'unknown'),
		description,
		deviceId: Number(rawRecord.deviceId ?? 0),
		backend: backend ?? 'unknown',
		driver: driver ?? 'unknown',
	};
}

function getRequiredDeviceFeatures(adapter: GPUAdapter): GPUFeatureName[] {
	const requiredFeatures: GPUFeatureName[] = [];
	const timestampFeature = 'timestamp-query' as GPUFeatureName;

	if (adapter.features.has(timestampFeature)) {
		requiredFeatures.push(timestampFeature);
	}

	return requiredFeatures;
}

// ─── Adapter ──────────────────────────────────────────────────────────────────

export function getRequiredDeviceLimits(
	adapter: GPUAdapter,
): GPUDeviceDescriptor['requiredLimits'] {
	const limits: Record<string, number> = {
		maxBufferSize: adapter.limits.maxBufferSize,
		maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
	};

	const computeLimits = [
		'maxComputeWorkgroupStorageSize',
		'maxComputeInvocationsPerWorkgroup',
		'maxComputeWorkgroupSizeX',
		'maxComputeWorkgroupSizeY',
		'maxComputeWorkgroupSizeZ',
		'maxComputeWorkgroupsPerDimension',
	] as const;

	for (const key of computeLimits) {
		if (adapter.limits[key]) {
			limits[key] = adapter.limits[key];
		}
	}

	return limits;
}

export async function getGpuAdapter(): Promise<GPUAdapter | null> {
	if (_initialized) return _adapter;

	try {
		const gpu = await bootstrap();
		_adapter = await gpu.requestAdapter({
			powerPreference:
				(process.env.GPU_POWER_PREFERENCE as GPUPowerPreference) ?? 'high-performance',
		});
	} catch (err) {
		console.warn('[GPU] Adapter request failed:', err);
		_adapter = null;
	}

	_initialized = true;
	return _adapter;
}

// ─── Device ───────────────────────────────────────────────────────────────────

export async function getGpuDevice(): Promise<GPUDevice> {
	if (_recoveryPromise) {
		console.warn("[GPU] Oczekiwanie na reset sterownika Windows (TDR)...");
		await _recoveryPromise;
	}

	if (_device) return _device;

	const adapter = await getGpuAdapter();
	if (!adapter) throw new Error('No WebGPU adapter found on this machine.');

	const requiredFeatures = getRequiredDeviceFeatures(adapter);
	_device = await adapter.requestDevice({
		label: 'thesis-device',
		...(requiredFeatures.length ? { requiredFeatures } : {}),
		requiredLimits: getRequiredDeviceLimits(adapter),
	});

	void _device.lost.then((info) => {
		console.warn(`[GPU] Device lost (reason: ${info.reason}): ${info.message}`);
		_device = null;
		_adapter = null;
		_gpu = null;
		_initialized = false;

		_recoveryPromise = new Promise(resolve => {
			setTimeout(() => {
				_recoveryPromise = null;
				console.log("[GPU] Przerwa na reset sterownika zakończona. Można próbować ponownie.");
				resolve();
			}, 5000);
		});
	});

	return _device;
}

// ─── DedicatedDevice ──────────────────────────────────────────────────────────

export async function createDedicatedGpuDevice(
	label = 'thesis-dedicated-device',
): Promise<GPUDevice> {
	const adapter = await getGpuAdapter();
	if (!adapter) throw new Error('No WebGPU adapter found on this machine.');

	try {
		console.log(`[GPU] Creating dedicated device: "${label}"...`);
		const requiredFeatures = getRequiredDeviceFeatures(adapter);
		const device = await adapter.requestDevice({
			label,
			...(requiredFeatures.length ? { requiredFeatures } : {}),
			requiredLimits: getRequiredDeviceLimits(adapter),
		});

		void device.lost.then((info) => {
			console.warn(`[GPU] Dedicated device lost (reason: ${info.reason}): ${info.message}`);
		});

		console.log(`[GPU] ✓ Dedicated device created: "${label}"`);
		return device;
	} catch (err) {
		console.error(`[GPU] ✗ Failed to create dedicated device:`, err);
		// Reset adapter on failure — it may be in bad state
		if (String(err).includes('consumed')) {
			console.warn(`[GPU] Adapter appears consumed, resetting singletons...`);
			_adapter = null;
			_device = null;
			_gpu = null;
			_initialized = false;
		}
		throw err;
	}
}

// ─── Warmup ───────────────────────────────────────────────────────────────────

export async function warmupGpu(): Promise<void> {
	try {
		const device = await getGpuDevice();
		const adapter = (await getGpuAdapter())!;
		const info = await getAdapterInfo(adapter);

		console.log(`[GPU] ✅ Ready — ${info.vendor} / ${info.architecture} (${info.description})`);

		const encoder = device.createCommandEncoder({ label: 'warmup' });
		device.queue.submit([encoder.finish()]);
		await device.queue.onSubmittedWorkDone();
		console.log('[GPU] ✅ Command queue verified');
	} catch (err) {
		console.warn('[GPU] ⚠️  Warmup failed (server will still start):', err);
	}
}
