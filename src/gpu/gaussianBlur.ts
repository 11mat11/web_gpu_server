import { readFileSync } from 'node:fs';
import { getGpuDevice } from './device.js';

const WORKGROUP_SIZE = 16;
const KERNEL_WEIGHTS = [0.0625, 0.25, 0.375, 0.25, 0.0625];

const gaussianBlurShaderSource = readFileSync(
	new URL('./shaders/gaussianBlur.wgsl', import.meta.url),
	'utf8',
);

type TimingSource = 'gpu-timestamp' | 'cpu-clock';

export interface GaussianBlurWebGpuOptions {
	readback?: boolean;
}

export interface GaussianBlurWebGpuResult {
	output: Uint32Array | null;
	gpuDurationMs: number;
	backendDurationMs: number;
	timingSource: TimingSource;
	gpuMemoryBytes: number;
}

type GaussianPipelines = {
	horizontal: GPUComputePipeline;
	vertical: GPUComputePipeline;
};

const gaussianPipelineCache = new WeakMap<GPUDevice, GaussianPipelines>();

function getGaussianPipelines(device: GPUDevice): GaussianPipelines {
	const cached = gaussianPipelineCache.get(device);
	if (cached) return cached;

	const module = device.createShaderModule({
		label: 'gaussian-blur-shader',
		code: gaussianBlurShaderSource,
	});

	const horizontal = device.createComputePipeline({
		label: 'gaussian-blur-horizontal-pipeline',
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'mainHorizontal',
		},
	});

	const vertical = device.createComputePipeline({
		label: 'gaussian-blur-vertical-pipeline',
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'mainVertical',
		},
	});

	const pipelines = { horizontal, vertical };
	gaussianPipelineCache.set(device, pipelines);
	return pipelines;
}

function readDurationMsFromRawTimestamps(start: bigint, end: bigint): number | null {
	const deltaTicks = end >= start ? end - start : (1n << 64n) - start + end;
	return deltaTicks > 0n ? Number(deltaTicks) / 1e6 : null;
}

export function packRgbaBytesToU32(input: Uint8Array): Uint32Array {
	const pixels = Math.floor(input.length / 4);
	const packed = new Uint32Array(pixels);

	for (let i = 0; i < pixels; i++) {
		const base = i * 4;
		const r = input[base] ?? 0;
		const g = input[base + 1] ?? 0;
		const b = input[base + 2] ?? 0;
		const a = input[base + 3] ?? 255;
		packed[i] = r | (g << 8) | (b << 16) | (a << 24);
	}

	return packed;
}

export function unpackU32ToRgbaBytes(input: Uint32Array): Uint8Array {
	const out = new Uint8Array(input.length * 4);
	for (let i = 0; i < input.length; i++) {
		const value = input[i] ?? 0;
		const base = i * 4;
		out[base] = value & 0xff;
		out[base + 1] = (value >>> 8) & 0xff;
		out[base + 2] = (value >>> 16) & 0xff;
		out[base + 3] = (value >>> 24) & 0xff;
	}
	return out;
}

function clampByte(value: number): number {
	if (value < 0) return 0;
	if (value > 255) return 255;
	return value;
}

export function gaussianBlurCpu(input: Uint32Array, width: number, height: number): Uint32Array {
	const totalPixels = width * height;
	if (input.length !== totalPixels) {
		throw new Error(`Invalid input length. Expected ${totalPixels} pixels.`);
	}

	const temp = new Uint32Array(totalPixels);
	const output = new Uint32Array(totalPixels);

	for (let y = 0; y < height; y++) {
		const rowOffset = y * width;
		for (let x = 0; x < width; x++) {
			let r = 0;
			let g = 0;
			let b = 0;
			let a = 0;

			for (let k = -2; k <= 2; k++) {
				const sx = Math.min(Math.max(x + k, 0), width - 1);
				const pixel = input[rowOffset + sx] ?? 0;
				const weight = KERNEL_WEIGHTS[k + 2] ?? 0;

				r += (pixel & 0xff) * weight;
				g += ((pixel >>> 8) & 0xff) * weight;
				b += ((pixel >>> 16) & 0xff) * weight;
				a += ((pixel >>> 24) & 0xff) * weight;
			}

			const packed =
				clampByte(Math.round(r)) |
				(clampByte(Math.round(g)) << 8) |
				(clampByte(Math.round(b)) << 16) |
				(clampByte(Math.round(a)) << 24);
			temp[rowOffset + x] = packed >>> 0;
		}
	}

	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			let r = 0;
			let g = 0;
			let b = 0;
			let a = 0;

			for (let k = -2; k <= 2; k++) {
				const sy = Math.min(Math.max(y + k, 0), height - 1);
				const pixel = temp[sy * width + x] ?? 0;
				const weight = KERNEL_WEIGHTS[k + 2] ?? 0;

				r += (pixel & 0xff) * weight;
				g += ((pixel >>> 8) & 0xff) * weight;
				b += ((pixel >>> 16) & 0xff) * weight;
				a += ((pixel >>> 24) & 0xff) * weight;
			}

			const packed =
				clampByte(Math.round(r)) |
				(clampByte(Math.round(g)) << 8) |
				(clampByte(Math.round(b)) << 16) |
				(clampByte(Math.round(a)) << 24);
			output[y * width + x] = packed >>> 0;
		}
	}

	return output;
}

export async function gaussianBlurWebGpu(
	width: number,
	height: number,
	input: Uint32Array,
	options: GaussianBlurWebGpuOptions = {},
): Promise<GaussianBlurWebGpuResult> {
	const totalPixels = width * height;
	if (input.length !== totalPixels) {
		throw new Error(`Invalid input length. Expected ${totalPixels} pixels.`);
	}

	const device = await getGpuDevice();
	const { horizontal, vertical } = getGaussianPipelines(device);
	const shouldReadback = options.readback ?? false;
	const bufferBytes = totalPixels * Uint32Array.BYTES_PER_ELEMENT;

	if (bufferBytes > device.limits.maxStorageBufferBindingSize) {
		throw new Error('Image buffer size exceeds GPU limits.');
	}
	const cpuStartMs = performance.now();
	const sizeData = new Uint32Array([width, height, 0, 0]);
	const sizeBuffer = device.createBuffer({
		label: 'gaussian-size',
		size: sizeData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	const srcBuffer = device.createBuffer({
		label: 'gaussian-src',
		size: bufferBytes,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	const tempBuffer = device.createBuffer({
		label: 'gaussian-temp',
		size: bufferBytes,
		usage: GPUBufferUsage.STORAGE,
	});

	const dstBuffer = device.createBuffer({
		label: 'gaussian-dst',
		size: bufferBytes,
		usage: GPUBufferUsage.STORAGE | (shouldReadback ? GPUBufferUsage.COPY_SRC : 0),
	});

	const readbackBuffer = shouldReadback
		? device.createBuffer({
				label: 'gaussian-readback',
				size: bufferBytes,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			})
		: null;

	const supportsTimestampQuery = device.features.has('timestamp-query');
	const querySet = supportsTimestampQuery
		? device.createQuerySet({
				label: 'gaussian-blur-timestamps',
				type: 'timestamp',
				count: 4,
			})
		: null;
	const queryResolveBuffer = querySet
		? device.createBuffer({
				label: 'gaussian-blur-query-resolve',
				size: 32,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			})
		: null;
	const queryReadbackBuffer = querySet
		? device.createBuffer({
				label: 'gaussian-blur-query-readback',
				size: 32,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			})
		: null;

	const gpuMemoryBytes =
		sizeBuffer.size +
		srcBuffer.size +
		tempBuffer.size +
		dstBuffer.size +
		(readbackBuffer?.size ?? 0) +
		(queryResolveBuffer?.size ?? 0) +
		(queryReadbackBuffer?.size ?? 0);

	try {
		device.queue.writeBuffer(sizeBuffer, 0, sizeData);
		device.queue.writeBuffer(srcBuffer, 0, input);

		const horizontalBindGroup = device.createBindGroup({
			label: 'gaussian-horizontal-bind-group',
			layout: horizontal.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: sizeBuffer } },
				{ binding: 1, resource: { buffer: srcBuffer } },
				{ binding: 2, resource: { buffer: tempBuffer } },
			],
		});

		const verticalBindGroup = device.createBindGroup({
			label: 'gaussian-vertical-bind-group',
			layout: vertical.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: sizeBuffer } },
				{ binding: 1, resource: { buffer: tempBuffer } },
				{ binding: 2, resource: { buffer: dstBuffer } },
			],
		});

		const encoder = device.createCommandEncoder({ label: 'gaussian-blur-encoder' });

		const horizontalPass = encoder.beginComputePass({
			label: 'gaussian-blur-horizontal-pass',
			...(querySet
				? {
						timestampWrites: {
							querySet,
							beginningOfPassWriteIndex: 0,
							endOfPassWriteIndex: 1,
						},
					}
				: {}),
		});
		horizontalPass.setPipeline(horizontal);
		horizontalPass.setBindGroup(0, horizontalBindGroup);
		horizontalPass.dispatchWorkgroups(
			Math.ceil(width / WORKGROUP_SIZE),
			Math.ceil(height / WORKGROUP_SIZE),
			1,
		);
		horizontalPass.end();

		const verticalPass = encoder.beginComputePass({
			label: 'gaussian-blur-vertical-pass',
			...(querySet
				? {
						timestampWrites: {
							querySet,
							beginningOfPassWriteIndex: 2,
							endOfPassWriteIndex: 3,
						},
					}
				: {}),
		});
		verticalPass.setPipeline(vertical);
		verticalPass.setBindGroup(0, verticalBindGroup);
		verticalPass.dispatchWorkgroups(
			Math.ceil(width / WORKGROUP_SIZE),
			Math.ceil(height / WORKGROUP_SIZE),
			1,
		);
		verticalPass.end();

		if (readbackBuffer) {
			encoder.copyBufferToBuffer(dstBuffer, 0, readbackBuffer, 0, bufferBytes);
		}

		if (querySet && queryResolveBuffer && queryReadbackBuffer) {
			encoder.resolveQuerySet(querySet, 0, 4, queryResolveBuffer, 0);
			encoder.copyBufferToBuffer(queryResolveBuffer, 0, queryReadbackBuffer, 0, 32);
		}

		device.queue.submit([encoder.finish()]);

		const waits: Promise<void>[] = [];
		waits.push(device.queue.onSubmittedWorkDone());
		if (readbackBuffer) waits.push(readbackBuffer.mapAsync(GPUMapMode.READ));
		if (queryReadbackBuffer) waits.push(queryReadbackBuffer.mapAsync(GPUMapMode.READ));
		await Promise.all(waits);

		const cpuDurationMs = performance.now() - cpuStartMs;

		let output: Uint32Array | null = null;
		if (readbackBuffer) {
			const mapped = readbackBuffer.getMappedRange();
			output = new Uint32Array(mapped.slice(0));
			readbackBuffer.unmap();
		}

		let gpuDurationMs = cpuDurationMs;
		let timingSource: TimingSource = 'cpu-clock';

		if (queryReadbackBuffer) {
			const mapped = queryReadbackBuffer.getMappedRange();
			const timestamps = new BigUint64Array(mapped.slice(0));
			queryReadbackBuffer.unmap();

			if (timestamps.length >= 4) {
				const horizontalMs = readDurationMsFromRawTimestamps(timestamps[0], timestamps[1]);
				const verticalMs = readDurationMsFromRawTimestamps(timestamps[2], timestamps[3]);
				if (horizontalMs !== null && verticalMs !== null) {
					gpuDurationMs = horizontalMs + verticalMs;
					timingSource = 'gpu-timestamp';
				}
			}
		}

		return {
			output,
			gpuDurationMs,
			backendDurationMs: cpuDurationMs,
			timingSource,
			gpuMemoryBytes,
		};
	} finally {
		sizeBuffer.destroy();
		srcBuffer.destroy();
		tempBuffer.destroy();
		dstBuffer.destroy();
		if (readbackBuffer) readbackBuffer.destroy();
		if (queryResolveBuffer) queryResolveBuffer.destroy();
		if (queryReadbackBuffer) queryReadbackBuffer.destroy();
		if (querySet) querySet.destroy();
	}
}
