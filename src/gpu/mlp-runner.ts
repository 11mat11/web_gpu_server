import { readFileSync } from 'node:fs';
import { getGpuDevice } from './device.js';

const MLP_INPUT_SIZE = 16384;
const MLP_HIDDEN1_SIZE = 2048;
const MLP_HIDDEN2_SIZE = 512;
const MLP_OUTPUT_SIZE = 10;

const W1_COUNT = MLP_INPUT_SIZE * MLP_HIDDEN1_SIZE;
const B1_COUNT = MLP_HIDDEN1_SIZE;
const W2_COUNT = MLP_HIDDEN1_SIZE * MLP_HIDDEN2_SIZE;
const B2_COUNT = MLP_HIDDEN2_SIZE;
const W3_COUNT = MLP_HIDDEN2_SIZE * MLP_OUTPUT_SIZE;
const B3_COUNT = MLP_OUTPUT_SIZE;

const TOTAL_WEIGHT_COUNT = W1_COUNT + B1_COUNT + W2_COUNT + B2_COUNT + W3_COUNT + B3_COUNT;
const GEMV_PARAMS_BYTES = 16;
const TOTAL_TIMESTAMP_BYTES = 48;

const shaderSource = readFileSync(new URL('./shaders/mlp.wgsl', import.meta.url), 'utf8');

type TimingSource = 'gpu-timestamp' | 'cpu-clock';

export interface MlpMemoryMetrics {
	gpuAllocatedBytes: number;
	hostAllocatedBytes: number;
}

export interface LoadedWebGpuMlpModel {
	device: GPUDevice;
	w1Buffer: GPUBuffer;
	b1Buffer: GPUBuffer;
	w2Buffer: GPUBuffer;
	b2Buffer: GPUBuffer;
	w3Buffer: GPUBuffer;
	b3Buffer: GPUBuffer;
	inputBuffer: GPUBuffer;
	hidden1Buffer: GPUBuffer;
	hidden2Buffer: GPUBuffer;
	outputBuffer: GPUBuffer;
	memory: MlpMemoryMetrics;
}

export interface WebGpuMlpPredictResult {
	logits: Float32Array;
	gpuDurationMs: number;
	backendDurationMs: number;
	timingSource: TimingSource;
	gpuMemoryBytes: number;
}

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();

function getMlpPipeline(device: GPUDevice): GPUComputePipeline {
	const cached = pipelineCache.get(device);
	if (cached) return cached;

	const module = device.createShaderModule({
		label: 'mlp-gemv-shader',
		code: shaderSource,
	});

	const pipeline = device.createComputePipeline({
		label: 'mlp-gemv-pipeline',
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'main',
		},
	});

	pipelineCache.set(device, pipeline);
	return pipeline;
}

function createGemvParams(inputSize: number, outputSize: number, applyRelu: boolean): Uint32Array {
	return new Uint32Array([inputSize, outputSize, applyRelu ? 1 : 0, 0]);
}

function readDurationMsFromRaw(start: bigint, end: bigint): number | null {
	const deltaTicks = end >= start ? end - start : (1n << 64n) - start + end;
	return deltaTicks > 0n ? Number(deltaTicks) / 1e6 : null;
}

function createStorageBuffer(device: GPUDevice, label: string, size: number): GPUBuffer {
	return device.createBuffer({
		label,
		size,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
}

export async function loadMlpModelToWebGpu(weights: Float32Array): Promise<LoadedWebGpuMlpModel> {
	if (weights.length !== TOTAL_WEIGHT_COUNT) {
		throw new Error(
			`Invalid MLP weights length. Expected ${TOTAL_WEIGHT_COUNT}, got ${weights.length}.`,
		);
	}

	const device = await getGpuDevice();
	const pipeline = getMlpPipeline(device);
	const pipelineLayout = pipeline.getBindGroupLayout(0);
	void pipelineLayout;

	const w1Bytes = W1_COUNT * Float32Array.BYTES_PER_ELEMENT;
	const b1Bytes = B1_COUNT * Float32Array.BYTES_PER_ELEMENT;
	const w2Bytes = W2_COUNT * Float32Array.BYTES_PER_ELEMENT;
	const b2Bytes = B2_COUNT * Float32Array.BYTES_PER_ELEMENT;
	const w3Bytes = W3_COUNT * Float32Array.BYTES_PER_ELEMENT;
	const b3Bytes = B3_COUNT * Float32Array.BYTES_PER_ELEMENT;

	const inputBytes = MLP_INPUT_SIZE * Float32Array.BYTES_PER_ELEMENT;
	const hidden1Bytes = MLP_HIDDEN1_SIZE * Float32Array.BYTES_PER_ELEMENT;
	const hidden2Bytes = MLP_HIDDEN2_SIZE * Float32Array.BYTES_PER_ELEMENT;
	const outputBytes = MLP_OUTPUT_SIZE * Float32Array.BYTES_PER_ELEMENT;

	const maxStorage = device.limits.maxStorageBufferBindingSize;
	for (const [name, bytes] of [
		['w1', w1Bytes],
		['w2', w2Bytes],
	] as const) {
		if (bytes > maxStorage) {
			throw new Error(
				`Buffer ${name} exceeds maxStorageBufferBindingSize (${bytes} > ${maxStorage}).`,
			);
		}
	}

	const w1Buffer = createStorageBuffer(device, 'mlp-w1', w1Bytes);
	const b1Buffer = createStorageBuffer(device, 'mlp-b1', b1Bytes);
	const w2Buffer = createStorageBuffer(device, 'mlp-w2', w2Bytes);
	const b2Buffer = createStorageBuffer(device, 'mlp-b2', b2Bytes);
	const w3Buffer = createStorageBuffer(device, 'mlp-w3', w3Bytes);
	const b3Buffer = createStorageBuffer(device, 'mlp-b3', b3Bytes);

	const inputBuffer = createStorageBuffer(device, 'mlp-input', inputBytes);
	const hidden1Buffer = createStorageBuffer(device, 'mlp-h1', hidden1Bytes);
	const hidden2Buffer = createStorageBuffer(device, 'mlp-h2', hidden2Bytes);
	const outputBuffer = device.createBuffer({
		label: 'mlp-output',
		size: outputBytes,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	try {
		let offset = 0;
		device.queue.writeBuffer(w1Buffer, 0, weights, offset, W1_COUNT);
		offset += W1_COUNT;
		device.queue.writeBuffer(b1Buffer, 0, weights, offset, B1_COUNT);
		offset += B1_COUNT;
		device.queue.writeBuffer(w2Buffer, 0, weights, offset, W2_COUNT);
		offset += W2_COUNT;
		device.queue.writeBuffer(b2Buffer, 0, weights, offset, B2_COUNT);
		offset += B2_COUNT;
		device.queue.writeBuffer(w3Buffer, 0, weights, offset, W3_COUNT);
		offset += W3_COUNT;
		device.queue.writeBuffer(b3Buffer, 0, weights, offset, B3_COUNT);

		await device.queue.onSubmittedWorkDone();

		const gpuAllocatedBytes =
			w1Bytes +
			b1Bytes +
			w2Bytes +
			b2Bytes +
			w3Bytes +
			b3Bytes +
			inputBytes +
			hidden1Bytes +
			hidden2Bytes +
			outputBytes;

		return {
			device,
			w1Buffer,
			b1Buffer,
			w2Buffer,
			b2Buffer,
			w3Buffer,
			b3Buffer,
			inputBuffer,
			hidden1Buffer,
			hidden2Buffer,
			outputBuffer,
			memory: {
				gpuAllocatedBytes,
				hostAllocatedBytes: weights.byteLength,
			},
		};
	} catch (error) {
		w1Buffer.destroy();
		b1Buffer.destroy();
		w2Buffer.destroy();
		b2Buffer.destroy();
		w3Buffer.destroy();
		b3Buffer.destroy();
		inputBuffer.destroy();
		hidden1Buffer.destroy();
		hidden2Buffer.destroy();
		outputBuffer.destroy();
		throw error;
	}
}

export async function predictWithWebGpuMlp(
	model: LoadedWebGpuMlpModel,
	input: Float32Array,
): Promise<WebGpuMlpPredictResult> {
	if (input.length !== MLP_INPUT_SIZE) {
		throw new Error(`Invalid input length. Expected ${MLP_INPUT_SIZE}.`);
	}
	const cpuStart = performance.now();
	const pipeline = getMlpPipeline(model.device);
	const device = model.device;
	const outputBytes = MLP_OUTPUT_SIZE * Float32Array.BYTES_PER_ELEMENT;

	const layer1Params = createGemvParams(MLP_INPUT_SIZE, MLP_HIDDEN1_SIZE, true);
	const layer2Params = createGemvParams(MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE, true);
	const layer3Params = createGemvParams(MLP_HIDDEN2_SIZE, MLP_OUTPUT_SIZE, false);

	const params1Buffer = device.createBuffer({
		label: 'mlp-params-1',
		size: GEMV_PARAMS_BYTES,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	const params2Buffer = device.createBuffer({
		label: 'mlp-params-2',
		size: GEMV_PARAMS_BYTES,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
	const params3Buffer = device.createBuffer({
		label: 'mlp-params-3',
		size: GEMV_PARAMS_BYTES,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	const logitsReadbackBuffer = device.createBuffer({
		label: 'mlp-logits-readback',
		size: outputBytes,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	const supportsTimestampQuery = device.features.has('timestamp-query');
	const querySet = supportsTimestampQuery
		? device.createQuerySet({
				label: 'mlp-query-set',
				type: 'timestamp',
				count: 6,
			})
		: null;
	const queryResolveBuffer = querySet
		? device.createBuffer({
				label: 'mlp-query-resolve',
				size: TOTAL_TIMESTAMP_BYTES,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			})
		: null;
	const queryReadbackBuffer = querySet
		? device.createBuffer({
				label: 'mlp-query-readback',
				size: TOTAL_TIMESTAMP_BYTES,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			})
		: null;

	try {
		device.queue.writeBuffer(model.inputBuffer, 0, input);
		device.queue.writeBuffer(params1Buffer, 0, layer1Params);
		device.queue.writeBuffer(params2Buffer, 0, layer2Params);
		device.queue.writeBuffer(params3Buffer, 0, layer3Params);

		const bindGroup1 = device.createBindGroup({
			label: 'mlp-bindgroup-1',
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: params1Buffer } },
				{ binding: 1, resource: { buffer: model.inputBuffer } },
				{ binding: 2, resource: { buffer: model.w1Buffer } },
				{ binding: 3, resource: { buffer: model.b1Buffer } },
				{ binding: 4, resource: { buffer: model.hidden1Buffer } },
			],
		});

		const bindGroup2 = device.createBindGroup({
			label: 'mlp-bindgroup-2',
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: params2Buffer } },
				{ binding: 1, resource: { buffer: model.hidden1Buffer } },
				{ binding: 2, resource: { buffer: model.w2Buffer } },
				{ binding: 3, resource: { buffer: model.b2Buffer } },
				{ binding: 4, resource: { buffer: model.hidden2Buffer } },
			],
		});

		const bindGroup3 = device.createBindGroup({
			label: 'mlp-bindgroup-3',
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: params3Buffer } },
				{ binding: 1, resource: { buffer: model.hidden2Buffer } },
				{ binding: 2, resource: { buffer: model.w3Buffer } },
				{ binding: 3, resource: { buffer: model.b3Buffer } },
				{ binding: 4, resource: { buffer: model.outputBuffer } },
			],
		});

		const encoder = device.createCommandEncoder({ label: 'mlp-inference-encoder' });

		const pass1 = encoder.beginComputePass({
			label: 'mlp-layer-1-pass',
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
		pass1.setPipeline(pipeline);
		pass1.setBindGroup(0, bindGroup1);
		pass1.dispatchWorkgroups(MLP_HIDDEN1_SIZE, 1, 1);
		pass1.end();

		const pass2 = encoder.beginComputePass({
			label: 'mlp-layer-2-pass',
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
		pass2.setPipeline(pipeline);
		pass2.setBindGroup(0, bindGroup2);
		pass2.dispatchWorkgroups(MLP_HIDDEN2_SIZE, 1, 1);
		pass2.end();

		const pass3 = encoder.beginComputePass({
			label: 'mlp-layer-3-pass',
			...(querySet
				? {
						timestampWrites: {
							querySet,
							beginningOfPassWriteIndex: 4,
							endOfPassWriteIndex: 5,
						},
					}
				: {}),
		});
		pass3.setPipeline(pipeline);
		pass3.setBindGroup(0, bindGroup3);
		pass3.dispatchWorkgroups(MLP_OUTPUT_SIZE, 1, 1);
		pass3.end();

		encoder.copyBufferToBuffer(model.outputBuffer, 0, logitsReadbackBuffer, 0, outputBytes);

		if (querySet && queryResolveBuffer && queryReadbackBuffer) {
			encoder.resolveQuerySet(querySet, 0, 6, queryResolveBuffer, 0);
			encoder.copyBufferToBuffer(
				queryResolveBuffer,
				0,
				queryReadbackBuffer,
				0,
				TOTAL_TIMESTAMP_BYTES,
			);
		}

		device.queue.submit([encoder.finish()]);

		const waitTasks: Promise<void>[] = [
			device.queue.onSubmittedWorkDone(),
			logitsReadbackBuffer.mapAsync(GPUMapMode.READ),
		];
		if (queryReadbackBuffer) {
			waitTasks.push(queryReadbackBuffer.mapAsync(GPUMapMode.READ));
		}
		await Promise.all(waitTasks);

		const logitsData = logitsReadbackBuffer.getMappedRange();
		const logits = new Float32Array(logitsData.slice(0));
		logitsReadbackBuffer.unmap();

		const cpuDurationMs = performance.now() - cpuStart;
		let gpuDurationMs = cpuDurationMs;
		let timingSource: TimingSource = 'cpu-clock';

		if (queryReadbackBuffer) {
			const mapped = queryReadbackBuffer.getMappedRange();
			const timestamps = new BigUint64Array(mapped.slice(0));
			queryReadbackBuffer.unmap();

			if (timestamps.length >= 6) {
				const l1 = readDurationMsFromRaw(timestamps[0], timestamps[1]);
				const l2 = readDurationMsFromRaw(timestamps[2], timestamps[3]);
				const l3 = readDurationMsFromRaw(timestamps[4], timestamps[5]);

				if (l1 !== null && l2 !== null && l3 !== null) {
					gpuDurationMs = l1 + l2 + l3;
					timingSource = 'gpu-timestamp';
				}
			}
		}

		return {
			logits,
			gpuDurationMs,
			backendDurationMs: cpuDurationMs,
			timingSource,
			gpuMemoryBytes:
				params1Buffer.size +
				params2Buffer.size +
				params3Buffer.size +
				logitsReadbackBuffer.size +
				(queryResolveBuffer?.size ?? 0) +
				(queryReadbackBuffer?.size ?? 0),
		};
	} finally {
		params1Buffer.destroy();
		params2Buffer.destroy();
		params3Buffer.destroy();
		logitsReadbackBuffer.destroy();
		if (queryResolveBuffer) queryResolveBuffer.destroy();
		if (queryReadbackBuffer) queryReadbackBuffer.destroy();
		if (querySet) querySet.destroy();
	}
}

export function unloadWebGpuMlpModel(model: LoadedWebGpuMlpModel): void {
	model.w1Buffer.destroy();
	model.b1Buffer.destroy();
	model.w2Buffer.destroy();
	model.b2Buffer.destroy();
	model.w3Buffer.destroy();
	model.b3Buffer.destroy();
	model.inputBuffer.destroy();
	model.hidden1Buffer.destroy();
	model.hidden2Buffer.destroy();
	model.outputBuffer.destroy();
}

export const mlpLayout = {
	inputSize: MLP_INPUT_SIZE,
	hidden1Size: MLP_HIDDEN1_SIZE,
	hidden2Size: MLP_HIDDEN2_SIZE,
	outputSize: MLP_OUTPUT_SIZE,
	totalWeightCount: TOTAL_WEIGHT_COUNT,
} as const;
