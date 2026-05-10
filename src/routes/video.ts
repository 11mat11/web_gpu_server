import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { encode as encodeJpeg } from 'jpeg-js';
import {
	computeHistogramCuda,
	getCudaRuntimeState,
	initVideoPipelineCuda,
	processVideoFrameCuda,
	unloadVideoPipelineCuda,
} from '../cuda/cudaBackend.js';
import {
	computeHistogramWebGpu,
	initWebGpuVideoPipeline,
	processVideoFrameWebGpu,
	unloadWebGpuVideoPipeline,
	type LoadedWebGpuVideoPipeline,
	type VideoQuality,
	videoLayout,
} from '../gpu/video-runner.js';
import { VideoManager } from '../video/VideoManager.js';
interface FastifyStreamSocket {
	on(event: 'message', listener: (raw: Buffer) => Promise<void> | void): void;
	on(event: 'close', listener: () => Promise<void> | void): void;
	send(data: string, cb?: (err?: Error) => void): void;
}
type StreamBackend = 'webgpu' | 'cuda';
type WithMemory = { memory: { gpuAllocatedBytes: number } };
type WithGpuBytes = { gpuMemoryBytes: number };
const SelectSchema = z
	.object({
		action: z.literal('select').describe('Akcja wyboru źródła wideo.').example('select'),
		fileName: z
			.string()
			.min(1)
			.describe('Nazwa pliku BIN z klatkami RGBA.')
			.example('video_frames_1080p_rgba.bin'),
		backend: z.enum(['webgpu', 'cuda']).describe('Backend przetwarzania wideo.').example('webgpu'),
		quality: z
			.enum(['1080p', '720p', '480p', '160p'])
			.describe('Docelowa jakość/rozmiar klatki.')
			.example('720p'),
		compress: z.boolean().describe('Czy kompresować ramkę do JPEG (stream).').example(false),
	})
	.describe('Komenda sterująca wyborem strumienia wideo.')
	.example({
		action: 'select',
		fileName: 'video_frames_1080p_rgba.bin',
		backend: 'webgpu',
		quality: '720p',
		compress: false,
	});

const ResizeSchema = z
	.object({
		action: z.literal('resize').describe('Akcja zmiany jakości strumienia.').example('resize'),
		quality: z
			.enum(['1080p', '720p', '480p', '160p'])
			.describe('Nowa jakość/rozmiar klatki.')
			.example('480p'),
	})
	.describe('Komenda zmiany jakości strumienia.')
	.example({ action: 'resize', quality: '480p' });

const PauseSchema = z
	.object({
		action: z.literal('pause').describe('Akcja wstrzymania streamu.').example('pause'),
	})
	.describe('Komenda pauzy strumienia.')
	.example({ action: 'pause' });

const ResumeSchema = z
	.object({
		action: z.literal('resume').describe('Akcja wznowienia streamu.').example('resume'),
	})
	.describe('Komenda wznowienia strumienia.')
	.example({ action: 'resume' });

const StopSchema = z
	.object({
		action: z.literal('stop').describe('Akcja zatrzymania streamu.').example('stop'),
	})
	.describe('Komenda zatrzymania strumienia.')
	.example({ action: 'stop' });

const HistogramBodySchema = z
	.object({
		fileName: z
			.string()
			.min(1)
			.describe('Nazwa pliku BIN z klatkami RGBA.')
			.example('video_frames_1080p_rgba.bin'),
		frameIndex: z.number().int().min(0).describe('Indeks klatki do analizy histogramu.').example(0),
		backend: z.enum(['webgpu', 'cuda']).describe('Backend obliczeń histogramu.').example('webgpu'),
	})
	.describe('Parametry żądania histogramu RGB dla pojedynczej klatki.')
	.example({ fileName: 'video_frames_1080p_rgba.bin', frameIndex: 0, backend: 'webgpu' });

function asBase64(buffer: Buffer): string {
	return buffer.toString('base64');
}

function sliceFrame(video: Buffer, frameIndex: number): Buffer {
	const offset = frameIndex * videoLayout.srcFrameBytes;
	return video.subarray(offset, offset + videoLayout.srcFrameBytes);
}

export async function videoRoute(server: FastifyInstance) {
	const manager = VideoManager.getInstance();

	server.post(
		'/histogram',
		{
			schema: {
				tags: ['video'],
				summary: 'Single-shot RGB histogram for a specific 1080p frame',
				body: {
					type: 'object',
					required: ['fileName', 'frameIndex', 'backend'],
					properties: {
						fileName: { type: 'string' },
						frameIndex: { type: 'number' },
						backend: { type: 'string', enum: ['webgpu', 'cuda'] },
					},
				},
				response: {
					200: {
						type: 'object',
						properties: {
							histogram: { type: 'array', items: { type: 'number' }, minItems: 768, maxItems: 768 },
							gpuDurationMs: { type: 'number' },
							backendDurationMs: { type: 'number' },
							serverDurationMs: { type: 'number' },
							timingSource: { type: 'string', enum: ['gpu-timestamp', 'cpu-clock'] },
							backend: { type: 'string', enum: ['webgpu', 'cuda'] },
							memory: {
								type: 'object',
								description: 'Ujednolicony raport pamięci dla żądania obliczeniowego.',
								properties: {
									gpuBytes: { type: ['number', 'null'] },
									hostBytes: { type: ['number', 'null'] },
									serverRssBytes: { type: 'number' },
								},
							},
						},
					},
					400: {
						type: 'object',
						properties: { error: { type: 'string' }, message: { type: 'string' } },
					},
					500: {
						type: 'object',
						properties: { error: { type: 'string' }, message: { type: 'string' } },
					},
				},
			},
		},
		async (req, reply) => {
			const parsed = HistogramBodySchema.safeParse(req.body);
			if (!parsed.success) {
				return reply.code(400).send({
					error: 'invalid_input',
					message: 'Body must contain fileName, frameIndex >= 0 and backend (webgpu|cuda).',
				});
			}

			try {
				const startedAt = performance.now();
				const video = await manager.getVideoBuffer(parsed.data.fileName);

				if (video.byteLength % videoLayout.srcFrameBytes !== 0) {
					return reply.code(400).send({
						error: 'invalid_video_file',
						message: 'Video file is not aligned to 1080p RGBA frame size.',
					});
				}

				const frameCount = video.byteLength / videoLayout.srcFrameBytes;
				if (parsed.data.frameIndex >= frameCount) {
					return reply.code(400).send({
						error: 'frame_index_out_of_range',
						message: `frameIndex must be in range [0, ${Math.max(frameCount - 1, 0)}].`,
					});
				}

				const frame = sliceFrame(video, parsed.data.frameIndex);

				const result =
					parsed.data.backend === 'webgpu'
						? await computeHistogramWebGpu(frame)
						: await computeHistogramCuda(frame);

				const serverDurationMs = performance.now() - startedAt;
				const memory = {
					gpuBytes: 'memory' in result ? result.memory.gpuAllocatedBytes : result.gpuMemoryBytes,
					hostBytes: video.byteLength + result.histogram.length * Uint32Array.BYTES_PER_ELEMENT,
					serverRssBytes: process.memoryUsage().rss,
				};
				return reply.send({
					histogram: result.histogram,
					gpuDurationMs: Number(result.gpuDurationMs.toFixed(3)),
					backendDurationMs: Number(result.backendDurationMs.toFixed(3)),
					serverDurationMs: Number(serverDurationMs.toFixed(3)),
					timingSource: result.timingSource,
					backend: parsed.data.backend,
					memory,
				});
			} catch (error) {
				return reply.code(500).send({
					error: 'histogram_failed',
					message: error instanceof Error ? error.message : 'Failed to compute video histogram.',
				});
			}
		},
	);

	server.get(
		'/list',
		{
			schema: {
				tags: ['video'],
				summary: 'Lista plików wideo BIN dostępnych do streamingu',
				response: {
					200: {
						type: 'object',
						properties: {
							files: { type: 'array', items: { type: 'string' } },
						},
					},
					400: {
						type: 'object',
						properties: {
							error: { type: 'string' },
							message: { type: 'string' },
						},
					},
					500: {
						type: 'object',
						properties: {
							error: { type: 'string' },
							message: { type: 'string' },
						},
					},
				},
			},
		},
		async (_req, reply) => {
			try {
				const files = await manager.listVideos();
				return reply.send({ files });
			} catch (error) {
				return reply.code(500).send({
					error: 'video_list_failed',
					message: error instanceof Error ? error.message : 'Failed to list videos.',
				});
			}
		},
	);

	server.get('/stream', { websocket: true }, (connection: unknown) => {
		const socket = connection as FastifyStreamSocket;
		let isStreaming = false;
		let isPaused = false;
		let isFrameInFlight = false;
		let isFrameScheduled = false;

		let selectedBackend: StreamBackend = 'webgpu';
		let selectedQuality: VideoQuality = '1080p';
		let selectedCompress = false;

		let loadedVideo: Buffer | null = null;
		let frameCount = 0;
		let frameIndex = 0;

		let webgpuPipeline: LoadedWebGpuVideoPipeline | null = null;
		let cudaPipelineReady = false;
		let pipelineGpuBytes = 0;

		const cleanupBackend = async (): Promise<void> => {
			if (webgpuPipeline) {
				unloadWebGpuVideoPipeline(webgpuPipeline);
				webgpuPipeline = null;
			}

			if (cudaPipelineReady) {
				await unloadVideoPipelineCuda();
				cudaPipelineReady = false;
			}

			pipelineGpuBytes = 0;
		};

		const stopStreaming = async (): Promise<void> => {
			isStreaming = false;
			isPaused = false;
			isFrameScheduled = false;
			await cleanupBackend();
			loadedVideo = null;
			frameCount = 0;
			frameIndex = 0;
		};

		const scheduleNextFrame = (): void => {
			if (!isStreaming || isPaused || isFrameScheduled) return;
			isFrameScheduled = true;
			setImmediate(() => {
				isFrameScheduled = false;
				void streamNextFrame();
			});
		};

		const sendError = (code: string, message: string): void => {
			socket.send(
				JSON.stringify({
					type: 'error',
					error: code,
					message,
				}),
			);
		};

		const streamNextFrame = async (): Promise<void> => {
			if (!isStreaming || isPaused || !loadedVideo || frameCount === 0) return;
			if (isFrameInFlight) return;

			isFrameInFlight = true;

			const startedAt = performance.now();
			const sourceFrame = sliceFrame(loadedVideo, frameIndex);

			try {
				let rgba = sourceFrame;
				let gpuDurationMs = 0;
				let backendDurationMs = 0;
				let timingSource: 'gpu-timestamp' | 'cpu-clock' = 'cpu-clock';
				let width: number = videoLayout.srcWidth;
				let height: number = videoLayout.srcHeight;
				let gpuBytes: number | null = null;
				let hostBytes = 0;

				if (selectedQuality !== '1080p') {
					if (selectedBackend === 'webgpu') {
						if (!webgpuPipeline) {
							throw new Error('WebGPU pipeline is not initialized.');
						}
						const result = await processVideoFrameWebGpu(
							webgpuPipeline,
							sourceFrame,
							selectedQuality,
						);
						rgba = result.rgba;
						gpuDurationMs = result.gpuDurationMs;
						backendDurationMs = result.backendDurationMs;
						timingSource = result.timingSource;
						width = result.width;
						height = result.height;
						gpuBytes =
							'memory' in result
								? (result as WithMemory).memory.gpuAllocatedBytes
								: (result as WithGpuBytes).gpuMemoryBytes;
					} else {
						const result = await processVideoFrameCuda(sourceFrame, selectedQuality);
						rgba = result.rgba;
						gpuDurationMs = result.gpuDurationMs;
						backendDurationMs = result.backendDurationMs;
						timingSource = result.timingSource;
						if (selectedQuality === '720p') {
							width = videoLayout.dstWidth720;
							height = videoLayout.dstHeight720;
						} else if (selectedQuality === '480p') {
							width = videoLayout.dstWidth480;
							height = videoLayout.dstHeight480;
						} else {
							width = videoLayout.dstWidth160;
							height = videoLayout.dstHeight160;
						}
						gpuBytes =
							'memory' in result
								? (result as WithMemory).memory.gpuAllocatedBytes
								: (result as WithGpuBytes).gpuMemoryBytes;
					}
				}

				let frameDataBase64: string;
				let format: 'rgba' | 'jpeg';
				const baseHostBytes = selectedQuality === '1080p' ? 0 : rgba.byteLength;
				if (selectedCompress) {
					const encoded = encodeJpeg({ data: rgba, width, height }, 80);
					const encodedBuffer = Buffer.from(encoded.data);
					frameDataBase64 = asBase64(encodedBuffer);
					format = 'jpeg';
					hostBytes = baseHostBytes + encodedBuffer.byteLength;
				} else {
					frameDataBase64 = asBase64(rgba);
					format = 'rgba';
					hostBytes = baseHostBytes;
				}

				const serverDurationMs = performance.now() - startedAt;
				const memory = {
					gpuBytes,
					hostBytes,
					serverRssBytes: process.memoryUsage().rss,
				};

				await new Promise<void>((resolve, reject) => {
					socket.send(
						JSON.stringify({
							type: 'frame',
							frameIndex,
							width,
							height,
							quality: selectedQuality,
							format,
							frameDataBase64,
							gpuDurationMs: Number(gpuDurationMs.toFixed(3)),
							backendDurationMs: Number(backendDurationMs.toFixed(3)),
							serverDurationMs: Number(serverDurationMs.toFixed(3)),
							memory,
							timingSource,
						}),
						(err?: Error) => {
							if (err) {
								reject(err);
								return;
							}
							resolve();
						},
					);
				});

				frameIndex = (frameIndex + 1) % frameCount;
			} catch (error) {
				sendError(
					'stream_failed',
					error instanceof Error ? error.message : 'Unknown stream error.',
				);
				isFrameInFlight = false;
				await stopStreaming();
				return;
			}

			isFrameInFlight = false;
			scheduleNextFrame();
		};

		socket.on('message', async (raw: Buffer) => {
			const text = raw.toString('utf8');
			let payload: unknown;

			try {
				payload = JSON.parse(text);
			} catch {
				sendError('invalid_json', 'Message must be valid JSON.');
				return;
			}

			const trySelect = SelectSchema.safeParse(payload);
			if (trySelect.success) {
				try {
					await stopStreaming();

					const hostLoadStart = performance.now();
					const video = await manager.getVideoBuffer(trySelect.data.fileName);
					const hostLoadTimeMs = performance.now() - hostLoadStart;

					if (video.byteLength % videoLayout.srcFrameBytes !== 0) {
						throw new Error('Video file size is not aligned to 1080p RGBA frame stride.');
					}

					frameCount = video.byteLength / videoLayout.srcFrameBytes;
					if (frameCount < 1) {
						throw new Error('Selected video has no frames.');
					}

					loadedVideo = video;
					selectedBackend = trySelect.data.backend;
					selectedQuality = trySelect.data.quality;
					selectedCompress = trySelect.data.compress;

					const gpuInitStart = performance.now();
					if (selectedBackend === 'webgpu') {
						webgpuPipeline = await initWebGpuVideoPipeline();
						pipelineGpuBytes = webgpuPipeline.gpuMemoryBytes;
					} else {
						const runtime = getCudaRuntimeState();
						if (!runtime.enabled) {
							throw new Error(`CUDA backend unavailable: ${runtime.reason}`);
						}

						const initResult = await initVideoPipelineCuda({
							srcWidth: videoLayout.srcWidth,
							srcHeight: videoLayout.srcHeight,
							dstWidth: videoLayout.dstWidth720,
							dstHeight: videoLayout.dstHeight720,
						});
						cudaPipelineReady = true;
						pipelineGpuBytes = initResult.memory.gpuAllocatedBytes;
					}
					const gpuInitTimeMs = performance.now() - gpuInitStart;

					const memory = {
						gpuBytes: pipelineGpuBytes,
						hostBytes: loadedVideo?.byteLength ?? 0,
						serverRssBytes: process.memoryUsage().rss,
					};

					frameIndex = 0;
					isStreaming = true;
					isPaused = false;

					socket.send(
						JSON.stringify({
							type: 'selected',
							fileName: trySelect.data.fileName,
							backend: selectedBackend,
							quality: selectedQuality,
							compress: selectedCompress,
							frameCount,
							hostLoadTimeMs: Number(hostLoadTimeMs.toFixed(3)),
							gpuInitTimeMs: Number(gpuInitTimeMs.toFixed(3)),
							memory,
						}),
					);

					scheduleNextFrame();
				} catch (error) {
					sendError(
						'select_failed',
						error instanceof Error ? error.message : 'Failed to initialize stream.',
					);
					await stopStreaming();
				}
				return;
			}

			const tryResize = ResizeSchema.safeParse(payload);
			if (tryResize.success) {
				selectedQuality = tryResize.data.quality;
				socket.send(
					JSON.stringify({
						type: 'resized',
						quality: selectedQuality,
					}),
				);
				return;
			}

			if (PauseSchema.safeParse(payload).success) {
				isPaused = true;
				socket.send(JSON.stringify({ type: 'paused' }));
				return;
			}

			if (ResumeSchema.safeParse(payload).success) {
				const shouldWake = isStreaming && isPaused;
				isPaused = false;
				socket.send(JSON.stringify({ type: 'resumed' }));
				if (shouldWake) {
					scheduleNextFrame();
				}
				return;
			}

			if (StopSchema.safeParse(payload).success) {
				await stopStreaming();
				socket.send(JSON.stringify({ type: 'stopped' }));
				return;
			}

			sendError('invalid_action', 'Supported actions: select, resize, pause, resume, stop.');
		});

		socket.on('close', () => {
			void stopStreaming();
		});
	});
}
