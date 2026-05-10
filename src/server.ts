import Fastify from 'fastify';
import cors from '@fastify/cors';
import swagger from '@fastify/swagger';
import swaggerUi from '@fastify/swagger-ui';
import websocket from '@fastify/websocket';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import './zod-extensions.js';
import { healthRoute } from './routes/health.js';
import { gpuInfoRoute } from './routes/gpu-info.js';
import { gpuStressRoute } from './routes/gpu-stress.js';
import { imageRoute } from './routes/image.js';
import { matrixRoute } from './routes/matrix.js';
import { ai } from './routes/ai.js';
import { videoRoute } from './routes/video.js';
import { renderRoute } from './routes/render.js';

const DEFAULT_REQUEST_TIMEOUT_MS = 60 * 60 * 1000;

function getRequestTimeoutMs(): number {
	const raw = process.env.SERVER_REQUEST_TIMEOUT_MS;
	if (!raw) return DEFAULT_REQUEST_TIMEOUT_MS;

	const parsed = Number(raw);
	if (!Number.isFinite(parsed) || parsed < 0) {
		console.warn(
			`[Server] Invalid SERVER_REQUEST_TIMEOUT_MS="${raw}". Falling back to ${DEFAULT_REQUEST_TIMEOUT_MS} ms.`,
		);
		return DEFAULT_REQUEST_TIMEOUT_MS;
	}

	return Math.floor(parsed);
}

export async function buildServer() {
	const requestTimeoutMs = getRequestTimeoutMs();

	const keyPath = path.resolve(process.cwd(), 'server.key');
	const certPath = path.resolve(process.cwd(), 'server.cert');
	const useHttps = existsSync(keyPath) && existsSync(certPath);

	const serverOptions = {
		requestTimeout: requestTimeoutMs,
		bodyLimit: 50 * 1024 * 1024,
		logger: {
			transport:
				process.env.NODE_ENV === 'development'
					? { target: 'pino-pretty', options: { colorize: true } }
					: undefined,
		},
		...(useHttps
			? {
					https: {
						key: readFileSync(keyPath),
						cert: readFileSync(certPath),
					},
				}
			: {}),
	};

	const server = Fastify(serverOptions);

	// ─── Plugins ────────────────────────────────────────────────────────────────
	await server.register(cors, {
		origin: true,
		methods: ['GET', 'POST', 'DELETE'],
	});

	await server.register(swagger, {
		openapi: {
			openapi: '3.0.0',
			info: {
				title: 'WebGPU Thesis API',
				description: 'REST API for WebGPU vs CUDA benchmark server',
				version: '0.1.0',
			},
			tags: [
				{ name: 'system', description: 'Server & GPU diagnostics' },
				{ name: 'image', description: 'Image processing (filters)' },
				{ name: 'matrix', description: 'Matrix operations' },
				{ name: 'ai', description: 'Stateful MLP inference pipeline' },
				{ name: 'video', description: 'Video streaming + dynamic downscaling benchmarks' },
				{ name: 'render', description: 'Procedural SDF scene rendering benchmarks' },
			],
		},
	});

	await server.register(websocket);

	await server.register(swaggerUi, {
		routePrefix: '/docs',
		uiConfig: { docExpansion: 'list' },
	});

	// ─── Routes ─────────────────────────────────────────────────────────────────
	await server.register(healthRoute);
	await server.register(gpuInfoRoute, { prefix: '/gpu' });
	await server.register(gpuStressRoute, { prefix: '/gpu/stress' });
	await server.register(imageRoute, { prefix: '/image' });
	await server.register(matrixRoute, { prefix: '/matrix' });
	await server.register(ai, { prefix: '/ai' });
	await server.register(videoRoute, { prefix: '/video' });
	await server.register(renderRoute, { prefix: '/render' });

	return server;
}
