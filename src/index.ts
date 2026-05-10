import 'dotenv/config';
import https from 'node:https';
import { buildServer } from './server.js';
import { warmupGpu } from './gpu/device.js';

const PORT = Number(process.env.PORT) || 3000;
const HOST = process.env.HOST || '0.0.0.0';

async function main() {
	const server = await buildServer();

	const protocol = server.server instanceof https.Server ? 'https' : 'http';

	try {
		await server.listen({ port: PORT, host: HOST });
		console.log(`\n🚀 Server running at ${protocol}://${HOST}:${PORT}`);
		console.log(`📖 Swagger UI:   ${protocol}://localhost:${PORT}/docs`);
		console.log(`❤️  Health:       ${protocol}://localhost:${PORT}/health\n`);

		await warmupGpu();
	} catch (err) {
		server.log.error(err);
		process.exit(1);
	}
}

main().catch(console.error);
