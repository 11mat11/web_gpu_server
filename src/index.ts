import { buildServer } from './server.js'
import { warmupGpu } from './gpu/device.js'

const PORT = Number(process.env.PORT) || 3000
const HOST = process.env.HOST || '0.0.0.0'

async function main() {
  const server = await buildServer()

  try {
    await server.listen({ port: PORT, host: HOST })
    console.log(`\n🚀 Server running at http://${HOST}:${PORT}`)
    console.log(`📖 Swagger UI:   http://localhost:${PORT}/docs`)
    console.log(`❤️  Health:       http://localhost:${PORT}/health\n`)

    // Eagerly claim GPU so first request doesn't pay init cost
    await warmupGpu()
  } catch (err) {
    server.log.error(err)
    process.exit(1)
  }
}

main()