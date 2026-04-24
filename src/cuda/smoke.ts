import { multiplyMatrixCuda } from './cudaBackend.js'

async function main() {
  const size = Number(process.argv[2] ?? 32)
  if (!Number.isFinite(size) || size <= 0) {
    throw new Error('Size must be a positive number.')
  }

  const result = await multiplyMatrixCuda({
    size: Math.floor(size),
    inputMode: 'random',
    optimized: true,
    readback: true,
    randomMin: -1,
    randomMax: 1,
  })

  console.log(
    JSON.stringify(
      {
        size,
        timingSource: result.timingSource,
        generationDurationMs: Number((result.generationDurationMs ?? 0).toFixed(4)),
        multiplyDurationMs: Number(result.multiplyDurationMs.toFixed(4)),
        totalDurationMs: Number(result.totalDurationMs.toFixed(4)),
        outputLength: result.output?.length ?? 0,
      },
      null,
      2,
    ),
  )
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error)
  console.error(`[cuda-smoke] ${message}`)
  process.exit(1)
})

