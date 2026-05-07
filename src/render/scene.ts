const LCG_A = 1664525
const LCG_C = 1013904223
const LCG_M = 4294967296

export const RENDER_WIDTH = 1920
export const RENDER_HEIGHT = 1080
export const SHAPE_FLOATS = 12

export type ShapeType = 0 | 1 | 2 | 3

function createLcg(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (Math.imul(state, LCG_A) + LCG_C) >>> 0
    return state
  }
}

function nextFloat(nextU32: () => number): number {
  return nextU32() / LCG_M
}

export function generateScene(seed: number, count: number): Float32Array {
  const safeCount = Math.max(0, Math.floor(count))
  const out = new Float32Array(safeCount * SHAPE_FLOATS)
  const nextU32 = createLcg(seed)

  const minSize = 14
  const maxSize = 160

  for (let i = 0; i < safeCount; i++) {
    const type = Math.floor(nextFloat(nextU32) * 3) as ShapeType
    const size = minSize + nextFloat(nextU32) * (maxSize - minSize)

    const spanX = Math.max(0, RENDER_WIDTH - 2 * size)
    const spanY = Math.max(0, RENDER_HEIGHT - 2 * size)

    const x = size + nextFloat(nextU32) * spanX
    const y = size + nextFloat(nextU32) * spanY
    const depth = nextFloat(nextU32)

    const r = nextFloat(nextU32)
    const g = nextFloat(nextU32)
    const b = nextFloat(nextU32)
    const a = 1.0

    const base = i * SHAPE_FLOATS
    out[base + 0] = type
    out[base + 1] = x
    out[base + 2] = y
    out[base + 3] = size
    out[base + 4] = depth
    out[base + 5] = r
    out[base + 6] = g
    out[base + 7] = b
    out[base + 8] = a
    out[base + 9] = 0.0
    out[base + 10] = 0.0
    out[base + 11] = 0.0
  }

  return out
}

export function getShapeBufferBytes(count: number): number {
  return Math.max(0, Math.floor(count)) * SHAPE_FLOATS * Float32Array.BYTES_PER_ELEMENT
}

