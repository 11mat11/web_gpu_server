# WebGPU Thesis Server

REST API server for WebGPU vs CUDA performance analysis (Master's thesis).

## Requirements

- Node.js 22+ LTS
- GPU with Vulkan/Metal/DX12 support (for WebGPU headless)
- NVIDIA GPU + CUDA Toolkit (for CUDA comparison — later phase)

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Copy env file
cp .env.example .env

# 3. Start dev server (hot reload)
npm run dev
```

Server starts at: http://localhost:3000  
Swagger UI:       http://localhost:3000/docs

Request timeout (server-side) defaults to 1 hour via `SERVER_REQUEST_TIMEOUT_MS=3600000`.
Set `SERVER_REQUEST_TIMEOUT_MS=0` to disable request timeout completely.

## Project Structure

```
src/
├── index.ts              # Entry point
├── server.ts             # Fastify factory + plugins
├── routes/
│   ├── health.ts         # GET  /health
│   ├── gpu-info.ts       # GET  /gpu/info
│   ├── image.ts          # POST /image/filter
│   ├── matrix.ts         # POST /matrix/multiply
│   └── benchmark.ts      # POST /benchmark/start
│                         # GET  /benchmark/status/:id
│                         # GET  /benchmark/results
├── gpu/
│   ├── device.ts         # GPUAdapter/GPUDevice singleton
│   ├── shaders/
│   │   ├── matrixMul.wgsl
│   │   └── gaussianBlur.wgsl
│   ├── matrixMul.ts      # (TODO) WebGPU matrix pipeline
│   └── imageFilter.ts    # (TODO) WebGPU image pipeline
└── benchmarks/
    └── runner.ts         # (TODO) CLI benchmark runner
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Server health |
| GET | /gpu/info | GPU adapter info |
| POST | /image/filter | Run image filter |
| POST | /matrix/multiply | Run matrix multiply |
| POST | /benchmark/start | Start benchmark job |
| GET | /benchmark/status/:id | Poll job status |
| GET | /benchmark/results | All results |
| DELETE | /benchmark/results | Clear results |

## Matrix Multiply (`POST /matrix/multiply`)

- Supports `backend: webgpu` (default) and `backend: cpu`
- `inputMode: random` (default) generates both matrices using `randomMin`/`randomMax`
- `inputMode: custom` accepts `matrixA` and `matrixB` as NxN arrays
- Response returns `matrixC` plus timing metadata only when `size <= 100`
- `inputMode: random` i `inputMode: custom` na backendzie `webgpu` używają tego samego shadera `matrixMul.wgsl`
- Very large `inputMode: custom` requests fall back to CPU if the GPU buffer limits would be exceeded, so the result stays correct instead of failing halfway

Random matrices example:

```bash
curl -X POST http://localhost:3000/matrix/multiply \
  -H "Content-Type: application/json" \
  -d '{"size":256,"backend":"webgpu","inputMode":"random","randomMin":0,"randomMax":10}'
```

Custom 5x5 matrices example:

```bash
curl -X POST http://localhost:3000/matrix/multiply \
  -H "Content-Type: application/json" \
  -d '{
    "size":5,
    "backend":"webgpu",
    "inputMode":"custom",
    "matrixA":[
      [1,2,3,4,5],
      [6,7,8,9,10],
      [11,12,13,14,15],
      [16,17,18,19,20],
      [21,22,23,24,25]
    ],
    "matrixB":[
      [5,4,3,2,1],
      [1,2,3,4,5],
      [2,2,2,2,2],
      [0,1,0,1,0],
      [3,1,4,1,5]
    ]
  }'
```

Example response:

```json
{
  "backend": "webgpu",
  "size": 3,
  "inputMode": "custom",
  "durationMs": 0.123,
  "gflops": 0,
  "matrixC": [
    [30, 24, 18],
    [84, 69, 54],
    [138, 114, 90]
  ]
}
```

## Scripts

```bash
npm run dev      # Dev server with hot reload
npm run build    # Compile TypeScript
npm run start    # Run compiled server
npm run test     # Run unit tests
npm run bench    # Run CLI benchmarks
```
