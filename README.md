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

Environment variables are loaded from `.env` via `dotenv` in `src/index.ts`.

Request timeout (server-side) defaults to 1 hour via `SERVER_REQUEST_TIMEOUT_MS=3600000`.
Set `SERVER_REQUEST_TIMEOUT_MS=0` to disable request timeout completely.

CUDA runtime policy:
- `CUDA_ENABLED=auto` (default): CUDA is enabled only when NVIDIA GPU is detected and native addon exists.
- `CUDA_ENABLED=false`: hard-disable CUDA backend (no C++ addon loading).
- `CUDA_ENABLED=true`: allow CUDA path, but it still requires NVIDIA GPU + built addon.

When CUDA is unavailable, `POST /matrix/multiply` with `backend: "cuda"` returns `400 cuda_unavailable`.

CUDA is optional. If you only test WebGPU/CPU, you do not need to build the native C++ addon.
Leave `CUDA_ENABLED=auto` or set `CUDA_ENABLED=false` to completely skip CUDA runtime paths.
The CUDA addon is only required for `backend: "cuda"` calls and for CUDA-specific endpoints.

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
| GET | /ai/status | AI status for models (`mlp`, `cnn`) and backends |
| POST | /ai/load | Load selected model (`mlp`/`cnn`) or both |
| POST | /ai/predict/mlp | Run MLP inference on `webgpu` or `cuda` |
| POST | /ai/predict/cnn | Run Mini-VGG CNN inference on `webgpu` or `cuda` |
| POST | /ai/unload | Unload selected model (`mlp`/`cnn`) or both |
| POST | /render | Procedural SDF scene renderer (webgpu-render/webgpu-compute/cuda) |
| GET | /benchmark/status/:id | Poll job status |
| GET | /benchmark/results | All results |
| DELETE | /benchmark/results | Clear results |

## Matrix Multiply (`POST /matrix/multiply`)

- Supports `backend: webgpu` (default) and `backend: cpu`
- `inputMode: random` (default) generates both matrices using `randomMin`/`randomMax`
- `inputMode: custom` accepts `matrixA` and `matrixB` as NxN arrays
- `optimized: true` enables tiled multiplication (`matrixMulTiled.wgsl`), `optimized: false` uses naive multiplication (`matrixMul.wgsl`)
- Response returns `matrixC` plus timing metadata only when `size <= 100`
- Very large `inputMode: custom` requests fall back to CPU if the GPU buffer limits would be exceeded, so the result stays correct instead of failing halfway

Random matrices example (tiled):

```bash
curl -X POST http://localhost:3000/matrix/multiply \
  -H "Content-Type: application/json" \
  -d '{"size":256,"backend":"webgpu","inputMode":"random","optimized":true,"randomMin":0,"randomMax":10}'
```

Random matrices example (naive):

```bash
curl -X POST http://localhost:3000/matrix/multiply \
  -H "Content-Type: application/json" \
  -d '{"size":256,"backend":"webgpu","inputMode":"random","optimized":false,"randomMin":0,"randomMax":10}'
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

Response includes measured `processMemory` (`before`, `after`) from `process.memoryUsage()` for the exact request, instead of synthetic memory estimates.

## AI Multi-Model Pipeline

`POST /ai/load` accepts optional selector:

```json
{
  "model": "cnn",
  "webgpu": true,
  "cuda": false
}
```

- `model`: optional (`"mlp"` or `"cnn"`); when omitted, server tries to load both models
- `webgpu`/`cuda`: optional backend flags; when omitted, server tries both backends

Wagi:

- MLP: `src/ai/mega_mnist_weights.bin`
- CNN: `src/ai/cifar10_mini_vgg_weights.bin`

`POST /ai/predict/mlp` expects:

```json
{
  "backend": "cuda",
  "input": [0.0, 0.1, 0.2]
}
```

Powyższa tablica `input` jest skrócona tylko poglądowo.

- `backend`: `"cuda"` or `"webgpu"`
- `input`: exactly `16384` float values
- response includes: `prediction`, `probabilities`, `gpuDurationMs`, `totalDurationMs`, `timingSource`

`POST /ai/predict/cnn` expects `input` with exactly `49152` float values (`128x128x3`, CHW).

CNN response includes additionally:

- `predictionLabel` (`airplane`, `automobile`, ..., `truck`)
- `memoryEstimate` for CNN model instance

`GET /ai/status` returns current lifecycle state, loaded models, backend status and memory breakdown per model.

`POST /ai/unload` accepts the same selector as `/ai/load` and releases only selected model/backend resources.

## Scripts

```bash
npm run dev      # Dev server with hot reload
npm run build    # Compile TypeScript
npm run start    # Run compiled server
npm run test     # Run unit tests
npm run bench    # Run CLI benchmarks
npm run render:bench # Run render scene benchmark
```
