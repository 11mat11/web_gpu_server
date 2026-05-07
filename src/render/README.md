# Render Scene

Minimal notes for the procedural SDF renderer.

## Endpoint

`POST /render`

Body example:

```
{
  "seed": 1234,
  "count": 2000,
  "backend": "webgpu-render"
}
```

Response fields: `imageBase64` (RGBA), `gpuTimeMs`, `serverTimeMs`, `gpuMemoryBytes`.

## Benchmark

```
npm run render:bench -- --backend=webgpu-render --seed=1234 --count=2000
```

