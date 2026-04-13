#include "matrix_kernels.h"

#include <cstdint>

namespace {

constexpr int kRandomFillBlockSize = 256;
constexpr int kNaiveBlockSize = 16;
constexpr int kTiledBlockSize = 16;
constexpr int kTiledSize = 32;

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
  uint32_t v = x * 747796405u + 2891336453u;
  v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  v = (v >> 22u) ^ v;
  return v;
}

__device__ __forceinline__ float random01(uint32_t state) {
  return static_cast<float>(hash32(state)) / 4294967295.0f;
}

__global__ void randomFillKernel(
  float* out,
  uint32_t count,
  uint32_t seed,
  float minValue,
  float maxValue,
  uint32_t salt
) {
  const uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  const float span = maxValue - minValue;

  for (uint32_t idx = globalId; idx < count; idx += stride) {
    const uint32_t baseSeed = seed ^ idx;
    out[idx] = minValue + random01(baseSeed ^ salt) * span;
  }
}

__global__ void matrixMulNaiveKernel(const float* __restrict__ matrixA, const float* __restrict__ matrixB, float* __restrict__ matrixC, int size) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= size || col >= size) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < size; ++k) {
    sum += matrixA[row * size + k] * matrixB[k * size + col];
  }

  matrixC[row * size + col] = sum;
}

__global__ void matrixMulTiledKernel(const float* __restrict__ matrixA, const float* __restrict__ matrixB, float* __restrict__ matrixC, int size) {
  __shared__ float tileA[kTiledSize][kTiledSize + 1];
  __shared__ float tileB[kTiledSize][kTiledSize + 1];

  const int localRow = threadIdx.y * 2;
  const int localCol = threadIdx.x * 2;

  const int row = blockIdx.y * kTiledSize + localRow;
  const int col = blockIdx.x * kTiledSize + localCol;

  float sum00 = 0.0f;
  float sum01 = 0.0f;
  float sum10 = 0.0f;
  float sum11 = 0.0f;

  const int row1 = row + 1;
  const int col1 = col + 1;
  const int tileCount = (size + kTiledSize - 1) / kTiledSize;

  for (int tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
    const int aBaseCol = tileIndex * kTiledSize + localCol;
    const int bBaseRow = tileIndex * kTiledSize + localRow;

    const int aCol0 = aBaseCol;
    const int aCol1 = aBaseCol + 1;
    const int bRow0 = bBaseRow;
    const int bRow1 = bBaseRow + 1;

    tileA[localRow][localCol] = (row < size && aCol0 < size) ? matrixA[row * size + aCol0] : 0.0f;
    tileA[localRow][localCol + 1] = (row < size && aCol1 < size) ? matrixA[row * size + aCol1] : 0.0f;
    tileA[localRow + 1][localCol] = (row1 < size && aCol0 < size) ? matrixA[row1 * size + aCol0] : 0.0f;
    tileA[localRow + 1][localCol + 1] = (row1 < size && aCol1 < size) ? matrixA[row1 * size + aCol1] : 0.0f;

    tileB[localRow][localCol] = (bRow0 < size && col < size) ? matrixB[bRow0 * size + col] : 0.0f;
    tileB[localRow][localCol + 1] = (bRow0 < size && col1 < size) ? matrixB[bRow0 * size + col1] : 0.0f;
    tileB[localRow + 1][localCol] = (bRow1 < size && col < size) ? matrixB[bRow1 * size + col] : 0.0f;
    tileB[localRow + 1][localCol + 1] = (bRow1 < size && col1 < size) ? matrixB[bRow1 * size + col1] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < kTiledSize; ++k) {
      const float a0 = tileA[localRow][k];
      const float a1 = tileA[localRow + 1][k];
      const float b0 = tileB[k][localCol];
      const float b1 = tileB[k][localCol + 1];
      sum00 += a0 * b0;
      sum01 += a0 * b1;
      sum10 += a1 * b0;
      sum11 += a1 * b1;
    }

    __syncthreads();
  }

  if (row < size && col < size) matrixC[row * size + col] = sum00;
  if (row < size && col1 < size) matrixC[row * size + col1] = sum01;
  if (row1 < size && col < size) matrixC[row1 * size + col] = sum10;
  if (row1 < size && col1 < size) matrixC[row1 * size + col1] = sum11;
}

} // namespace

void launchRandomFillKernel(
  float* out,
  int size,
  unsigned int seed,
  float minValue,
  float maxValue,
  unsigned int salt,
  cudaStream_t stream
) {
  const uint32_t count = static_cast<uint32_t>(size) * static_cast<uint32_t>(size);
  const int blocks = static_cast<int>((count + kRandomFillBlockSize - 1) / kRandomFillBlockSize);
  randomFillKernel<<<blocks, kRandomFillBlockSize, 0, stream>>>(out, count, seed, minValue, maxValue, salt);
}

void launchMatrixMulNaiveKernel(
  const float* matrixA,
  const float* matrixB,
  float* matrixC,
  int size,
  cudaStream_t stream
) {
  const dim3 block(kNaiveBlockSize, kNaiveBlockSize);
  const dim3 grid((size + kNaiveBlockSize - 1) / kNaiveBlockSize, (size + kNaiveBlockSize - 1) / kNaiveBlockSize);
  matrixMulNaiveKernel<<<grid, block, 0, stream>>>(matrixA, matrixB, matrixC, size);
}

void launchMatrixMulTiledKernel(
  const float* matrixA,
  const float* matrixB,
  float* matrixC,
  int size,
  cudaStream_t stream
) {
  const dim3 block(kTiledBlockSize, kTiledBlockSize);
  const dim3 grid((size + kTiledSize - 1) / kTiledSize, (size + kTiledSize - 1) / kTiledSize);
  matrixMulTiledKernel<<<grid, block, 0, stream>>>(matrixA, matrixB, matrixC, size);
}

