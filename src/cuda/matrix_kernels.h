#pragma once

#include <cuda_runtime.h>

void launchRandomFillKernel(
  float* out,
  int size,
  unsigned int seed,
  float minValue,
  float maxValue,
  unsigned int salt,
  cudaStream_t stream
);

void launchMatrixMulNaiveKernel(
  const float* matrixA,
  const float* matrixB,
  float* matrixC,
  int size,
  cudaStream_t stream
);

void launchMatrixMulTiledKernel(
  const float* matrixA,
  const float* matrixB,
  float* matrixC,
  int size,
  cudaStream_t stream
);

