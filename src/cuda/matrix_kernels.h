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

void launchMlpGemvKernel(
  const float* inputVector,
  const float* weights,
  const float* bias,
  float* outputVector,
  int inputSize,
  int outputSize,
  bool applyRelu,
  cudaStream_t stream
);

void launchCnnConv2dKernel(
  const float* input,
  const float* weights,
  const float* bias,
  float* output,
  int inChannels,
  int outChannels,
  int inHeight,
  int inWidth,
  bool applyRelu,
  cudaStream_t stream
);

void launchCnnMaxPool2x2Kernel(
  const float* input,
  float* output,
  int channels,
  int inHeight,
  int inWidth,
  cudaStream_t stream
);

void launchVideoBilinearDownscaleKernel(
  const unsigned char* inputRgba,
  unsigned char* outputRgba,
  int srcWidth,
  int srcHeight,
  int dstWidth,
  int dstHeight,
  cudaStream_t stream
);

void launchVideoRgbHistogramKernel(
  const unsigned char* inputRgba,
  unsigned int* histogram,
  int pixelCount,
  cudaStream_t stream
);

void launchRenderSceneKernel(
  const float* shapes,
  int shapeCount,
  unsigned char* outputRgba,
  int width,
  int height,
  cudaStream_t stream
);
