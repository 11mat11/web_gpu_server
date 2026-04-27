#include "matrix_kernels.h"

#include <cstdint>

namespace {

constexpr int kRandomFillBlockSize = 256;
constexpr int kNaiveBlockSize = 16;
constexpr int kTiledBlockSize = 16;
constexpr int kTiledSize = 32;
constexpr int kMlpGemvBlockSize = 256;
constexpr int kMlpGemvTileSize = 256;
constexpr int kCnnConvBlockSize = 16;
constexpr int kCnnPoolBlockSize = 16;
constexpr int kVideoBlockSize = 16;
constexpr int kVideoHistogramBlockSize = 256;

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

__global__ void mlpGemvKernel(
  const float* __restrict__ inputVector,
  const float* __restrict__ weights,
  const float* __restrict__ bias,
  float* __restrict__ outputVector,
  int inputSize,
  int outputSize,
  int applyRelu
) {
  __shared__ float inputTile[kMlpGemvTileSize];
  __shared__ float partialSums[kMlpGemvBlockSize];

  const int outputIndex = static_cast<int>(blockIdx.x);
  const int localIndex = static_cast<int>(threadIdx.x);

  if (outputIndex >= outputSize) {
    return;
  }

  float localSum = 0.0f;

  for (int base = 0; base < inputSize; base += kMlpGemvTileSize) {
    const int inputIndex = base + localIndex;
    inputTile[localIndex] = (inputIndex < inputSize) ? inputVector[inputIndex] : 0.0f;
    __syncthreads();

    const int tileLength = min(kMlpGemvTileSize, inputSize - base);
    for (int k = localIndex; k < tileLength; k += kMlpGemvBlockSize) {
      localSum += inputTile[k] * weights[(base + k) * outputSize + outputIndex];
    }
    __syncthreads();
  }

  partialSums[localIndex] = localSum;
  __syncthreads();

  for (int stride = kMlpGemvBlockSize / 2; stride > 0; stride >>= 1) {
    if (localIndex < stride) {
      partialSums[localIndex] += partialSums[localIndex + stride];
    }
    __syncthreads();
  }

  if (localIndex == 0) {
    float value = partialSums[0] + bias[outputIndex];
    if (applyRelu != 0 && value < 0.0f) {
      value = 0.0f;
    }
    outputVector[outputIndex] = value;
  }
}

__global__ void cnnConv2dKernel(
  const float* __restrict__ input,
  const float* __restrict__ weights,
  const float* __restrict__ bias,
  float* __restrict__ output,
  int inChannels,
  int outChannels,
  int inHeight,
  int inWidth,
  int applyRelu
) {
  const int outX = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int outY = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int outChannel = static_cast<int>(blockIdx.z);

  if (outX >= inWidth || outY >= inHeight || outChannel >= outChannels) {
    return;
  }

  float sum = bias[outChannel];

  for (int inChannel = 0; inChannel < inChannels; ++inChannel) {
    for (int ky = 0; ky < 3; ++ky) {
      const int inY = outY + ky - 1;
      if (inY < 0 || inY >= inHeight) {
        continue;
      }

      for (int kx = 0; kx < 3; ++kx) {
        const int inX = outX + kx - 1;
        if (inX < 0 || inX >= inWidth) {
          continue;
        }

        const int inputIndex = (inChannel * inHeight + inY) * inWidth + inX;
        const int weightIndex = (((outChannel * inChannels) + inChannel) * 3 + ky) * 3 + kx;
        sum += input[inputIndex] * weights[weightIndex];
      }
    }
  }

  if (applyRelu != 0 && sum < 0.0f) {
    sum = 0.0f;
  }

  const int outputIndex = (outChannel * inHeight + outY) * inWidth + outX;
  output[outputIndex] = sum;
}

__global__ void cnnMaxPool2x2Kernel(
  const float* __restrict__ input,
  float* __restrict__ output,
  int channels,
  int inHeight,
  int inWidth
) {
  const int outWidth = inWidth / 2;
  const int outHeight = inHeight / 2;

  const int outX = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int outY = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int channel = static_cast<int>(blockIdx.z);

  if (outX >= outWidth || outY >= outHeight || channel >= channels) {
    return;
  }

  const int inBaseX = outX * 2;
  const int inBaseY = outY * 2;

  float maxValue = -3.402823466e+38F;
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int inX = inBaseX + dx;
      const int inY = inBaseY + dy;
      const int inIndex = (channel * inHeight + inY) * inWidth + inX;
      maxValue = fmaxf(maxValue, input[inIndex]);
    }
  }

  const int outIndex = (channel * outHeight + outY) * outWidth + outX;
  output[outIndex] = maxValue;
}

__global__ void videoBilinearDownscaleKernel(
  const uchar4* __restrict__ input,
  uchar4* __restrict__ output,
  int srcWidth,
  int srcHeight,
  int dstWidth,
  int dstHeight
) {
  const int outX = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int outY = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

  if (outX >= dstWidth || outY >= dstHeight) {
    return;
  }

  const float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
  const float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);

  const float srcX = (static_cast<float>(outX) + 0.5f) * scaleX - 0.5f;
  const float srcY = (static_cast<float>(outY) + 0.5f) * scaleY - 0.5f;

  int x0 = static_cast<int>(floorf(srcX));
  int y0 = static_cast<int>(floorf(srcY));
  x0 = max(0, min(x0, srcWidth - 1));
  y0 = max(0, min(y0, srcHeight - 1));
  const int x1 = min(x0 + 1, srcWidth - 1);
  const int y1 = min(y0 + 1, srcHeight - 1);

  const float fx = srcX - static_cast<float>(x0);
  const float fy = srcY - static_cast<float>(y0);

  const uchar4 p00 = input[y0 * srcWidth + x0];
  const uchar4 p10 = input[y0 * srcWidth + x1];
  const uchar4 p01 = input[y1 * srcWidth + x0];
  const uchar4 p11 = input[y1 * srcWidth + x1];

  const float invFx = 1.0f - fx;
  const float invFy = 1.0f - fy;

  const float4 top = make_float4(
    static_cast<float>(p00.x) * invFx + static_cast<float>(p10.x) * fx,
    static_cast<float>(p00.y) * invFx + static_cast<float>(p10.y) * fx,
    static_cast<float>(p00.z) * invFx + static_cast<float>(p10.z) * fx,
    static_cast<float>(p00.w) * invFx + static_cast<float>(p10.w) * fx
  );

  const float4 bottom = make_float4(
    static_cast<float>(p01.x) * invFx + static_cast<float>(p11.x) * fx,
    static_cast<float>(p01.y) * invFx + static_cast<float>(p11.y) * fx,
    static_cast<float>(p01.z) * invFx + static_cast<float>(p11.z) * fx,
    static_cast<float>(p01.w) * invFx + static_cast<float>(p11.w) * fx
  );

  const float4 out = make_float4(
    top.x * invFy + bottom.x * fy,
    top.y * invFy + bottom.y * fy,
    top.z * invFy + bottom.z * fy,
    top.w * invFy + bottom.w * fy
  );

  output[outY * dstWidth + outX] = make_uchar4(
    static_cast<unsigned char>(fminf(fmaxf(out.x, 0.0f), 255.0f)),
    static_cast<unsigned char>(fminf(fmaxf(out.y, 0.0f), 255.0f)),
    static_cast<unsigned char>(fminf(fmaxf(out.z, 0.0f), 255.0f)),
    static_cast<unsigned char>(fminf(fmaxf(out.w, 0.0f), 255.0f))
  );
}

__global__ void videoRgbHistogramKernel(
  const uchar4* __restrict__ input,
  unsigned int* __restrict__ histogram,
  int pixelCount
) {
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= pixelCount) {
    return;
  }

  const uchar4 pixel = input[idx];
  atomicAdd(&histogram[static_cast<unsigned int>(pixel.x)], 1U);
  atomicAdd(&histogram[256U + static_cast<unsigned int>(pixel.y)], 1U);
  atomicAdd(&histogram[512U + static_cast<unsigned int>(pixel.z)], 1U);
}

}

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

void launchMlpGemvKernel(
  const float* inputVector,
  const float* weights,
  const float* bias,
  float* outputVector,
  int inputSize,
  int outputSize,
  bool applyRelu,
  cudaStream_t stream
) {
  const dim3 block(kMlpGemvBlockSize);
  const dim3 grid(static_cast<unsigned int>(outputSize));
  mlpGemvKernel<<<grid, block, 0, stream>>>(
    inputVector,
    weights,
    bias,
    outputVector,
    inputSize,
    outputSize,
    applyRelu ? 1 : 0
  );
}

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
) {
  const dim3 block(kCnnConvBlockSize, kCnnConvBlockSize);
  const dim3 grid(
    static_cast<unsigned int>((inWidth + kCnnConvBlockSize - 1) / kCnnConvBlockSize),
    static_cast<unsigned int>((inHeight + kCnnConvBlockSize - 1) / kCnnConvBlockSize),
    static_cast<unsigned int>(outChannels)
  );

  cnnConv2dKernel<<<grid, block, 0, stream>>>(
    input,
    weights,
    bias,
    output,
    inChannels,
    outChannels,
    inHeight,
    inWidth,
    applyRelu ? 1 : 0
  );
}

void launchCnnMaxPool2x2Kernel(
  const float* input,
  float* output,
  int channels,
  int inHeight,
  int inWidth,
  cudaStream_t stream
) {
  const int outHeight = inHeight / 2;
  const int outWidth = inWidth / 2;
  const dim3 block(kCnnPoolBlockSize, kCnnPoolBlockSize);
  const dim3 grid(
    static_cast<unsigned int>((outWidth + kCnnPoolBlockSize - 1) / kCnnPoolBlockSize),
    static_cast<unsigned int>((outHeight + kCnnPoolBlockSize - 1) / kCnnPoolBlockSize),
    static_cast<unsigned int>(channels)
  );

  cnnMaxPool2x2Kernel<<<grid, block, 0, stream>>>(input, output, channels, inHeight, inWidth);
}

void launchVideoBilinearDownscaleKernel(
  const unsigned char* inputRgba,
  unsigned char* outputRgba,
  int srcWidth,
  int srcHeight,
  int dstWidth,
  int dstHeight,
  cudaStream_t stream
) {
  const dim3 block(kVideoBlockSize, kVideoBlockSize);
  const dim3 grid(
    static_cast<unsigned int>((dstWidth + kVideoBlockSize - 1) / kVideoBlockSize),
    static_cast<unsigned int>((dstHeight + kVideoBlockSize - 1) / kVideoBlockSize)
  );

  videoBilinearDownscaleKernel<<<grid, block, 0, stream>>>(
    reinterpret_cast<const uchar4*>(inputRgba),
    reinterpret_cast<uchar4*>(outputRgba),
    srcWidth,
    srcHeight,
    dstWidth,
    dstHeight
  );
}

void launchVideoRgbHistogramKernel(
  const unsigned char* inputRgba,
  unsigned int* histogram,
  int pixelCount,
  cudaStream_t stream
) {
  const int blocks = (pixelCount + kVideoHistogramBlockSize - 1) / kVideoHistogramBlockSize;
  videoRgbHistogramKernel<<<blocks, kVideoHistogramBlockSize, 0, stream>>>(
    reinterpret_cast<const uchar4*>(inputRgba),
    histogram,
    pixelCount
  );
}

