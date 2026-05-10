#include <cstdint>

#include "matrix_kernels.h"

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
constexpr int kRenderBlockSize = 16;
constexpr int kGaussianBlockSize = 16;

__device__ __forceinline__ float gaussianWeight(int index) {
    constexpr float kWeights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    return kWeights[index];
}

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    uint32_t v = x * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    v = (v >> 22u) ^ v;
    return v;
}

__device__ __forceinline__ float random01(uint32_t state) {
    return static_cast<float>(hash32(state)) / 4294967295.0f;
}

__global__ void randomFillKernel(float* out, uint32_t count, uint32_t seed, float minValue,
				 float maxValue, uint32_t salt) {
    const uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const float span = maxValue - minValue;

    for (uint32_t idx = globalId; idx < count; idx += stride) {
	const uint32_t baseSeed = seed ^ idx;
	out[idx] = minValue + random01(baseSeed ^ salt) * span;
    }
}

__global__ void matrixMulNaiveKernel(const float* __restrict__ matrixA,
				     const float* __restrict__ matrixB, float* __restrict__ matrixC,
				     int size) {
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size || col >= size) {
	return;
    }

    float sum = 0.0f;
    for (int k = 0; k < size; ++k) {
	sum += matrixA[row * size + k] * matrixB[k * size + col];
    }

    matrixC[row * size + col] = sum;
}

__global__ void matrixMulTiledKernel(const float* __restrict__ matrixA,
				     const float* __restrict__ matrixB, float* __restrict__ matrixC,
				     int size) {
    __shared__ float tileA[kTiledSize][kTiledSize + 1];
    __shared__ float tileB[kTiledSize][kTiledSize + 1];

    const uint32_t localRow = threadIdx.y * 2;
    const uint32_t localCol = threadIdx.x * 2;

    const uint32_t row = blockIdx.y * kTiledSize + localRow;
    const uint32_t col = blockIdx.x * kTiledSize + localCol;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    const uint32_t row1 = row + 1;
    const uint32_t col1 = col + 1;
    const int tileCount = (size + kTiledSize - 1) / kTiledSize;

    for (int tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
	const uint32_t aBaseCol = tileIndex * kTiledSize + localCol;
	const uint32_t bBaseRow = tileIndex * kTiledSize + localRow;

	const uint32_t aCol0 = aBaseCol;
	const uint32_t aCol1 = aBaseCol + 1;
	const uint32_t bRow0 = bBaseRow;
	const uint32_t bRow1 = bBaseRow + 1;

	tileA[localRow][localCol] =
	    (row < size && aCol0 < size) ? matrixA[row * size + aCol0] : 0.0f;
	tileA[localRow][localCol + 1] =
	    (row < size && aCol1 < size) ? matrixA[row * size + aCol1] : 0.0f;
	tileA[localRow + 1][localCol] =
	    (row1 < size && aCol0 < size) ? matrixA[row1 * size + aCol0] : 0.0f;
	tileA[localRow + 1][localCol + 1] =
	    (row1 < size && aCol1 < size) ? matrixA[row1 * size + aCol1] : 0.0f;

	tileB[localRow][localCol] =
	    (bRow0 < size && col < size) ? matrixB[bRow0 * size + col] : 0.0f;
	tileB[localRow][localCol + 1] =
	    (bRow0 < size && col1 < size) ? matrixB[bRow0 * size + col1] : 0.0f;
	tileB[localRow + 1][localCol] =
	    (bRow1 < size && col < size) ? matrixB[bRow1 * size + col] : 0.0f;
	tileB[localRow + 1][localCol + 1] =
	    (bRow1 < size && col1 < size) ? matrixB[bRow1 * size + col1] : 0.0f;

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

    if (row < size && col < size)
	matrixC[row * size + col] = sum00;
    if (row < size && col1 < size)
	matrixC[row * size + col1] = sum01;
    if (row1 < size && col < size)
	matrixC[row1 * size + col] = sum10;
    if (row1 < size && col1 < size)
	matrixC[row1 * size + col1] = sum11;
}

__global__ void mlpGemvKernel(const float* __restrict__ inputVector,
			      const float* __restrict__ weights, const float* __restrict__ bias,
			      float* __restrict__ outputVector, int inputSize, int outputSize,
			      int applyRelu) {
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

__global__ void cnnConv2dKernel(const float* __restrict__ input, const float* __restrict__ weights,
				const float* __restrict__ bias, float* __restrict__ output,
				int inChannels, int outChannels, int inHeight, int inWidth,
				int applyRelu) {
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

__global__ void cnnMaxPool2x2Kernel(const float* __restrict__ input, float* __restrict__ output,
				    int channels, int inHeight, int inWidth) {
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

__global__ void videoBilinearDownscaleKernel(const uchar4* __restrict__ input,
					     uchar4* __restrict__ output, int srcWidth,
					     int srcHeight, int dstWidth, int dstHeight) {
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

    const float4 top =
	make_float4(static_cast<float>(p00.x) * invFx + static_cast<float>(p10.x) * fx,
		    static_cast<float>(p00.y) * invFx + static_cast<float>(p10.y) * fx,
		    static_cast<float>(p00.z) * invFx + static_cast<float>(p10.z) * fx,
		    static_cast<float>(p00.w) * invFx + static_cast<float>(p10.w) * fx);

    const float4 bottom =
	make_float4(static_cast<float>(p01.x) * invFx + static_cast<float>(p11.x) * fx,
		    static_cast<float>(p01.y) * invFx + static_cast<float>(p11.y) * fx,
		    static_cast<float>(p01.z) * invFx + static_cast<float>(p11.z) * fx,
		    static_cast<float>(p01.w) * invFx + static_cast<float>(p11.w) * fx);

    const float4 out = make_float4(top.x * invFy + bottom.x * fy, top.y * invFy + bottom.y * fy,
				   top.z * invFy + bottom.z * fy, top.w * invFy + bottom.w * fy);

    output[outY * dstWidth + outX] =
	make_uchar4(static_cast<unsigned char>(fminf(fmaxf(roundf(out.x), 0.0f), 255.0f)),
		    static_cast<unsigned char>(fminf(fmaxf(roundf(out.y), 0.0f), 255.0f)),
		    static_cast<unsigned char>(fminf(fmaxf(roundf(out.z), 0.0f), 255.0f)),
		    static_cast<unsigned char>(fminf(fmaxf(roundf(out.w), 0.0f), 255.0f)));
}

__global__ void videoRgbHistogramKernel(const uchar4* __restrict__ input,
					unsigned int* __restrict__ histogram, int pixelCount) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= pixelCount) {
	return;
    }

    const uchar4 pixel = input[idx];
    atomicAdd(&histogram[static_cast<unsigned int>(pixel.x)], 1U);
    atomicAdd(&histogram[256U + static_cast<unsigned int>(pixel.y)], 1U);
    atomicAdd(&histogram[512U + static_cast<unsigned int>(pixel.z)], 1U);
}

__global__ void gaussianBlurHorizontalKernel(const unsigned int* __restrict__ input,
					     unsigned int* __restrict__ output, int width,
					     int height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= width || y >= height) {
	return;
    }

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = -2; k <= 2; ++k) {
	const int sx = max(0, min(x + k, width - 1));
	const unsigned int pixel = input[y * width + sx];
	const float w = gaussianWeight(k + 2);

	acc.x += static_cast<float>(pixel & 0xFFu) * w;
	acc.y += static_cast<float>((pixel >> 8u) & 0xFFu) * w;
	acc.z += static_cast<float>((pixel >> 16u) & 0xFFu) * w;
	acc.w += static_cast<float>((pixel >> 24u) & 0xFFu) * w;
    }

    auto r = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.x), 0.0f), 255.0f));
    auto g = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.y), 0.0f), 255.0f));
    auto b = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.z), 0.0f), 255.0f));
    auto a = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.w), 0.0f), 255.0f));

    output[y * width + x] = r | (g << 8u) | (b << 16u) | (a << 24u);
}

__global__ void gaussianBlurVerticalKernel(const unsigned int* __restrict__ input,
					   unsigned int* __restrict__ output, int width,
					   int height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= width || y >= height) {
	return;
    }

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = -2; k <= 2; ++k) {
	const int sy = max(0, min(y + k, height - 1));
	const unsigned int pixel = input[sy * width + x];
	const float w = gaussianWeight(k + 2);

	acc.x += static_cast<float>(pixel & 0xFFu) * w;
	acc.y += static_cast<float>((pixel >> 8u) & 0xFFu) * w;
	acc.z += static_cast<float>((pixel >> 16u) & 0xFFu) * w;
	acc.w += static_cast<float>((pixel >> 24u) & 0xFFu) * w;
    }

    auto r = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.x), 0.0f), 255.0f));
    auto g = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.y), 0.0f), 255.0f));
    auto b = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.z), 0.0f), 255.0f));
    auto a = static_cast<unsigned int>(fminf(fmaxf(roundf(acc.w), 0.0f), 255.0f));

    output[y * width + x] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
__device__ __forceinline__ float2 make_f2(float x, float y) { return make_float2(x, y); }

__device__ __forceinline__ float2 sub2(float2 a, float2 b) { return make_f2(a.x - b.x, a.y - b.y); }

__device__ __forceinline__ float2 mul2(float2 a, float b) { return make_f2(a.x * b, a.y * b); }

__device__ __forceinline__ float2 abs2(float2 v) { return make_f2(fabsf(v.x), fabsf(v.y)); }

__device__ __forceinline__ float length2(float2 v) { return sqrtf(v.x * v.x + v.y * v.y); }

__device__ __forceinline__ float signf_fast(float v) {
    return static_cast<float>((v > 0.0f) - (v < 0.0f));
}

__device__ float sdfCircle(float2 p, float r) { return length2(p) - r; }

__device__ float sdfBox(float2 p, float r) {
    const float2 d = sub2(abs2(p), make_f2(r, r));
    const float2 dmax = make_f2(fmaxf(d.x, 0.0f), fmaxf(d.y, 0.0f));
    const float outside = length2(dmax);
    const float inside = fminf(fmaxf(d.x, d.y), 0.0f);
    return outside + inside;
}

__device__ float sdfTriangle(float2 pIn, float r) {
    constexpr float k = 1.7320508f;
    float2 p = pIn;
    p.x = fabsf(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0f) {
	p = mul2(make_f2(p.x - k * p.y, -k * p.x - p.y), 0.5f);
    }
    p.x = p.x - fminf(fmaxf(p.x, -2.0f * r), 0.0f);
    return -length2(p) * signf_fast(p.y);
}

__global__ void renderSceneKernel(const float* __restrict__ shapes, int shapeCount,
				  uchar4* __restrict__ output, int width, int height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= width || y >= height) {
	return;
    }

    const float pixelX = static_cast<float>(x) + 0.5f;
    const float pixelY = static_cast<float>(y) + 0.5f;

    float bestDepth = 2.0f;
    float4 bestColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < shapeCount; ++i) {
	const float* shape = shapes + i * 12;
	const int typeId = static_cast<int>(roundf(shape[0]));
	const float2 local = make_f2(pixelX - shape[1], pixelY - shape[2]);
	const float size = shape[3];
	const float depth = shape[4];

	float dist = 1.0f;
	if (typeId == 0) {
	    dist = sdfCircle(local, size);
	} else if (typeId == 1) {
	    dist = sdfBox(local, size);
	} else {
	    dist = sdfTriangle(local, size);
	}

	if (dist <= 0.0f && depth < bestDepth) {
	    bestDepth = depth;
	    bestColor = make_float4(shape[5], shape[6], shape[7], shape[8]);
	}
    }

    output[y * width + x] = make_uchar4(
	static_cast<unsigned char>(fminf(fmaxf(roundf(bestColor.x * 255.0f), 0.0f), 255.0f)),
	static_cast<unsigned char>(fminf(fmaxf(roundf(bestColor.y * 255.0f), 0.0f), 255.0f)),
	static_cast<unsigned char>(fminf(fmaxf(roundf(bestColor.z * 255.0f), 0.0f), 255.0f)),
	static_cast<unsigned char>(fminf(fmaxf(roundf(bestColor.w * 255.0f), 0.0f), 255.0f)));
}

}  // namespace

void launchRandomFillKernel(float* out, int size, unsigned int seed, float minValue, float maxValue,
			    unsigned int salt, cudaStream_t stream) {
    const uint32_t count = static_cast<uint32_t>(size) * static_cast<uint32_t>(size);
    const int blocks = static_cast<int>((count + kRandomFillBlockSize - 1) / kRandomFillBlockSize);
    randomFillKernel<<<blocks, kRandomFillBlockSize, 0, stream>>>(out, count, seed, minValue,
								  maxValue, salt);
}

void launchMatrixMulNaiveKernel(const float* matrixA, const float* matrixB, float* matrixC,
				int size, cudaStream_t stream) {
    constexpr dim3 block(kNaiveBlockSize, kNaiveBlockSize);
    const dim3 grid((size + kNaiveBlockSize - 1) / kNaiveBlockSize,
		    (size + kNaiveBlockSize - 1) / kNaiveBlockSize);
    matrixMulNaiveKernel<<<grid, block, 0, stream>>>(matrixA, matrixB, matrixC, size);
}

void launchMatrixMulTiledKernel(const float* matrixA, const float* matrixB, float* matrixC,
				int size, cudaStream_t stream) {
    constexpr dim3 block(kTiledBlockSize, kTiledBlockSize);
    const dim3 grid((size + kTiledSize - 1) / kTiledSize, (size + kTiledSize - 1) / kTiledSize);
    matrixMulTiledKernel<<<grid, block, 0, stream>>>(matrixA, matrixB, matrixC, size);
}

void launchMlpGemvKernel(const float* inputVector, const float* weights, const float* bias,
			 float* outputVector, int inputSize, int outputSize, bool applyRelu,
			 cudaStream_t stream) {
    constexpr dim3 block(kMlpGemvBlockSize);
    const dim3 grid(static_cast<unsigned int>(outputSize));
    mlpGemvKernel<<<grid, block, 0, stream>>>(inputVector, weights, bias, outputVector, inputSize,
					      outputSize, applyRelu ? 1 : 0);
}

void launchCnnConv2dKernel(const float* input, const float* weights, const float* bias,
			   float* output, int inChannels, int outChannels, int inHeight,
			   int inWidth, bool applyRelu, cudaStream_t stream) {
    constexpr dim3 block(kCnnConvBlockSize, kCnnConvBlockSize);
    const dim3 grid(
	static_cast<unsigned int>((inWidth + kCnnConvBlockSize - 1) / kCnnConvBlockSize),
	static_cast<unsigned int>((inHeight + kCnnConvBlockSize - 1) / kCnnConvBlockSize),
	static_cast<unsigned int>(outChannels));

    cnnConv2dKernel<<<grid, block, 0, stream>>>(input, weights, bias, output, inChannels,
						outChannels, inHeight, inWidth, applyRelu ? 1 : 0);
}

void launchCnnMaxPool2x2Kernel(const float* input, float* output, int channels, int inHeight,
			       int inWidth, cudaStream_t stream) {
    const int outHeight = inHeight / 2;
    const int outWidth = inWidth / 2;
    constexpr dim3 block(kCnnPoolBlockSize, kCnnPoolBlockSize);
    const dim3 grid(
	static_cast<unsigned int>((outWidth + kCnnPoolBlockSize - 1) / kCnnPoolBlockSize),
	static_cast<unsigned int>((outHeight + kCnnPoolBlockSize - 1) / kCnnPoolBlockSize),
	static_cast<unsigned int>(channels));

    cnnMaxPool2x2Kernel<<<grid, block, 0, stream>>>(input, output, channels, inHeight, inWidth);
}

void launchVideoBilinearDownscaleKernel(const unsigned char* inputRgba, unsigned char* outputRgba,
					int srcWidth, int srcHeight, int dstWidth, int dstHeight,
					cudaStream_t stream) {
    constexpr dim3 block(kVideoBlockSize, kVideoBlockSize);
    const dim3 grid(static_cast<unsigned int>((dstWidth + kVideoBlockSize - 1) / kVideoBlockSize),
		    static_cast<unsigned int>((dstHeight + kVideoBlockSize - 1) / kVideoBlockSize));

    videoBilinearDownscaleKernel<<<grid, block, 0, stream>>>(
	reinterpret_cast<const uchar4*>(inputRgba), reinterpret_cast<uchar4*>(outputRgba), srcWidth,
	srcHeight, dstWidth, dstHeight);
}

void launchVideoRgbHistogramKernel(const unsigned char* inputRgba, unsigned int* histogram,
				   int pixelCount, cudaStream_t stream) {
    const int blocks = (pixelCount + kVideoHistogramBlockSize - 1) / kVideoHistogramBlockSize;
    videoRgbHistogramKernel<<<blocks, kVideoHistogramBlockSize, 0, stream>>>(
	reinterpret_cast<const uchar4*>(inputRgba), histogram, pixelCount);
}

void launchGaussianBlurHorizontalKernel(const unsigned char* inputRgba, unsigned char* outputRgba,
					int width, int height, cudaStream_t stream) {
    constexpr dim3 block(kGaussianBlockSize, kGaussianBlockSize);
    const dim3 grid(
	static_cast<unsigned int>((width + kGaussianBlockSize - 1) / kGaussianBlockSize),
	static_cast<unsigned int>((height + kGaussianBlockSize - 1) / kGaussianBlockSize));

    gaussianBlurHorizontalKernel<<<grid, block, 0, stream>>>(
	reinterpret_cast<const unsigned int*>(inputRgba),
	reinterpret_cast<unsigned int*>(outputRgba), width, height);
}

void launchGaussianBlurVerticalKernel(const unsigned char* inputRgba, unsigned char* outputRgba,
				      int width, int height, cudaStream_t stream) {
    constexpr dim3 block(kGaussianBlockSize, kGaussianBlockSize);
    const dim3 grid(
	static_cast<unsigned int>((width + kGaussianBlockSize - 1) / kGaussianBlockSize),
	static_cast<unsigned int>((height + kGaussianBlockSize - 1) / kGaussianBlockSize));

    gaussianBlurVerticalKernel<<<grid, block, 0, stream>>>(
	reinterpret_cast<const unsigned int*>(inputRgba),
	reinterpret_cast<unsigned int*>(outputRgba), width, height);
}

void launchRenderSceneKernel(const float* shapes, int shapeCount, unsigned char* outputRgba,
			     int width, int height, cudaStream_t stream) {
    constexpr dim3 block(kRenderBlockSize, kRenderBlockSize);
    const dim3 grid(static_cast<unsigned int>((width + kRenderBlockSize - 1) / kRenderBlockSize),
		    static_cast<unsigned int>((height + kRenderBlockSize - 1) / kRenderBlockSize));

    renderSceneKernel<<<grid, block, 0, stream>>>(
	shapes, shapeCount, reinterpret_cast<uchar4*>(outputRgba), width, height);
}
