#include "cuda_worker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace {

constexpr unsigned int kSaltA = 0x9E3779B9U;
constexpr unsigned int kSaltB = 0x85EBCA6BU;

constexpr int kMlpInputSize = 16384;
constexpr int kMlpHidden1Size = 2048;
constexpr int kMlpHidden2Size = 512;
constexpr int kMlpOutputSize = 10;

constexpr int kCnnInputChannels = 3;
constexpr int kCnnInputHeight = 128;
constexpr int kCnnInputWidth = 128;
constexpr int kCnnConv1OutChannels = 16;
constexpr int kCnnConv2OutChannels = 32;
constexpr int kCnnDense1Out = 128;
constexpr int kCnnOutputSize = 10;

constexpr int kCnnPool1Height = 64;
constexpr int kCnnPool1Width = 64;
constexpr int kCnnPool2Height = 32;
constexpr int kCnnPool2Width = 32;
constexpr int kCnnFlattenSize = kCnnConv2OutChannels * kCnnPool2Height * kCnnPool2Width;

constexpr size_t kMlpW1Count = static_cast<size_t>(kMlpInputSize) * static_cast<size_t>(kMlpHidden1Size);
constexpr size_t kMlpB1Count = static_cast<size_t>(kMlpHidden1Size);
constexpr size_t kMlpW2Count = static_cast<size_t>(kMlpHidden1Size) * static_cast<size_t>(kMlpHidden2Size);
constexpr size_t kMlpB2Count = static_cast<size_t>(kMlpHidden2Size);
constexpr size_t kMlpW3Count = static_cast<size_t>(kMlpHidden2Size) * static_cast<size_t>(kMlpOutputSize);
constexpr size_t kMlpB3Count = static_cast<size_t>(kMlpOutputSize);
constexpr size_t kMlpTotalWeightsCount = kMlpW1Count + kMlpB1Count + kMlpW2Count + kMlpB2Count + kMlpW3Count + kMlpB3Count;

constexpr size_t kCnnConv1WCount = static_cast<size_t>(kCnnConv1OutChannels) * static_cast<size_t>(kCnnInputChannels) * 3U * 3U;
constexpr size_t kCnnConv1BCount = static_cast<size_t>(kCnnConv1OutChannels);
constexpr size_t kCnnConv2WCount = static_cast<size_t>(kCnnConv2OutChannels) * static_cast<size_t>(kCnnConv1OutChannels) * 3U * 3U;
constexpr size_t kCnnConv2BCount = static_cast<size_t>(kCnnConv2OutChannels);
constexpr size_t kCnnDense1WCount = static_cast<size_t>(kCnnFlattenSize) * static_cast<size_t>(kCnnDense1Out);
constexpr size_t kCnnDense1BCount = static_cast<size_t>(kCnnDense1Out);
constexpr size_t kCnnDense2WCount = static_cast<size_t>(kCnnDense1Out) * static_cast<size_t>(kCnnOutputSize);
constexpr size_t kCnnDense2BCount = static_cast<size_t>(kCnnOutputSize);
constexpr size_t kCnnTotalWeightsCount =
  kCnnConv1WCount +
  kCnnConv1BCount +
  kCnnConv2WCount +
  kCnnConv2BCount +
  kCnnDense1WCount +
  kCnnDense1BCount +
  kCnnDense2WCount +
  kCnnDense2BCount;

std::string GetStringProperty(const Napi::Object& obj, const char* key, const std::string& fallback) {
  const Napi::Value value = obj.Get(key);
  if (value.IsString()) {
    return value.As<Napi::String>().Utf8Value();
  }
  return fallback;
}

bool GetBoolProperty(const Napi::Object& obj, const char* key, bool fallback) {
  const Napi::Value value = obj.Get(key);
  if (value.IsBoolean()) {
    return value.As<Napi::Boolean>().Value();
  }
  return fallback;
}

float GetFloatProperty(const Napi::Object& obj, const char* key, float fallback) {
  const Napi::Value value = obj.Get(key);
  if (value.IsNumber()) {
    return static_cast<float>(value.As<Napi::Number>().DoubleValue());
  }
  return fallback;
}

uint32_t GetUint32Property(const Napi::Object& obj, const char* key, uint32_t fallback) {
  const Napi::Value value = obj.Get(key);
  if (value.IsNumber()) {
    return static_cast<uint32_t>(value.As<Napi::Number>().Uint32Value());
  }
  return fallback;
}

std::vector<float> ParseFloat32Array(const Napi::Object& inputObj, const char* key, size_t expectedLength) {
  const Napi::Value value = inputObj.Get(key);
  if (!value.IsTypedArray()) {
    throw std::runtime_error(std::string(key) + " must be a Float32Array.");
  }

  const Napi::TypedArray typedArray = value.As<Napi::TypedArray>();
  if (typedArray.TypedArrayType() != napi_float32_array) {
    throw std::runtime_error(std::string(key) + " must be a Float32Array.");
  }

  const Napi::Float32Array array = value.As<Napi::Float32Array>();
  if (array.ElementLength() != expectedLength) {
    throw std::runtime_error(std::string(key) + " has invalid length.");
  }

  std::vector<float> out(expectedLength);
  std::memcpy(out.data(), array.Data(), expectedLength * sizeof(float));
  return out;
}

double ToMiB(size_t bytes) {
  return std::round((static_cast<double>(bytes) / (1024.0 * 1024.0)) * 1000.0) / 1000.0;
}

Napi::Object BuildMemoryEstimate(Napi::Env env, size_t gpuAllocatedBytes, size_t hostAllocatedBytes) {
  Napi::Object memory = Napi::Object::New(env);
  memory.Set("gpuAllocatedBytes", Napi::Number::New(env, static_cast<double>(gpuAllocatedBytes)));
  memory.Set("gpuAllocatedMiB", Napi::Number::New(env, ToMiB(gpuAllocatedBytes)));
  memory.Set("hostAllocatedBytes", Napi::Number::New(env, static_cast<double>(hostAllocatedBytes)));
  memory.Set("hostAllocatedMiB", Napi::Number::New(env, ToMiB(hostAllocatedBytes)));
  return memory;
}

size_t ComputeMlpGpuBytes() {
  const size_t weightsBytes = (kMlpW1Count + kMlpB1Count + kMlpW2Count + kMlpB2Count + kMlpW3Count + kMlpB3Count) * sizeof(float);
  const size_t activationsBytes = (kMlpInputSize + kMlpHidden1Size + kMlpHidden2Size + kMlpOutputSize) * sizeof(float);
  return weightsBytes + activationsBytes;
}

size_t ComputeCnnGpuBytes() {
  const size_t weightsBytes =
    (kCnnConv1WCount +
     kCnnConv1BCount +
     kCnnConv2WCount +
     kCnnConv2BCount +
     kCnnDense1WCount +
     kCnnDense1BCount +
     kCnnDense2WCount +
     kCnnDense2BCount) * sizeof(float);

  const size_t activationsBytes =
    (static_cast<size_t>(kCnnInputChannels) * static_cast<size_t>(kCnnInputHeight) * static_cast<size_t>(kCnnInputWidth) +
     static_cast<size_t>(kCnnConv1OutChannels) * static_cast<size_t>(kCnnInputHeight) * static_cast<size_t>(kCnnInputWidth) +
     static_cast<size_t>(kCnnConv1OutChannels) * static_cast<size_t>(kCnnPool1Height) * static_cast<size_t>(kCnnPool1Width) +
     static_cast<size_t>(kCnnConv2OutChannels) * static_cast<size_t>(kCnnPool1Height) * static_cast<size_t>(kCnnPool1Width) +
     static_cast<size_t>(kCnnConv2OutChannels) * static_cast<size_t>(kCnnPool2Height) * static_cast<size_t>(kCnnPool2Width) +
     static_cast<size_t>(kCnnDense1Out) +
     static_cast<size_t>(kCnnOutputSize)) * sizeof(float);

  return weightsBytes + activationsBytes;
}

struct CudaMlpModelState {
  std::mutex mutex;
  bool loaded = false;

  float* dW1 = nullptr;
  float* dB1 = nullptr;
  float* dW2 = nullptr;
  float* dB2 = nullptr;
  float* dW3 = nullptr;
  float* dB3 = nullptr;

  float* dInput = nullptr;
  float* dH1 = nullptr;
  float* dH2 = nullptr;
  float* dOut = nullptr;

  size_t gpuAllocatedBytes = 0;
};

CudaMlpModelState gMlpModel;

struct CudaCnnModelState {
  std::mutex mutex;
  bool loaded = false;

  float* dConv1W = nullptr;
  float* dConv1B = nullptr;
  float* dConv2W = nullptr;
  float* dConv2B = nullptr;
  float* dDense1W = nullptr;
  float* dDense1B = nullptr;
  float* dDense2W = nullptr;
  float* dDense2B = nullptr;

  float* dInput = nullptr;
  float* dConv1Out = nullptr;
  float* dPool1Out = nullptr;
  float* dConv2Out = nullptr;
  float* dPool2Out = nullptr;
  float* dDense1Out = nullptr;
  float* dOut = nullptr;

  size_t gpuAllocatedBytes = 0;
};

CudaCnnModelState gCnnModel;

void FreeMlpModelLocked() {
  if (gMlpModel.dW1) {
    cudaFree(gMlpModel.dW1);
    gMlpModel.dW1 = nullptr;
  }
  if (gMlpModel.dB1) {
    cudaFree(gMlpModel.dB1);
    gMlpModel.dB1 = nullptr;
  }
  if (gMlpModel.dW2) {
    cudaFree(gMlpModel.dW2);
    gMlpModel.dW2 = nullptr;
  }
  if (gMlpModel.dB2) {
    cudaFree(gMlpModel.dB2);
    gMlpModel.dB2 = nullptr;
  }
  if (gMlpModel.dW3) {
    cudaFree(gMlpModel.dW3);
    gMlpModel.dW3 = nullptr;
  }
  if (gMlpModel.dB3) {
    cudaFree(gMlpModel.dB3);
    gMlpModel.dB3 = nullptr;
  }
  if (gMlpModel.dInput) {
    cudaFree(gMlpModel.dInput);
    gMlpModel.dInput = nullptr;
  }
  if (gMlpModel.dH1) {
    cudaFree(gMlpModel.dH1);
    gMlpModel.dH1 = nullptr;
  }
  if (gMlpModel.dH2) {
    cudaFree(gMlpModel.dH2);
    gMlpModel.dH2 = nullptr;
  }
  if (gMlpModel.dOut) {
    cudaFree(gMlpModel.dOut);
    gMlpModel.dOut = nullptr;
  }

  gMlpModel.loaded = false;
  gMlpModel.gpuAllocatedBytes = 0;
}

void FreeCnnModelLocked() {
  if (gCnnModel.dConv1W) {
    cudaFree(gCnnModel.dConv1W);
    gCnnModel.dConv1W = nullptr;
  }
  if (gCnnModel.dConv1B) {
    cudaFree(gCnnModel.dConv1B);
    gCnnModel.dConv1B = nullptr;
  }
  if (gCnnModel.dConv2W) {
    cudaFree(gCnnModel.dConv2W);
    gCnnModel.dConv2W = nullptr;
  }
  if (gCnnModel.dConv2B) {
    cudaFree(gCnnModel.dConv2B);
    gCnnModel.dConv2B = nullptr;
  }
  if (gCnnModel.dDense1W) {
    cudaFree(gCnnModel.dDense1W);
    gCnnModel.dDense1W = nullptr;
  }
  if (gCnnModel.dDense1B) {
    cudaFree(gCnnModel.dDense1B);
    gCnnModel.dDense1B = nullptr;
  }
  if (gCnnModel.dDense2W) {
    cudaFree(gCnnModel.dDense2W);
    gCnnModel.dDense2W = nullptr;
  }
  if (gCnnModel.dDense2B) {
    cudaFree(gCnnModel.dDense2B);
    gCnnModel.dDense2B = nullptr;
  }

  if (gCnnModel.dInput) {
    cudaFree(gCnnModel.dInput);
    gCnnModel.dInput = nullptr;
  }
  if (gCnnModel.dConv1Out) {
    cudaFree(gCnnModel.dConv1Out);
    gCnnModel.dConv1Out = nullptr;
  }
  if (gCnnModel.dPool1Out) {
    cudaFree(gCnnModel.dPool1Out);
    gCnnModel.dPool1Out = nullptr;
  }
  if (gCnnModel.dConv2Out) {
    cudaFree(gCnnModel.dConv2Out);
    gCnnModel.dConv2Out = nullptr;
  }
  if (gCnnModel.dPool2Out) {
    cudaFree(gCnnModel.dPool2Out);
    gCnnModel.dPool2Out = nullptr;
  }
  if (gCnnModel.dDense1Out) {
    cudaFree(gCnnModel.dDense1Out);
    gCnnModel.dDense1Out = nullptr;
  }
  if (gCnnModel.dOut) {
    cudaFree(gCnnModel.dOut);
    gCnnModel.dOut = nullptr;
  }

  gCnnModel.loaded = false;
  gCnnModel.gpuAllocatedBytes = 0;
}

} // namespace

CudaMatrixRequest ParseCudaMatrixRequest(const Napi::CallbackInfo& info) {
  if (info.Length() < 1 || !info[0].IsObject()) {
    throw std::runtime_error("multiplyMatrixCuda expects an options object.");
  }

  const Napi::Object options = info[0].As<Napi::Object>();

  CudaMatrixRequest request;
  request.size = options.Get("size").As<Napi::Number>().Int32Value();
  request.optimized = GetBoolProperty(options, "optimized", false);
  request.readback = GetBoolProperty(options, "readback", true);
  request.randomInput = GetStringProperty(options, "inputMode", "random") != "custom";
  const float randomMin = GetFloatProperty(options, "randomMin", 0.0F);
  const float randomMax = GetFloatProperty(options, "randomMax", 1.0F);
  request.randomMin = std::min(randomMin, randomMax);
  request.randomMax = std::max(randomMin, randomMax);
  request.randomSeed = GetUint32Property(options, "randomSeed", 0U);

  if (request.size <= 0) {
    throw std::runtime_error("size must be a positive integer.");
  }

  const auto totalElements = static_cast<size_t>(request.size) * static_cast<size_t>(request.size);
  if (totalElements > (std::numeric_limits<size_t>::max() / sizeof(float))) {
    throw std::runtime_error("Requested matrix size is too large.");
  }

  if (!request.randomInput) {
    request.matrixA = ParseFloat32Array(options, "matrixA", totalElements);
    request.matrixB = ParseFloat32Array(options, "matrixB", totalElements);
  }

  return request;
}

CudaMatrixWorker::CudaMatrixWorker(Napi::Env env, const CudaMatrixRequest& request)
  : Napi::AsyncWorker(env),
    deferred_(Napi::Promise::Deferred::New(env)),
    request_(request) {}

CudaMatrixWorker::~CudaMatrixWorker() {
  Cleanup();
}

Napi::Promise CudaMatrixWorker::GetPromise() const {
  return deferred_.Promise();
}

void CudaMatrixWorker::Execute() {
  try {
    const auto matrixElements = static_cast<size_t>(request_.size) * static_cast<size_t>(request_.size);
    const auto matrixBytes = matrixElements * sizeof(float);

    gpuAllocatedBytes_ = matrixBytes * 3;
    hostAllocatedBytes_ = (request_.randomInput ? 0 : matrixBytes * 2) + (request_.readback ? matrixBytes : 0);

    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&dMatrixA_), matrixBytes));
    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&dMatrixB_), matrixBytes));
    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&dMatrixC_), matrixBytes));

    if (request_.randomInput) {
      CUDA_CHECK_THROW(cudaEventCreate(&generationStartEvent_));
      CUDA_CHECK_THROW(cudaEventCreate(&generationStopEvent_));

      CUDA_CHECK_THROW(cudaEventRecord(generationStartEvent_, nullptr));
      launchRandomFillKernel(dMatrixA_, request_.size, request_.randomSeed, request_.randomMin, request_.randomMax, kSaltA, nullptr);
      launchRandomFillKernel(dMatrixB_, request_.size, request_.randomSeed, request_.randomMin, request_.randomMax, kSaltB, nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(generationStopEvent_, nullptr));
      CUDA_CHECK_THROW(cudaEventSynchronize(generationStopEvent_));

      float generationDurationMs = 0.0F;
      CUDA_CHECK_THROW(cudaEventElapsedTime(&generationDurationMs, generationStartEvent_, generationStopEvent_));
      generationDurationMs_ = static_cast<double>(generationDurationMs);
    } else {
      CUDA_CHECK_THROW(cudaMemcpy(dMatrixA_, request_.matrixA.data(), matrixBytes, cudaMemcpyHostToDevice));
      CUDA_CHECK_THROW(cudaMemcpy(dMatrixB_, request_.matrixB.data(), matrixBytes, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK_THROW(cudaEventCreate(&multiplyStartEvent_));
    CUDA_CHECK_THROW(cudaEventCreate(&multiplyStopEvent_));

    CUDA_CHECK_THROW(cudaEventRecord(multiplyStartEvent_, nullptr));
    if (request_.optimized) {
      launchMatrixMulTiledKernel(dMatrixA_, dMatrixB_, dMatrixC_, request_.size, nullptr);
    } else {
      launchMatrixMulNaiveKernel(dMatrixA_, dMatrixB_, dMatrixC_, request_.size, nullptr);
    }
    CUDA_CHECK_THROW(cudaGetLastError());
    CUDA_CHECK_THROW(cudaEventRecord(multiplyStopEvent_, nullptr));
    CUDA_CHECK_THROW(cudaEventSynchronize(multiplyStopEvent_));

    float multiplyDurationMs = 0.0F;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&multiplyDurationMs, multiplyStartEvent_, multiplyStopEvent_));
    multiplyDurationMs_ = static_cast<double>(multiplyDurationMs);
    totalDurationMs_ = multiplyDurationMs_ + generationDurationMs_.value_or(0.0);

    if (request_.readback) {
      output_.resize(matrixElements);
      CUDA_CHECK_THROW(cudaMemcpy(output_.data(), dMatrixC_, matrixBytes, cudaMemcpyDeviceToHost));
    }
  } catch (const std::exception& ex) {
    SetError(ex.what());
  }
}

void CudaMatrixWorker::OnOK() {
  Napi::HandleScope scope(Env());
  deferred_.Resolve(BuildResult(Env()));
}

void CudaMatrixWorker::OnError(const Napi::Error& error) {
  Napi::HandleScope scope(Env());
  deferred_.Reject(error.Value());
}

Napi::Value CudaMatrixWorker::BuildResult(Napi::Env env) const {
  Napi::Object result = Napi::Object::New(env);

  if (request_.readback) {
    Napi::ArrayBuffer outputBuffer = Napi::ArrayBuffer::New(env, output_.size() * sizeof(float));
    std::memcpy(outputBuffer.Data(), output_.data(), output_.size() * sizeof(float));
    result.Set("output", Napi::Float32Array::New(env, output_.size(), outputBuffer, 0));
  } else {
    result.Set("output", env.Null());
  }

  if (generationDurationMs_.has_value()) {
    result.Set("generationDurationMs", Napi::Number::New(env, generationDurationMs_.value()));
  } else {
    result.Set("generationDurationMs", env.Null());
  }

  result.Set("multiplyDurationMs", Napi::Number::New(env, multiplyDurationMs_));
  result.Set("totalDurationMs", Napi::Number::New(env, totalDurationMs_));
  result.Set("timingSource", Napi::String::New(env, "gpu-timestamp"));
  result.Set("memoryEstimate", BuildMemoryEstimate(env, gpuAllocatedBytes_, hostAllocatedBytes_));

  return result;
}

double CudaMatrixWorker::ToMiB(size_t bytes) {
  return std::round((static_cast<double>(bytes) / (1024.0 * 1024.0)) * 1000.0) / 1000.0;
}

void CudaMatrixWorker::Cleanup() {
  if (generationStartEvent_) {
    cudaEventDestroy(generationStartEvent_);
    generationStartEvent_ = nullptr;
  }
  if (generationStopEvent_) {
    cudaEventDestroy(generationStopEvent_);
    generationStopEvent_ = nullptr;
  }
  if (multiplyStartEvent_) {
    cudaEventDestroy(multiplyStartEvent_);
    multiplyStartEvent_ = nullptr;
  }
  if (multiplyStopEvent_) {
    cudaEventDestroy(multiplyStopEvent_);
    multiplyStopEvent_ = nullptr;
  }

  if (dMatrixA_) {
    cudaFree(dMatrixA_);
    dMatrixA_ = nullptr;
  }
  if (dMatrixB_) {
    cudaFree(dMatrixB_);
    dMatrixB_ = nullptr;
  }
  if (dMatrixC_) {
    cudaFree(dMatrixC_);
    dMatrixC_ = nullptr;
  }
}

class CudaMlpLoadWorker final : public Napi::AsyncWorker {
public:
  CudaMlpLoadWorker(Napi::Env env, std::vector<float> weights)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      weights_(std::move(weights)) {}

  Napi::Promise GetPromise() const {
    return deferred_.Promise();
  }

  void Execute() override {
    std::lock_guard<std::mutex> lock(gMlpModel.mutex);

    try {
      if (weights_.size() != kMlpTotalWeightsCount) {
        throw std::runtime_error("weights must match the expected MLP layout size.");
      }

      FreeMlpModelLocked();

      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dW1), kMlpW1Count * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dB1), kMlpB1Count * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dW2), kMlpW2Count * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dB2), kMlpB2Count * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dW3), kMlpW3Count * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dB3), kMlpB3Count * sizeof(float)));

      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dInput), kMlpInputSize * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dH1), kMlpHidden1Size * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dH2), kMlpHidden2Size * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gMlpModel.dOut), kMlpOutputSize * sizeof(float)));

      const float* cursor = weights_.data();
      CUDA_CHECK_THROW(cudaMemcpy(gMlpModel.dW1, cursor, kMlpW1Count * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kMlpW1Count;
      CUDA_CHECK_THROW(cudaMemcpy(gMlpModel.dB1, cursor, kMlpB1Count * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kMlpB1Count;
      CUDA_CHECK_THROW(cudaMemcpy(gMlpModel.dW2, cursor, kMlpW2Count * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kMlpW2Count;
      CUDA_CHECK_THROW(cudaMemcpy(gMlpModel.dB2, cursor, kMlpB2Count * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kMlpB2Count;
      CUDA_CHECK_THROW(cudaMemcpy(gMlpModel.dW3, cursor, kMlpW3Count * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kMlpW3Count;
      CUDA_CHECK_THROW(cudaMemcpy(gMlpModel.dB3, cursor, kMlpB3Count * sizeof(float), cudaMemcpyHostToDevice));

      gMlpModel.loaded = true;
      gMlpModel.gpuAllocatedBytes = ComputeMlpGpuBytes();

      hostAllocatedBytes_ = kMlpTotalWeightsCount * sizeof(float);
      gpuAllocatedBytes_ = gMlpModel.gpuAllocatedBytes;
    } catch (const std::exception& ex) {
      FreeMlpModelLocked();
      SetError(ex.what());
    }
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    Napi::Object result = Napi::Object::New(Env());
    result.Set("status", Napi::String::New(Env(), "loaded"));
    result.Set("memoryEstimate", BuildMemoryEstimate(Env(), gpuAllocatedBytes_, hostAllocatedBytes_));
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& error) override {
    Napi::HandleScope scope(Env());
    deferred_.Reject(error.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  std::vector<float> weights_;
  size_t hostAllocatedBytes_ = 0;
  size_t gpuAllocatedBytes_ = 0;
};

class CudaMlpPredictWorker final : public Napi::AsyncWorker {
public:
  CudaMlpPredictWorker(Napi::Env env, std::vector<float> input)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      input_(std::move(input)) {}

  Napi::Promise GetPromise() const {
    return deferred_.Promise();
  }

  void Execute() override {
    const auto totalStart = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(gMlpModel.mutex);

    if (!gMlpModel.loaded) {
      SetError("CUDA MLP model is not loaded.");
      return;
    }

    std::array<cudaEvent_t, 6> events = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    auto destroyEvents = [&events]() {
      for (cudaEvent_t& event : events) {
        if (event != nullptr) {
          cudaEventDestroy(event);
          event = nullptr;
        }
      }
    };

    try {
      for (cudaEvent_t& event : events) {
        CUDA_CHECK_THROW(cudaEventCreate(&event));
      }

      CUDA_CHECK_THROW(cudaMemcpy(gMlpModel.dInput, input_.data(), kMlpInputSize * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CHECK_THROW(cudaEventRecord(events[0], nullptr));
      launchMlpGemvKernel(gMlpModel.dInput, gMlpModel.dW1, gMlpModel.dB1, gMlpModel.dH1, kMlpInputSize, kMlpHidden1Size, true, nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[1], nullptr));

      CUDA_CHECK_THROW(cudaEventRecord(events[2], nullptr));
      launchMlpGemvKernel(gMlpModel.dH1, gMlpModel.dW2, gMlpModel.dB2, gMlpModel.dH2, kMlpHidden1Size, kMlpHidden2Size, true, nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[3], nullptr));

      CUDA_CHECK_THROW(cudaEventRecord(events[4], nullptr));
      launchMlpGemvKernel(gMlpModel.dH2, gMlpModel.dW3, gMlpModel.dB3, gMlpModel.dOut, kMlpHidden2Size, kMlpOutputSize, false, nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[5], nullptr));
      CUDA_CHECK_THROW(cudaEventSynchronize(events[5]));

      float layer1Ms = 0.0F;
      float layer2Ms = 0.0F;
      float layer3Ms = 0.0F;
      CUDA_CHECK_THROW(cudaEventElapsedTime(&layer1Ms, events[0], events[1]));
      CUDA_CHECK_THROW(cudaEventElapsedTime(&layer2Ms, events[2], events[3]));
      CUDA_CHECK_THROW(cudaEventElapsedTime(&layer3Ms, events[4], events[5]));
      gpuDurationMs_ = static_cast<double>(layer1Ms + layer2Ms + layer3Ms);

      output_.resize(kMlpOutputSize);
      CUDA_CHECK_THROW(cudaMemcpy(output_.data(), gMlpModel.dOut, kMlpOutputSize * sizeof(float), cudaMemcpyDeviceToHost));

      const auto totalStop = std::chrono::steady_clock::now();
      totalDurationMs_ = std::chrono::duration<double, std::milli>(totalStop - totalStart).count();
      destroyEvents();
    } catch (const std::exception& ex) {
      destroyEvents();
      SetError(ex.what());
    }
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());

    Napi::Object result = Napi::Object::New(Env());
    Napi::ArrayBuffer logitsBuffer = Napi::ArrayBuffer::New(Env(), output_.size() * sizeof(float));
    std::memcpy(logitsBuffer.Data(), output_.data(), output_.size() * sizeof(float));

    result.Set("logits", Napi::Float32Array::New(Env(), output_.size(), logitsBuffer, 0));
    result.Set("gpuDurationMs", Napi::Number::New(Env(), gpuDurationMs_));
    result.Set("totalDurationMs", Napi::Number::New(Env(), totalDurationMs_));
    result.Set("timingSource", Napi::String::New(Env(), "gpu-timestamp"));

    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& error) override {
    Napi::HandleScope scope(Env());
    deferred_.Reject(error.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  std::vector<float> input_;
  std::vector<float> output_;
  double gpuDurationMs_ = 0.0;
  double totalDurationMs_ = 0.0;
};

class CudaMlpUnloadWorker final : public Napi::AsyncWorker {
public:
  explicit CudaMlpUnloadWorker(Napi::Env env)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)) {}

  Napi::Promise GetPromise() const {
    return deferred_.Promise();
  }

  void Execute() override {
    std::lock_guard<std::mutex> lock(gMlpModel.mutex);
    FreeMlpModelLocked();
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    Napi::Object result = Napi::Object::New(Env());
    result.Set("status", Napi::String::New(Env(), "unloaded"));
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& error) override {
    Napi::HandleScope scope(Env());
    deferred_.Reject(error.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
};

class CudaCnnLoadWorker final : public Napi::AsyncWorker {
public:
  CudaCnnLoadWorker(Napi::Env env, std::vector<float> weights)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      weights_(std::move(weights)) {}

  Napi::Promise GetPromise() const {
    return deferred_.Promise();
  }

  void Execute() override {
    std::lock_guard<std::mutex> lock(gCnnModel.mutex);

    try {
      if (weights_.size() != kCnnTotalWeightsCount) {
        throw std::runtime_error("weights must match the expected CNN layout size.");
      }

      FreeCnnModelLocked();

      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dConv1W), kCnnConv1WCount * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dConv1B), kCnnConv1BCount * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dConv2W), kCnnConv2WCount * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dConv2B), kCnnConv2BCount * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dDense1W), kCnnDense1WCount * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dDense1B), kCnnDense1BCount * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dDense2W), kCnnDense2WCount * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dDense2B), kCnnDense2BCount * sizeof(float)));

      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dInput),
        static_cast<size_t>(kCnnInputChannels) * static_cast<size_t>(kCnnInputHeight) * static_cast<size_t>(kCnnInputWidth) * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dConv1Out),
        static_cast<size_t>(kCnnConv1OutChannels) * static_cast<size_t>(kCnnInputHeight) * static_cast<size_t>(kCnnInputWidth) * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dPool1Out),
        static_cast<size_t>(kCnnConv1OutChannels) * static_cast<size_t>(kCnnPool1Height) * static_cast<size_t>(kCnnPool1Width) * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dConv2Out),
        static_cast<size_t>(kCnnConv2OutChannels) * static_cast<size_t>(kCnnPool1Height) * static_cast<size_t>(kCnnPool1Width) * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dPool2Out),
        static_cast<size_t>(kCnnConv2OutChannels) * static_cast<size_t>(kCnnPool2Height) * static_cast<size_t>(kCnnPool2Width) * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dDense1Out), static_cast<size_t>(kCnnDense1Out) * sizeof(float)));
      CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&gCnnModel.dOut), static_cast<size_t>(kCnnOutputSize) * sizeof(float)));

      const float* cursor = weights_.data();
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dConv1W, cursor, kCnnConv1WCount * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kCnnConv1WCount;
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dConv1B, cursor, kCnnConv1BCount * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kCnnConv1BCount;
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dConv2W, cursor, kCnnConv2WCount * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kCnnConv2WCount;
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dConv2B, cursor, kCnnConv2BCount * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kCnnConv2BCount;
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dDense1W, cursor, kCnnDense1WCount * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kCnnDense1WCount;
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dDense1B, cursor, kCnnDense1BCount * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kCnnDense1BCount;
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dDense2W, cursor, kCnnDense2WCount * sizeof(float), cudaMemcpyHostToDevice));
      cursor += kCnnDense2WCount;
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dDense2B, cursor, kCnnDense2BCount * sizeof(float), cudaMemcpyHostToDevice));

      gCnnModel.loaded = true;
      gCnnModel.gpuAllocatedBytes = ComputeCnnGpuBytes();

      hostAllocatedBytes_ = kCnnTotalWeightsCount * sizeof(float);
      gpuAllocatedBytes_ = gCnnModel.gpuAllocatedBytes;
    } catch (const std::exception& ex) {
      FreeCnnModelLocked();
      SetError(ex.what());
    }
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    Napi::Object result = Napi::Object::New(Env());
    result.Set("status", Napi::String::New(Env(), "loaded"));
    result.Set("memoryEstimate", BuildMemoryEstimate(Env(), gpuAllocatedBytes_, hostAllocatedBytes_));
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& error) override {
    Napi::HandleScope scope(Env());
    deferred_.Reject(error.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  std::vector<float> weights_;
  size_t hostAllocatedBytes_ = 0;
  size_t gpuAllocatedBytes_ = 0;
};

class CudaCnnPredictWorker final : public Napi::AsyncWorker {
public:
  CudaCnnPredictWorker(Napi::Env env, std::vector<float> input)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      input_(std::move(input)) {}

  Napi::Promise GetPromise() const {
    return deferred_.Promise();
  }

  void Execute() override {
    const auto totalStart = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(gCnnModel.mutex);

    if (!gCnnModel.loaded) {
      SetError("CUDA CNN model is not loaded.");
      return;
    }

    std::array<cudaEvent_t, 12> events = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    auto destroyEvents = [&events]() {
      for (cudaEvent_t& event : events) {
        if (event != nullptr) {
          cudaEventDestroy(event);
          event = nullptr;
        }
      }
    };

    try {
      for (cudaEvent_t& event : events) {
        CUDA_CHECK_THROW(cudaEventCreate(&event));
      }

      const size_t inputBytes =
        static_cast<size_t>(kCnnInputChannels) * static_cast<size_t>(kCnnInputHeight) * static_cast<size_t>(kCnnInputWidth) * sizeof(float);
      CUDA_CHECK_THROW(cudaMemcpy(gCnnModel.dInput, input_.data(), inputBytes, cudaMemcpyHostToDevice));

      CUDA_CHECK_THROW(cudaEventRecord(events[0], nullptr));
      launchCnnConv2dKernel(
        gCnnModel.dInput,
        gCnnModel.dConv1W,
        gCnnModel.dConv1B,
        gCnnModel.dConv1Out,
        kCnnInputChannels,
        kCnnConv1OutChannels,
        kCnnInputHeight,
        kCnnInputWidth,
        true,
        nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[1], nullptr));

      CUDA_CHECK_THROW(cudaEventRecord(events[2], nullptr));
      launchCnnMaxPool2x2Kernel(
        gCnnModel.dConv1Out,
        gCnnModel.dPool1Out,
        kCnnConv1OutChannels,
        kCnnInputHeight,
        kCnnInputWidth,
        nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[3], nullptr));

      CUDA_CHECK_THROW(cudaEventRecord(events[4], nullptr));
      launchCnnConv2dKernel(
        gCnnModel.dPool1Out,
        gCnnModel.dConv2W,
        gCnnModel.dConv2B,
        gCnnModel.dConv2Out,
        kCnnConv1OutChannels,
        kCnnConv2OutChannels,
        kCnnPool1Height,
        kCnnPool1Width,
        true,
        nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[5], nullptr));

      CUDA_CHECK_THROW(cudaEventRecord(events[6], nullptr));
      launchCnnMaxPool2x2Kernel(
        gCnnModel.dConv2Out,
        gCnnModel.dPool2Out,
        kCnnConv2OutChannels,
        kCnnPool1Height,
        kCnnPool1Width,
        nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[7], nullptr));

      CUDA_CHECK_THROW(cudaEventRecord(events[8], nullptr));
      launchMlpGemvKernel(
        gCnnModel.dPool2Out,
        gCnnModel.dDense1W,
        gCnnModel.dDense1B,
        gCnnModel.dDense1Out,
        kCnnFlattenSize,
        kCnnDense1Out,
        true,
        nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[9], nullptr));

      CUDA_CHECK_THROW(cudaEventRecord(events[10], nullptr));
      launchMlpGemvKernel(
        gCnnModel.dDense1Out,
        gCnnModel.dDense2W,
        gCnnModel.dDense2B,
        gCnnModel.dOut,
        kCnnDense1Out,
        kCnnOutputSize,
        false,
        nullptr);
      CUDA_CHECK_THROW(cudaGetLastError());
      CUDA_CHECK_THROW(cudaEventRecord(events[11], nullptr));
      CUDA_CHECK_THROW(cudaEventSynchronize(events[11]));

      float totalMs = 0.0F;
      for (int i = 0; i < 12; i += 2) {
        float partMs = 0.0F;
        CUDA_CHECK_THROW(cudaEventElapsedTime(&partMs, events[i], events[i + 1]));
        totalMs += partMs;
      }
      gpuDurationMs_ = static_cast<double>(totalMs);

      output_.resize(kCnnOutputSize);
      CUDA_CHECK_THROW(cudaMemcpy(output_.data(), gCnnModel.dOut, kCnnOutputSize * sizeof(float), cudaMemcpyDeviceToHost));

      const auto totalStop = std::chrono::steady_clock::now();
      totalDurationMs_ = std::chrono::duration<double, std::milli>(totalStop - totalStart).count();
      destroyEvents();
    } catch (const std::exception& ex) {
      destroyEvents();
      SetError(ex.what());
    }
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());

    Napi::Object result = Napi::Object::New(Env());
    Napi::ArrayBuffer logitsBuffer = Napi::ArrayBuffer::New(Env(), output_.size() * sizeof(float));
    std::memcpy(logitsBuffer.Data(), output_.data(), output_.size() * sizeof(float));

    result.Set("logits", Napi::Float32Array::New(Env(), output_.size(), logitsBuffer, 0));
    result.Set("gpuDurationMs", Napi::Number::New(Env(), gpuDurationMs_));
    result.Set("totalDurationMs", Napi::Number::New(Env(), totalDurationMs_));
    result.Set("timingSource", Napi::String::New(Env(), "gpu-timestamp"));
    result.Set("memoryEstimate", BuildMemoryEstimate(Env(), gCnnModel.gpuAllocatedBytes, kCnnTotalWeightsCount * sizeof(float)));

    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& error) override {
    Napi::HandleScope scope(Env());
    deferred_.Reject(error.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  std::vector<float> input_;
  std::vector<float> output_;
  double gpuDurationMs_ = 0.0;
  double totalDurationMs_ = 0.0;
};

class CudaCnnUnloadWorker final : public Napi::AsyncWorker {
public:
  explicit CudaCnnUnloadWorker(Napi::Env env)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)) {}

  Napi::Promise GetPromise() const {
    return deferred_.Promise();
  }

  void Execute() override {
    std::lock_guard<std::mutex> lock(gCnnModel.mutex);
    FreeCnnModelLocked();
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    Napi::Object result = Napi::Object::New(Env());
    result.Set("status", Napi::String::New(Env(), "unloaded"));
    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& error) override {
    Napi::HandleScope scope(Env());
    deferred_.Reject(error.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
};

Napi::Value MultiplyMatrixCuda(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  try {
    const CudaMatrixRequest request = ParseCudaMatrixRequest(info);
    auto* worker = new CudaMatrixWorker(env, request);
    Napi::Promise promise = worker->GetPromise();
    worker->Queue();
    return promise;
  } catch (const std::exception& ex) {
    Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value LoadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  try {
    if (info.Length() < 1 || !info[0].IsObject()) {
      throw std::runtime_error("loadModel expects an options object with weights Float32Array.");
    }

    const Napi::Object options = info[0].As<Napi::Object>();
    std::vector<float> weights = ParseFloat32Array(options, "weights", kMlpTotalWeightsCount);

    auto* worker = new CudaMlpLoadWorker(env, std::move(weights));
    Napi::Promise promise = worker->GetPromise();
    worker->Queue();
    return promise;
  } catch (const std::exception& ex) {
    Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value Predict(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  try {
    if (info.Length() < 1 || !info[0].IsObject()) {
      throw std::runtime_error("predict expects an options object with input Float32Array.");
    }

    const Napi::Object options = info[0].As<Napi::Object>();
    std::vector<float> input = ParseFloat32Array(options, "input", kMlpInputSize);

    auto* worker = new CudaMlpPredictWorker(env, std::move(input));
    Napi::Promise promise = worker->GetPromise();
    worker->Queue();
    return promise;
  } catch (const std::exception& ex) {
    Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value UnloadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  auto* worker = new CudaMlpUnloadWorker(env);
  Napi::Promise promise = worker->GetPromise();
  worker->Queue();
  return promise;
}

Napi::Value LoadCnnModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  try {
    if (info.Length() < 1 || !info[0].IsObject()) {
      throw std::runtime_error("loadCnnModel expects an options object with weights Float32Array.");
    }

    const Napi::Object options = info[0].As<Napi::Object>();
    std::vector<float> weights = ParseFloat32Array(options, "weights", kCnnTotalWeightsCount);

    auto* worker = new CudaCnnLoadWorker(env, std::move(weights));
    Napi::Promise promise = worker->GetPromise();
    worker->Queue();
    return promise;
  } catch (const std::exception& ex) {
    Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value PredictCnn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  try {
    if (info.Length() < 1 || !info[0].IsObject()) {
      throw std::runtime_error("predictCnn expects an options object with input Float32Array.");
    }

    const Napi::Object options = info[0].As<Napi::Object>();
    const size_t inputLength = static_cast<size_t>(kCnnInputChannels) * static_cast<size_t>(kCnnInputHeight) * static_cast<size_t>(kCnnInputWidth);
    std::vector<float> input = ParseFloat32Array(options, "input", inputLength);

    auto* worker = new CudaCnnPredictWorker(env, std::move(input));
    Napi::Promise promise = worker->GetPromise();
    worker->Queue();
    return promise;
  } catch (const std::exception& ex) {
    Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Value UnloadCnnModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  auto* worker = new CudaCnnUnloadWorker(env);
  Napi::Promise promise = worker->GetPromise();
  worker->Queue();
  return promise;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("multiplyMatrixCuda", Napi::Function::New(env, MultiplyMatrixCuda));
  exports.Set("loadModel", Napi::Function::New(env, LoadModel));
  exports.Set("predict", Napi::Function::New(env, Predict));
  exports.Set("unloadModel", Napi::Function::New(env, UnloadModel));
  exports.Set("loadCnnModel", Napi::Function::New(env, LoadCnnModel));
  exports.Set("predictCnn", Napi::Function::New(env, PredictCnn));
  exports.Set("unloadCnnModel", Napi::Function::New(env, UnloadCnnModel));
  return exports;
}

NODE_API_MODULE(cuda_matrix_addon, Init)

