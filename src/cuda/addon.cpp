#include "cuda_worker.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
using namespace Napi;
using namespace std;
namespace {

constexpr unsigned int kSaltA = 0x9E3779B9U;
constexpr unsigned int kSaltB = 0x85EBCA6BU;

string GetStringProperty(const Object& obj, const char* key, const string& fallback) {
  const Value value = obj.Get(key);
  if (value.IsString()) {
    return value.As<String>().Utf8Value();
  }
  return fallback;
}

bool GetBoolProperty(const Object& obj, const char* key, bool fallback) {
  const Value value = obj.Get(key);
  if (value.IsBoolean()) {
    return value.As<Boolean>().Value();
  }
  return fallback;
}

float GetFloatProperty(const Object& obj, const char* key, float fallback) {
  const Value value = obj.Get(key);
  if (value.IsNumber()) {
    return static_cast<float>(value.As<Number>().DoubleValue());
  }
  return fallback;
}

uint32_t GetUint32Property(const Object& obj, const char* key, uint32_t fallback) {
  const Value value = obj.Get(key);
  if (value.IsNumber()) {
    return static_cast<uint32_t>(value.As<Number>().Uint32Value());
  }
  return fallback;
}

vector<float> ParseFloat32Array(const Object& inputObj, const char* key, size_t expectedLength) {
  const Value value = inputObj.Get(key);
  if (!value.IsTypedArray()) {
    throw runtime_error(string(key) + " must be a Float32Array.");
  }

  const TypedArray typedArray = value.As<TypedArray>();
  if (typedArray.TypedArrayType() != napi_float32_array) {
    throw runtime_error(string(key) + " must be a Float32Array.");
  }

  const Float32Array array = value.As<Float32Array>();
  if (array.ElementLength() != expectedLength) {
    throw runtime_error(string(key) + " has invalid length.");
  }

  vector<float> out(expectedLength);
  memcpy(out.data(), array.Data(), expectedLength * sizeof(float));
  return out;
}

} // namespace

CudaMatrixRequest ParseCudaMatrixRequest(const CallbackInfo& info) {
  if (info.Length() < 1 || !info[0].IsObject()) {
    throw runtime_error("multiplyMatrixCuda expects an options object.");
  }

  const Object options = info[0].As<Object>();

  CudaMatrixRequest request;
  request.size = options.Get("size").As<Number>().Int32Value();
  request.optimized = GetBoolProperty(options, "optimized", false);
  request.readback = GetBoolProperty(options, "readback", true);
  request.randomInput = GetStringProperty(options, "inputMode", "random") != "custom";
  const float randomMin = GetFloatProperty(options, "randomMin", 0.0F);
  const float randomMax = GetFloatProperty(options, "randomMax", 1.0F);
  request.randomMin = min(randomMin, randomMax);
  request.randomMax = max(randomMin, randomMax);
  request.randomSeed = GetUint32Property(options, "randomSeed", 0U);

  if (request.size <= 0) {
    throw runtime_error("size must be a positive integer.");
  }

  const auto totalElements = static_cast<size_t>(request.size) * static_cast<size_t>(request.size);
  if (totalElements > (numeric_limits<size_t>::max() / sizeof(float))) {
    throw runtime_error("Requested matrix size is too large.");
  }

  if (!request.randomInput) {
    request.matrixA = ParseFloat32Array(options, "matrixA", totalElements);
    request.matrixB = ParseFloat32Array(options, "matrixB", totalElements);
  }

  return request;
}

CudaMatrixWorker::CudaMatrixWorker(Env env, const CudaMatrixRequest& request)
  : AsyncWorker(env),
    deferred_(Promise::Deferred::New(env)),
    request_(request) {}

CudaMatrixWorker::~CudaMatrixWorker() {
  Cleanup();
}

Promise CudaMatrixWorker::GetPromise() const {
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
  } catch (const exception& ex) {
    SetError(ex.what());
  }
}

void CudaMatrixWorker::OnOK() {
  HandleScope scope(Env());
  deferred_.Resolve(BuildResult(Env()));
}

void CudaMatrixWorker::OnError(const Error& error) {
  HandleScope scope(Env());
  deferred_.Reject(error.Value());
}

Value CudaMatrixWorker::BuildResult(Env env) const {
  Object result = Object::New(env);

  if (request_.readback) {
    ArrayBuffer outputBuffer = ArrayBuffer::New(env, output_.size() * sizeof(float));
    memcpy(outputBuffer.Data(), output_.data(), output_.size() * sizeof(float));
    result.Set("output", Float32Array::New(env, output_.size(), outputBuffer, 0));
  } else {
    result.Set("output", env.Null());
  }

  if (generationDurationMs_.has_value()) {
    result.Set("generationDurationMs", Number::New(env, generationDurationMs_.value()));
  } else {
    result.Set("generationDurationMs", env.Null());
  }

  result.Set("multiplyDurationMs", Number::New(env, multiplyDurationMs_));
  result.Set("totalDurationMs", Number::New(env, totalDurationMs_));
  result.Set("timingSource", String::New(env, "gpu-timestamp"));

  Object memory = Object::New(env);
  memory.Set("gpuAllocatedBytes", Number::New(env, static_cast<double>(gpuAllocatedBytes_)));
  memory.Set("gpuAllocatedMiB", Number::New(env, ToMiB(gpuAllocatedBytes_)));
  memory.Set("hostAllocatedBytes", Number::New(env, static_cast<double>(hostAllocatedBytes_)));
  memory.Set("hostAllocatedMiB", Number::New(env, ToMiB(hostAllocatedBytes_)));
  result.Set("memoryEstimate", memory);

  return result;
}

double CudaMatrixWorker::ToMiB(size_t bytes) {
  return round((static_cast<double>(bytes) / (1024.0 * 1024.0)) * 1000.0) / 1000.0;
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

Value MultiplyMatrixCuda(const CallbackInfo& info) {
  Env env = info.Env();

  try {
    const CudaMatrixRequest request = ParseCudaMatrixRequest(info);
    auto* worker = new CudaMatrixWorker(env, request);
    Promise promise = worker->GetPromise();
    worker->Queue();
    return promise;
  } catch (const exception& ex) {
    Error::New(env, ex.what()).ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Object Init(Env env, Object exports) {
  exports.Set("multiplyMatrixCuda", Function::New(env, MultiplyMatrixCuda));
  return exports;
}

NODE_API_MODULE(cuda_matrix_addon, Init)



