#pragma once

#include <napi.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix_kernels.h"
using namespace Napi;
using namespace std;
#define CUDA_CHECK_THROW(call)                                                                                          \
  do {                                                                                                                  \
    const cudaError_t cudaStatus = (call);                                                                             \
    if (cudaStatus != cudaSuccess) {                                                                                   \
      throw runtime_error(string("CUDA error in ") + #call + ": " + cudaGetErrorString(cudaStatus));      \
    }                                                                                                                   \
  } while (0)

struct CudaMatrixRequest {
  int size = 0;
  bool optimized = false;
  bool readback = true;
  bool randomInput = true;
  float randomMin = 0.0F;
  float randomMax = 1.0F;
  uint32_t randomSeed = 0U;
  vector<float> matrixA;
  vector<float> matrixB;
};

class CudaMatrixWorker final : public AsyncWorker {
public:
  CudaMatrixWorker(Env env, const CudaMatrixRequest& request);
  ~CudaMatrixWorker() override;

  Promise GetPromise() const;

  void Execute() override;
  void OnOK() override;
  void OnError(const Error& error) override;

private:
  void Cleanup();
  Value BuildResult(Env env) const;
  static double ToMiB(size_t bytes);

  Promise::Deferred deferred_;
  CudaMatrixRequest request_;

  float* dMatrixA_ = nullptr;
  float* dMatrixB_ = nullptr;
  float* dMatrixC_ = nullptr;
  cudaEvent_t generationStartEvent_ = nullptr;
  cudaEvent_t generationStopEvent_ = nullptr;
  cudaEvent_t multiplyStartEvent_ = nullptr;
  cudaEvent_t multiplyStopEvent_ = nullptr;

  vector<float> output_;
  optional<double> generationDurationMs_;
  double multiplyDurationMs_ = 0.0;
  double totalDurationMs_ = 0.0;

  size_t gpuAllocatedBytes_ = 0;
  size_t hostAllocatedBytes_ = 0;
};

CudaMatrixRequest ParseCudaMatrixRequest(const CallbackInfo& info);



