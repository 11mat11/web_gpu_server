#pragma once

#include <cuda_runtime.h>
#include <napi.h>

#include <optional>
#include <vector>

#define CUDA_CHECK_THROW(call)                                                      \
    do {                                                                            \
	const cudaError_t cudaStatus = (call);                                      \
	if (cudaStatus != cudaSuccess) {                                            \
	    throw std::runtime_error(std::string("CUDA error in ") + #call + ": " + \
				     cudaGetErrorString(cudaStatus));               \
	}                                                                           \
    } while (0)

struct CudaMatrixRequest {
    int size = 0;
    bool optimized = false;
    bool readback = true;
    bool randomInput = true;
    float randomMin = 0.0F;
    float randomMax = 1.0F;
    uint32_t randomSeed = 0U;
    std::vector<float> matrixA;
    std::vector<float> matrixB;
};

class CudaMatrixWorker final : public Napi::AsyncWorker {
   public:
    CudaMatrixWorker(Napi::Env env, CudaMatrixRequest request);
    ~CudaMatrixWorker() override;

    [[nodiscard]] Napi::Promise GetPromise() const;

   protected:
    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& error) override;

   private:
    void Cleanup();
    [[nodiscard]] Napi::Value BuildResult(Napi::Env env) const;
    static double ToMiB(size_t bytes);

    Napi::Promise::Deferred deferred_;
    CudaMatrixRequest request_;

    float* dMatrixA_ = nullptr;
    float* dMatrixB_ = nullptr;
    float* dMatrixC_ = nullptr;
    cudaEvent_t generationStartEvent_ = nullptr;
    cudaEvent_t generationStopEvent_ = nullptr;
    cudaEvent_t multiplyStartEvent_ = nullptr;
    cudaEvent_t multiplyStopEvent_ = nullptr;

    std::vector<float> output_;
    std::optional<double> generationDurationMs_;
    double multiplyDurationMs_ = 0.0;
    double backendDurationMs_ = 0.0;

    size_t gpuAllocatedBytes_ = 0;
    size_t hostAllocatedBytes_ = 0;
};

CudaMatrixRequest ParseCudaMatrixRequest(const Napi::CallbackInfo& info);
