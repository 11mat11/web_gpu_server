{
  "variables": {
    "cuda_fast_math_flag%": "<!(node -p \"process.env.CUDA_FAST_MATH === '0' ? '' : '-use_fast_math'\")"
  },
  "targets": [
    {
      "target_name": "cuda_matrix_addon",
      "sources": [ "src/cuda/addon.cpp" ],
      "dependencies": [ "<!(node -p \"require('node-addon-api').gyp\")" ],
      "defines": [ "NAPI_CPP_EXCEPTIONS" ],
      "conditions": [
        ["OS=='win'", {
          "variables": {
            "cuda_path%": "<!(node -p \"process.env.CUDA_PATH || 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'\")"
          },
          "include_dirs": [
            "<!@(node -p \"require('node-addon-api').include\")",
            "<(cuda_path)/include"
          ],
          "actions": [
            {
              "action_name": "compile_cuda_kernels",
              "inputs": ["src/cuda/matrix_kernels.cu"],
              "outputs": ["<(INTERMEDIATE_DIR)/matrix_kernels.obj"],
              "action": [
                "<(cuda_path)\\bin\\nvcc.exe",
                "-c", "src\\cuda\\matrix_kernels.cu",
                "-o", "<(INTERMEDIATE_DIR)\\matrix_kernels.obj",
                "-O3", "<(cuda_fast_math_flag)",
                "-arch=sm_89", "-Xptxas=-O3,-v", "-Xcompiler", "/O2", "-Xcompiler", "/EHsc", "-std=c++17",
                "-I<(module_root_dir)\\node_modules\\node-addon-api"
              ]
            }
          ],
          "libraries": [ "<(cuda_path)/lib/x64/cudart.lib" ]
        }],
        ["OS!='win'", {
          "variables": {
            "cuda_root_unix%": "<!(node -p \"process.env.CUDA_PATH || '/usr'\")"
          },
          "include_dirs": [
            "<!@(node -p \"require('node-addon-api').include\")",
            "<(cuda_root_unix)/include"
          ],
          "actions": [
            {
              "action_name": "compile_cuda_kernels",
              "inputs": ["src/cuda/matrix_kernels.cu"],
              "outputs": ["<(INTERMEDIATE_DIR)/matrix_kernels.o"],
              "action": [
                "<(cuda_root_unix)/bin/nvcc",
                "-c", "src/cuda/matrix_kernels.cu",
                "-o", "<(INTERMEDIATE_DIR)/matrix_kernels.o",
                "-O3", "<(cuda_fast_math_flag)",
                "-arch=sm_89", "-Xptxas=-O3,-v",
                "-Xcompiler", "-O3,-fPIC,-fexceptions",
                "-std=c++17",
                "-I<!@(node -p \"require('node-addon-api').include\")"
              ]
            }
          ],
          "libraries": [
            "<(INTERMEDIATE_DIR)/matrix_kernels.o",
            "-L/usr/lib/x86_64-linux-gnu",
            "-lcudart"
          ],
          "cflags_cc": [ "-O3", "-std=c++17", "-fexceptions" ],
          "cflags!": [ "-fno-exceptions" ],
          "cflags_cc!": [ "-fno-exceptions" ]
        }]
      ]
    }
  ]
}
