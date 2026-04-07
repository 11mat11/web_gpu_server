// Matrix Multiplication shader (C = A * B)
// Each workgroup thread computes one element of the result matrix.

struct Dimensions {
  M: u32,  // rows of A
  N: u32,  // cols of B
  K: u32,  // cols of A / rows of B
}

@group(0) @binding(0) var<uniform>  dims : Dimensions;
@group(0) @binding(1) var<storage, read>       matA : array<f32>;
@group(0) @binding(2) var<storage, read>       matB : array<f32>;
@group(0) @binding(3) var<storage, read_write> matC : array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;

  if (row >= dims.M || col >= dims.N) { return; }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.K; k++) {
    sum += matA[row * dims.K + k] * matB[k * dims.N + col];
  }

  matC[row * dims.N + col] = sum;
}
