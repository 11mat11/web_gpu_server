struct TouchParams {
  stepWords:  u32,
  totalWords: u32,
  seed:       u32,
  _pad:       u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: TouchParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lane = gid.x;
  let wordIndex = lane * params.stepWords;

  if (wordIndex < params.totalWords) {
    // Touch every selected page and change value to force physical backing.
    data[wordIndex] = data[wordIndex] ^ (params.seed + lane);
  }
}

