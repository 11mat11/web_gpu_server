struct Dimensions {
  size: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<uniform> dims : Dimensions;
@group(0) @binding(1) var<storage, read> matA : array<f32>;
@group(0) @binding(2) var<storage, read> matB : array<f32>;
@group(0) @binding(3) var<storage, read_write> matC : array<f32>;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let size = dims.size;
  let isActive = row < size && col < size;

  var sum: f32 = 0.0;
  let tiles = (size + 15u) / 16u;

  for (var t: u32 = 0u; t < tiles; t++) {
    let aCol = t * 16u + lid.x;
    let bRow = t * 16u + lid.y;

    if (row < size && aCol < size) {
      tileA[lid.y][lid.x] = matA[row * size + aCol];
    } else {
      tileA[lid.y][lid.x] = 0.0;
    }

    if (bRow < size && col < size) {
      tileB[lid.y][lid.x] = matB[bRow * size + col];
    } else {
      tileB[lid.y][lid.x] = 0.0;
    }

    workgroupBarrier();

    if (isActive) {
      for (var k: u32 = 0u; k < 16u; k++) {
        sum += tileA[lid.y][k] * tileB[k][lid.x];
      }
    }

    workgroupBarrier();
  }

  if (isActive) {
    matC[row * size + col] = sum;
  }
}


