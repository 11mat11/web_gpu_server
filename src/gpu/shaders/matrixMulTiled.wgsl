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

var<workgroup> tileA: array<array<f32, 33>, 32>;
var<workgroup> tileB: array<array<f32, 33>, 32>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let localRow = lid.y * 2u;
  let localCol = lid.x * 2u;
  let row = wid.y * 32u + localRow;
  let col = wid.x * 32u + localCol;
  let row1 = row + 1u;
  let col1 = col + 1u;
  let size = dims.size;

  var sum00: f32 = 0.0;
  var sum01: f32 = 0.0;
  var sum10: f32 = 0.0;
  var sum11: f32 = 0.0;
  let tiles = (size + 31u) / 32u;

  for (var t: u32 = 0u; t < tiles; t++) {
    let aCol = t * 32u + localCol;
    let aCol1 = aCol + 1u;
    let bRow = t * 32u + localRow;
    let bRow1 = bRow + 1u;

    tileA[localRow][localCol] = select(0.0, matA[row * size + aCol], row < size && aCol < size);
    tileA[localRow][localCol + 1u] = select(0.0, matA[row * size + aCol1], row < size && aCol1 < size);
    tileA[localRow + 1u][localCol] = select(0.0, matA[row1 * size + aCol], row1 < size && aCol < size);
    tileA[localRow + 1u][localCol + 1u] = select(0.0, matA[row1 * size + aCol1], row1 < size && aCol1 < size);

    tileB[localRow][localCol] = select(0.0, matB[bRow * size + col], bRow < size && col < size);
    tileB[localRow][localCol + 1u] = select(0.0, matB[bRow * size + col1], bRow < size && col1 < size);
    tileB[localRow + 1u][localCol] = select(0.0, matB[bRow1 * size + col], bRow1 < size && col < size);
    tileB[localRow + 1u][localCol + 1u] = select(0.0, matB[bRow1 * size + col1], bRow1 < size && col1 < size);

    workgroupBarrier();

    for (var k: u32 = 0u; k < 32u; k++) {
      let a0 = tileA[localRow][k];
      let a1 = tileA[localRow + 1u][k];
      let b0 = tileB[k][localCol];
      let b1 = tileB[k][localCol + 1u];
      sum00 += a0 * b0;
      sum01 += a0 * b1;
      sum10 += a1 * b0;
      sum11 += a1 * b1;
    }

    workgroupBarrier();
  }

  if (row < size && col < size) {
    matC[row * size + col] = sum00;
  }
  if (row < size && col1 < size) {
    matC[row * size + col1] = sum01;
  }
  if (row1 < size && col < size) {
    matC[row1 * size + col] = sum10;
  }
  if (row1 < size && col1 < size) {
    matC[row1 * size + col1] = sum11;
  }
}


