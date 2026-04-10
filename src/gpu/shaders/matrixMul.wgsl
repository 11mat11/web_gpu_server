// Zunifikowana struktura wyrównana do 16 bajtów (std140)
struct Dimensions {
  size: u32,
  _pad1: u32, // Padding dla pełnej zgodności na każdym sprzęcie
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<uniform>  dims : Dimensions;
@group(0) @binding(1) var<storage, read>       matA : array<f32>;
@group(0) @binding(2) var<storage, read>       matB : array<f32>;
@group(0) @binding(3) var<storage, read_write> matC : array<f32>;

// Zmieniamy na 16x16 (256 wątków) dla lepszej utylizacji SM na karcie
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  let size = dims.size;

  // Bound check
  if (row >= size || col >= size) {
    return;
  }

  var sum: f32 = 0.0;
  // Wskazówka dla GPGPU: Pętla odczytuje matA poprawnie (coalesced),
  // ale odczytuje matB "skacząc" po kolumnach (stride = size),
  // co w tym naiwnym algorytmie niszczy cache L1. Będziemy to optymalizować później!
  for (var k: u32 = 0u; k < size; k++) {
    sum += matA[row * size + k] * matB[k * size + col];
  }

  matC[row * size + col] = sum;
}