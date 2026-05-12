@group(0) @binding(0) var<storage, read> inputRgba: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let pixelIndex: u32 = gid.x;
	if (pixelIndex >= arrayLength(&inputRgba)) {
		return;
	}

	let pixel: u32 = inputRgba[pixelIndex];
	let r: u32 = pixel & 0xFFu;
	let g: u32 = (pixel >> 8u) & 0xFFu;
	let b: u32 = (pixel >> 16u) & 0xFFu;

	atomicAdd(&histogram[r], 1u);
	atomicAdd(&histogram[256u + g], 1u);
	atomicAdd(&histogram[512u + b], 1u);
}
