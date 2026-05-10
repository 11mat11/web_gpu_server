struct RandomFillParams {
	size: u32,
	seed: u32,
	minValue: f32,
	maxValue: f32,
}

@group(0) @binding(0) var<uniform> params: RandomFillParams;
@group(0) @binding(1) var<storage, read_write> matA: array<f32>;
@group(0) @binding(2) var<storage, read_write> matB: array<f32>;

fn hash32(x: u32) -> u32 {
	var v = x * 747796405u + 2891336453u;
	v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
	v = (v >> 22u) ^ v;
	return v;
}

fn random01(state: u32) -> f32 {
	return f32(hash32(state)) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(
	@builtin(global_invocation_id) gid: vec3<u32>,
	@builtin(num_workgroups) numWorkgroups: vec3<u32>,
) {
	let strideX = numWorkgroups.x * 256u;
	let strideXY = strideX * numWorkgroups.y;
	let idx = gid.x + gid.y * strideX + gid.z * strideXY;
	let count = params.size * params.size;

	if (idx >= count) {
		return;
	}

	let span = params.maxValue - params.minValue;
	let baseSeed = params.seed ^ idx;
	let valueA = params.minValue + random01(baseSeed ^ 0x9E3779B9u) * span;
	let valueB = params.minValue + random01(baseSeed ^ 0x85EBCA6Bu) * span;

	matA[idx] = valueA;
	matB[idx] = valueB;
}
