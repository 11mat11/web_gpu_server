struct GemvParams {
	inputSize: u32,
	outputSize: u32,
	applyRelu: u32,
	_pad: u32,
}

@group(0) @binding(0) var<uniform> params: GemvParams;
@group(0) @binding(1) var<storage, read> inputVec: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputVec: array<f32>;

var<workgroup> inputTile: array<f32, 256>;
var<workgroup> partialSums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
	@builtin(workgroup_id) workgroupId: vec3<u32>,
	@builtin(local_invocation_id) localId: vec3<u32>,
) {
	let outputIndex = workgroupId.x;
	let lane = localId.x;

	if (outputIndex >= params.outputSize) {
		return;
	}

	var localSum = 0.0;
	let tileSize: u32 = 256u;

	var base: u32 = 0u;
	loop {
		if (base >= params.inputSize) {
			break;
		}

		let inputIndex = base + lane;
		if (inputIndex < params.inputSize) {
			inputTile[lane] = inputVec[inputIndex];
		} else {
			inputTile[lane] = 0.0;
		}
		workgroupBarrier();

		let remaining = params.inputSize - base;
		let currentTileSize = min(tileSize, remaining);

		var k = lane;
		loop {
			if (k >= currentTileSize) {
				break;
			}

			let weightIndex = (base + k) * params.outputSize + outputIndex;
			localSum = localSum + inputTile[k] * weights[weightIndex];
			k = k + tileSize;
		}

		workgroupBarrier();
		base = base + tileSize;
	}

	partialSums[lane] = localSum;
	workgroupBarrier();

	var stride: u32 = tileSize / 2u;
	loop {
		if (stride == 0u) {
			break;
		}

		if (lane < stride) {
			partialSums[lane] = partialSums[lane] + partialSums[lane + stride];
		}

		workgroupBarrier();
		stride = stride / 2u;
	}

	if (lane == 0u) {
		var value = partialSums[0] + bias[outputIndex];
		if (params.applyRelu != 0u && value < 0.0) {
			value = 0.0;
		}
		outputVec[outputIndex] = value;
	}
}
