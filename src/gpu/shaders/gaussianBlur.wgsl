struct ImageSize {
	width: u32,
	height: u32,
	_pad0: u32,
	_pad1: u32,
}

@group(0) @binding(0) var<uniform> size: ImageSize;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

const KERNEL = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

fn unpack(pixel: u32) -> vec4<f32> {
	return
		vec4<f32>(
			f32((pixel) & 0xFFu) / 255.0,
			f32((pixel >> 8) & 0xFFu) / 255.0,
			f32((pixel >> 16) & 0xFFu) / 255.0,
			f32((pixel >> 24) & 0xFFu) / 255.0,
		);
}

fn pack(c: vec4<f32>) -> u32 {
	let r = u32(clamp(c.r, 0.0, 1.0) * 255.0);
	let g = u32(clamp(c.g, 0.0, 1.0) * 255.0);
	let b = u32(clamp(c.b, 0.0, 1.0) * 255.0);
	let a = u32(clamp(c.a, 0.0, 1.0) * 255.0);
	return r | (g << 8) | (b << 16) | (a << 24);
}

fn blurHorizontal(x: u32, y: u32) -> vec4<f32> {
	var acc = vec4<f32>(0.0);
	for (var k = -2; k <= 2; k++) {
		let sx = clamp(i32(x) + k, 0, i32(size.width) - 1);
		acc += unpack(src[y * size.width + u32(sx)]) * KERNEL[k + 2];
	}
	return acc;
}

fn blurVertical(x: u32, y: u32) -> vec4<f32> {
	var acc = vec4<f32>(0.0);
	for (var k = -2; k <= 2; k++) {
		let sy = clamp(i32(y) + k, 0, i32(size.height) - 1);
		acc += unpack(src[u32(sy) * size.width + x]) * KERNEL[k + 2];
	}
	return acc;
}

@compute @workgroup_size(16, 16)
fn mainHorizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
	let x = gid.x;
	let y = gid.y;
	if (x >= size.width || y >= size.height) {
		return;
	}

	dst[y * size.width + x] = pack(blurHorizontal(x, y));
}

@compute @workgroup_size(16, 16)
fn mainVertical(@builtin(global_invocation_id) gid: vec3<u32>) {
	let x = gid.x;
	let y = gid.y;
	if (x >= size.width || y >= size.height) {
		return;
	}

	dst[y * size.width + x] = pack(blurVertical(x, y));
}
