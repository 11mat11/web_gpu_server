struct Shape {
	shapeType: f32,
	x: f32,
	y: f32,
	size: f32,
	depth: f32,
	r: f32,
	g: f32,
	b: f32,
	a: f32,
	pad0: f32,
	pad1: f32,
	pad2: f32,
}

struct RenderParams {
	width: f32,
	height: f32,
	pad0: f32,
	pad1: f32,
}

struct VertexOut {
	@builtin(position) position: vec4<f32>,
	@location(0) localPos: vec2<f32>,
	@location(1) shapeType: f32,
	@location(2) size: f32,
	@location(3) color: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> shapes: array<Shape>;
@group(0) @binding(1) var<uniform> params: RenderParams;

fn sdfCircle(p: vec2<f32>, radius: f32) -> f32 {
	return length(p) - radius;
}

fn sdfBox(p: vec2<f32>, radius: f32) -> f32 {
	let d = abs(p) - vec2<f32>(radius, radius);
	let outside = length(max(d, vec2<f32>(0.0f, 0.0f)));
	let inside = min(max(d.x, d.y), 0.0f);
	return outside + inside;
}

fn sdfTriangle(pIn: vec2<f32>, radius: f32) -> f32 {
	let k = 1.7320508f;
	var p = pIn;
	p.x = abs(p.x) - radius;
	p.y = p.y + radius / k;
	if (p.x + k * p.y > 0.0f) {
		p = vec2<f32>(p.x - k * p.y, -k * p.x - p.y) * 0.5f;
	}
	p.x = p.x - clamp(p.x, -2.0f * radius, 0.0f);
	return -length(p) * sign(p.y);
}

@vertex
fn vsMain(
	@location(0) position: vec2<f32>,
	@builtin(instance_index) instanceIndex: u32,
) -> VertexOut {
	let shape = shapes[instanceIndex];
	let local = position * shape.size;
	let world = vec2<f32>(shape.x, shape.y) + local;
	let ndcX = (world.x / params.width) * 2.0f - 1.0f;
	let ndcY = 1.0f - (world.y / params.height) * 2.0f;

	var out: VertexOut;
	out.position = vec4<f32>(ndcX, ndcY, shape.depth, 1.0f);
	out.localPos = local;
	out.shapeType = shape.shapeType;
	out.size = shape.size;
	out.color = vec4<f32>(shape.r, shape.g, shape.b, shape.a);
	return out;
}

@fragment
fn fsMain(input: VertexOut) -> @location(0) vec4<f32> {
	let typeId = u32(input.shapeType + 0.5f);
	var dist = 1.0f;

	if (typeId == 0u) {
		dist = sdfCircle(input.localPos, input.size);
	} else if (typeId == 1u) {
		dist = sdfBox(input.localPos, input.size);
	} else {
		dist = sdfTriangle(input.localPos, input.size);
	}

	if (dist > 0.0f) {
		discard;
	}

	return input.color;
}
