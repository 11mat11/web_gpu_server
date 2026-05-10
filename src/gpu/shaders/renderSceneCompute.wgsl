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
  width: u32,
  height: u32,
  shapeCount: u32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> shapes: array<Shape>;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

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

fn packRgba(color: vec4<f32>) -> u32 {
  let clamped = clamp(color, vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f), vec4<f32>(1.0f, 1.0f, 1.0f, 1.0f));
  let r = u32(round(clamped.x * 255.0f));
  let g = u32(round(clamped.y * 255.0f));
  let b = u32(round(clamped.z * 255.0f));
  let a = u32(round(clamped.w * 255.0f));
  return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let pixelX = f32(gid.x) + 0.5f;
  let pixelY = f32(gid.y) + 0.5f;

  var bestDepth = 2.0f;
  var bestColor = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

  var i = 0u;
    loop {
      if (i >= params.shapeCount) {
        break;
      }

      let typeId = u32(shapes[i].shapeType + 0.5f);
      let local = vec2<f32>(pixelX - shapes[i].x, pixelY - shapes[i].y);
      let size = shapes[i].size;
      let depth = shapes[i].depth;

      var dist = 1.0f;
      if (typeId == 0u) {
        dist = sdfCircle(local, size);
      } else if (typeId == 1u) {
        dist = sdfBox(local, size);
      } else {
        dist = sdfTriangle(local, size);
      }

      if (dist <= 0.0f && depth < bestDepth) {
        bestDepth = depth;
        bestColor = vec4<f32>(shapes[i].r, shapes[i].g, shapes[i].b, shapes[i].a);
      }

      i = i + 1u;
    }

  let outIndex = gid.y * params.width + gid.x;
  output[outIndex] = packRgba(bestColor);
}

