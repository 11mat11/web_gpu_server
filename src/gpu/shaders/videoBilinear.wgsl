struct VideoParams {
  srcWidth: u32,
  srcHeight: u32,
  dstWidth: u32,
  dstHeight: u32,
}

@group(0) @binding(0) var<uniform> params: VideoParams;
@group(0) @binding(1) var<storage, read> srcRgba: array<u32>;
@group(0) @binding(2) var<storage, read_write> dstRgba: array<u32>;

fn unpackRgba(pixel: u32) -> vec4<f32> {
  let r: f32 = f32(pixel & 0xFFu);
  let g: f32 = f32((pixel >> 8u) & 0xFFu);
  let b: f32 = f32((pixel >> 16u) & 0xFFu);
  let a: f32 = f32((pixel >> 24u) & 0xFFu);
  return vec4<f32>(r, g, b, a);
}

fn packRgba(value: vec4<f32>) -> u32 {
  let r: u32 = u32(clamp(round(value.x), 0.0f, 255.0f));
  let g: u32 = u32(clamp(round(value.y), 0.0f, 255.0f));
  let b: u32 = u32(clamp(round(value.z), 0.0f, 255.0f));
  let a: u32 = u32(clamp(round(value.w), 0.0f, 255.0f));
  return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>
) {
  let outX: u32 = gid.x;
  let outY: u32 = gid.y;

  if (outX >= params.dstWidth || outY >= params.dstHeight) {
    return;
  }

  let scaleX: f32 = f32(params.srcWidth) / f32(params.dstWidth);
  let scaleY: f32 = f32(params.srcHeight) / f32(params.dstHeight);

  let srcX: f32 = (f32(outX) + 0.5f) * scaleX - 0.5f;
  let srcY: f32 = (f32(outY) + 0.5f) * scaleY - 0.5f;

  var x0: i32 = i32(floor(srcX));
  var y0: i32 = i32(floor(srcY));

  x0 = clamp(x0, 0, i32(params.srcWidth) - 1);
  y0 = clamp(y0, 0, i32(params.srcHeight) - 1);

  let x1: i32 = min(x0 + 1, i32(params.srcWidth) - 1);
  let y1: i32 = min(y0 + 1, i32(params.srcHeight) - 1);

  let fx: f32 = srcX - f32(x0);
  let fy: f32 = srcY - f32(y0);

  let srcWidthI32: i32 = i32(params.srcWidth);
  let idx00: u32 = u32(y0 * srcWidthI32 + x0);
  let idx10: u32 = u32(y0 * srcWidthI32 + x1);
  let idx01: u32 = u32(y1 * srcWidthI32 + x0);
  let idx11: u32 = u32(y1 * srcWidthI32 + x1);

  let c00: vec4<f32> = unpackRgba(srcRgba[idx00]);
  let c10: vec4<f32> = unpackRgba(srcRgba[idx10]);
  let c01: vec4<f32> = unpackRgba(srcRgba[idx01]);
  let c11: vec4<f32> = unpackRgba(srcRgba[idx11]);

  let top: vec4<f32> = c00 * (1.0f - fx) + c10 * fx;
  let bottom: vec4<f32> = c01 * (1.0f - fx) + c11 * fx;
  let blended: vec4<f32> = top * (1.0f - fy) + bottom * fy;

  let dstIndex: u32 = outY * params.dstWidth + outX;
  dstRgba[dstIndex] = packRgba(blended);
}