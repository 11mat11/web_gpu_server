struct CnnParams {
  p0: u32,
  p1: u32,
  p2: u32,
  p3: u32,
  p4: u32,
  p5: u32,
  p6: u32,
  p7: u32,
}

@group(0) @binding(0) var<uniform> params: CnnParams;
@group(0) @binding(1) var<storage, read> inputBuf: array<f32>;
@group(0) @binding(2) var<storage, read> weightsBuf: array<f32>;
@group(0) @binding(3) var<storage, read> biasBuf: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputBuf: array<f32>;

var<workgroup> denseTile: array<f32, 256>;
var<workgroup> densePartials: array<f32, 256>;

@compute @workgroup_size(8, 8, 1)
fn conv2dRelu(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let outX = gid.x;
  let outY = gid.y;
  let outChannel = gid.z;

  let inChannels = params.p0;
  let outChannels = params.p1;
  let inHeight = params.p2;
  let inWidth = params.p3;
  let applyRelu = params.p4;

  if (outX >= inWidth || outY >= inHeight || outChannel >= outChannels) {
    return;
  }

  var acc = biasBuf[outChannel];

  for (var inCh: u32 = 0u; inCh < inChannels; inCh = inCh + 1u) {
    for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
      let inY = i32(outY) + i32(ky) - 1i;
      if (inY < 0i || inY >= i32(inHeight)) {
        continue;
      }

      for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
        let inX = i32(outX) + i32(kx) - 1i;
        if (inX < 0i || inX >= i32(inWidth)) {
          continue;
        }

        let inputIndex = (inCh * inHeight + u32(inY)) * inWidth + u32(inX);
        let weightIndex = (((outChannel * inChannels) + inCh) * 3u + ky) * 3u + kx;
        acc = acc + inputBuf[inputIndex] * weightsBuf[weightIndex];
      }
    }
  }

  if (applyRelu != 0u && acc < 0.0f) {
    acc = 0.0f;
  }

  let outputIndex = (outChannel * inHeight + outY) * inWidth + outX;
  outputBuf[outputIndex] = acc;
}

@compute @workgroup_size(8, 8, 1)
fn maxPool2x2(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let outX = gid.x;
  let outY = gid.y;
  let channel = gid.z;

  let channels = params.p0;
  let inHeight = params.p1;
  let inWidth = params.p2;

  let dummy = weightsBuf[0u] + biasBuf[0u];
  _ = dummy;

  let outHeight = inHeight / 2u;
  let outWidth = inWidth / 2u;

  if (outX >= outWidth || outY >= outHeight || channel >= channels) {
    return;
  }

  let inBaseX = outX * 2u;
  let inBaseY = outY * 2u;

  var maxValue: f32 = -3.402823466e+38f;
  for (var dy: u32 = 0u; dy < 2u; dy = dy + 1u) {
    for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
      let inX = inBaseX + dx;
      let inY = inBaseY + dy;
      let index = (channel * inHeight + inY) * inWidth + inX;
      maxValue = max(maxValue, inputBuf[index]);
    }
  }

  let outIndex = (channel * outHeight + outY) * outWidth + outX;
  outputBuf[outIndex] = maxValue;
}

@compute @workgroup_size(256)
fn denseGemv(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>,
) {
  let outputIndex = workgroupId.x;
  let lane = localId.x;

  let inputSize = params.p0;
  let outputSize = params.p1;
  let applyRelu = params.p2;

  if (outputIndex >= outputSize) {
    return;
  }

  var localSum = 0.0f;
  let tileSize: u32 = 256u;

  var base: u32 = 0u;
  loop {
    if (base >= inputSize) {
      break;
    }

    let inputIndex = base + lane;
    if (inputIndex < inputSize) {
      denseTile[lane] = inputBuf[inputIndex];
    } else {
      denseTile[lane] = 0.0f;
    }
    workgroupBarrier();

    let remaining = inputSize - base;
    let currentTileSize = min(tileSize, remaining);

    var k = lane;
    loop {
      if (k >= currentTileSize) {
        break;
      }
      let weightIndex = (base + k) * outputSize + outputIndex;
      localSum = localSum + denseTile[k] * weightsBuf[weightIndex];
      k = k + tileSize;
    }

    workgroupBarrier();
    base = base + tileSize;
  }

  densePartials[lane] = localSum;
  workgroupBarrier();

  var stride: u32 = tileSize / 2u;
  loop {
    if (stride == 0u) {
      break;
    }

    if (lane < stride) {
      densePartials[lane] = densePartials[lane] + densePartials[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lane == 0u) {
    var value = densePartials[0] + biasBuf[outputIndex];
    if (applyRelu != 0u && value < 0.0f) {
      value = 0.0f;
    }
    outputBuf[outputIndex] = value;
  }
}


