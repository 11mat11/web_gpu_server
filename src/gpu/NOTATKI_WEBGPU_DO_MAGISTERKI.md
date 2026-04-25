# Notatki robocze - WebGPU (matrix + AI pipeline)

To jest notatka techniczna do pracy magisterskiej. Opisuje aktualna sciezke WebGPU w serwerze Node.js: baseline matrix multiply oraz pipeline AI (MLP i Advanced VGG CNN).

## 1) Inicjalizacja i runtime

`device.ts` odpowiada za:

- bootstrap WebGPU w Node,
- cache adaptera i urzadzenia,
- negocjacje feature `timestamp-query`,
- serializacje limitow i wymagane limity urzadzenia.

To ten plik decyduje, czy metryki czasu beda mogly pochodzic z hardware timestampow GPU.

## 2) Matrix baseline

Sciezka matrix (`matrixMul.ts` + `matrixMul.wgsl` + `matrixMulTiled.wgsl`) pozostaje punktem odniesienia dla porownan CUDA/WebGPU.

- wariant naive: prosty GEMM,
- wariant tiled: workgroup memory i lepsza lokalnosc.

Wynik i metryki sa zwracane przez API z `timingSource`:

- `gpu-timestamp` gdy dostepny `timestamp-query`,
- `cpu-clock` jako fallback.

## 3) MLP inferencja WebGPU

`mlp-runner.ts` realizuje persistent model buffers i 3-pass inferencje:

1. GEMV + ReLU (input -> hidden1),
2. GEMV + ReLU (hidden1 -> hidden2),
3. GEMV bez ReLU (hidden2 -> logits).

Softmax liczony jest po stronie Node.js (w `AiManager`), po odczycie 10 logitow.

## 4) CNN inferencja WebGPU - Advanced VGG

`cnn-runner.ts` + `shaders/cnn.wgsl` realizuja nowa architekture Advanced VGG dla CIFAR-10.

### Wejscie i wyjscie

- wejscie: `3 x 128 x 128` (CHW, 49152 float),
- wyjscie: 10 logitow.

### Layout wag i offsety

Wagi sa ladowane z jednego pliku `.bin` wedlug stalej kolejnosci segmentow:

- conv1, conv2, conv3, conv4,
- dense1 (flatten 8192 -> 256),
- dense2 (256 -> 10),
- odpowiednie biasy.

`cnnLayout.totalWeightCount` musi byc zgodny z realnym plikiem wag.

### Pipeline wykonania (10 passow)

1. Conv1 + ReLU -> `[32, 128, 128]`
2. MaxPool1 -> `[32, 64, 64]`
3. Conv2 + ReLU -> `[64, 64, 64]`
4. MaxPool2 -> `[64, 32, 32]`
5. Conv3 + ReLU -> `[128, 32, 32]`
6. MaxPool3 -> `[128, 16, 16]`
7. Conv4 + ReLU -> `[128, 16, 16]`
8. MaxPool4 -> `[128, 8, 8]`
9. Dense1 + ReLU (`8192 -> 256`)
10. Dense2 bez ReLU (`256 -> 10`)

Wszystkie mapy cech zostaja na GPU. Odczyt CPU jest tylko po ostatnim passie.

## 5) `layout: 'auto'` i stabilnosc BindGroupLayout

W `cnn.wgsl` kernel `maxPool2x2` ma jawne dummy read z `weightsBuf` i `biasBuf`. Powod:

- przy `layout: 'auto'` kompilator moze usuwac nieuzywane bindingi,
- wtedy layout potoku pool nie zgadza sie z bind group tworzona w TS,
- skutkuje to bledami `Invalid ComputePipeline` / `Invalid BindGroupLayout`.

Dummy read utrzymuje stabilny kontrakt bindingow miedzy shaderem i `cnn-runner.ts`.

## 6) WGSL strict typing

W `cnn.wgsl` trzeba pilnowac scislego typowania:

- literały `i32` z `1i`, `0i` w arytmetyce indeksow,
- literały `u32` z `u` w petlach i dzieleniu,
- literały `f32` z `0.0f` / `-1000000.0f`.

To ogranicza ryzyko bledow kompilacji shaderow i kaskady bledow walidacji WebGPU.

## 7) Metryki czasu i pamieci

`cnn-runner.ts` korzysta z query set timestampow i sumuje czasy etapow inferencji. Gdy `timestamp-query` nie jest dostepne, fallback to `cpu-clock`.

`memoryEstimate` to estymacja oparta o jawne bufory:

- osobno dla modelu,
- agregowana wyzej przez `AiManager`.

Te wartosci sa deterministiczne (z kodu), nie sa odczytem z profilerow runtime.
