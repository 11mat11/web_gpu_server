# Notatki robocze - CUDA (matrix + AI pipeline)

To jest notatka techniczna do pracy magisterskiej. Opisuje aktualny stan kodu CUDA po stronie serwera Node.js: osobno baseline matrix multiply i osobno pipeline AI (MLP + Advanced VGG CNN dla CIFAR-10).

## 1) Warstwa integracyjna Node.js -> addon

Punktem wejscia po stronie TypeScript jest `cudaBackend.ts`. Ten plik nie liczy na GPU samodzielnie - tylko:

- sprawdza dostepnosc runtime CUDA,
- laduje addon `.node`,
- mapuje kontrakty JS/TS na wywolania N-API,
- zwraca wyniki w formacie oczekiwanym przez route i `AiManager`.

W `addon.cpp` znajduje sie most N-API. Dla matrix baseline eksportowana jest sciezka `multiplyMatrixCuda`. Dla AI eksportowane sa asynchroniczne metody:

- `loadModel` / `predict` / `unloadModel` (MLP),
- `loadCnnModel` / `predictCnn` / `unloadCnnModel` (CNN).

Kazda dluzsza operacja uruchamiana jest jako `AsyncWorker`, wiec event loop Node.js nie jest blokowany.

## 2) Matrix baseline (porownania CUDA vs WebGPU)

Baseline matrix multiply sluzy jako punkt odniesienia metodologii i metryk. Kerneli sa dwa:

- naive GEMM,
- tiled GEMM z `__shared__`.

Dodatkowo jest kernel losujacy dane na GPU. Czasy mierzone sa eventami CUDA (`cudaEventRecord`, `cudaEventElapsedTime`) i raportowane jako `timingSource = 'gpu-timestamp'`.

## 3) MLP inferencja (MNIST 128x128)

MLP jest ladowany do stanu globalnego addonu i trzyma persistent VRAM:

- wagi i biasy trzech warstw,
- bufory aktywacji (`input`, `h1`, `h2`, `out`).

Predykcja MLP wykonuje 3 uruchomienia GEMV (z ReLU dla warstw ukrytych) i jeden odczyt 10 logitow na hosta na koncu.

## 4) CNN inferencja - Advanced VGG

Aktualny pipeline CNN to wersja Advanced VGG dla wejscia `[3, 128, 128]` i nowego pliku wag `src/ai/cifar10_weights.bin`.

### Layout wag (Float32, kolejnosc w pliku)

1. `conv1.weight` `[32, 3, 3, 3]`
2. `conv1.bias` `[32]`
3. `conv2.weight` `[64, 32, 3, 3]`
4. `conv2.bias` `[64]`
5. `conv3.weight` `[128, 64, 3, 3]`
6. `conv3.bias` `[128]`
7. `conv4.weight` `[128, 128, 3, 3]`
8. `conv4.bias` `[128]`
9. `fc1.weight` `[8192, 256]`
10. `fc1.bias` `[256]`
11. `fc2.weight` `[256, 10]`
12. `fc2.bias` `[10]`

### Wykonanie inferencji CNN (10 krokow)

1. Conv1 + ReLU -> `[32, 128, 128]`
2. MaxPool1 -> `[32, 64, 64]`
3. Conv2 + ReLU -> `[64, 64, 64]`
4. MaxPool2 -> `[64, 32, 32]`
5. Conv3 + ReLU -> `[128, 32, 32]`
6. MaxPool3 -> `[128, 16, 16]`
7. Conv4 + ReLU -> `[128, 16, 16]`
8. MaxPool4 -> `[128, 8, 8]`
9. GEMV fc1 + ReLU (`8192 -> 256`)
10. GEMV fc2 (`256 -> 10`, bez ReLU)

Po drodze nie ma transferu GPU->CPU. Odczyt na hosta jest tylko raz, po kroku 10 (10 logitow).

### Pointers VRAM i cleanup

Stan CNN w `addon.cpp` trzyma osobne pointery dla:

- wszystkich wag/biasow (`conv1..4`, `dense1..2`),
- wszystkich aktywacji posrednich (`conv1Out`, `pool1Out`, ..., `pool4Out`, `dense1Out`, `out`).

`FreeCnnModelLocked()` musi zwalniac wszystkie pointery przez `cudaFree`, aby uniknac wyciekow przy wielokrotnym `load/unload`.

## 5) Kerneli CUDA i pomiar czasu

W `matrix_kernels.cu` sa kerneli wspolne dla matrix i AI:

- `launchCnnConv2dKernel` (3x3, stride 1, padding 1, opcjonalny ReLU),
- `launchCnnMaxPool2x2Kernel` (2x2, stride 2),
- `launchMlpGemvKernel` (dla MLP i warstw dense w CNN).

Dla CNN czas GPU liczony jest jako suma czasow wszystkich etapow (pary eventow start/stop). To daje metryke porownywalna z WebGPU `gpu-timestamp`.

## 6) Memory estimate

`memoryEstimate` zwracany z addonu to estymacja oparta o jawne alokacje:

- `gpuAllocatedBytes`: suma buforow w VRAM,
- `hostAllocatedBytes`: glownie rozmiar wag przekazanych z hosta przy `load`.

Wartosci `MiB` sa przeliczane jako `bytes / (1024 * 1024)` i zaokraglane do 3 miejsc po przecinku.

## 7) Build i narzedzia

`binding.gyp` kompiluje addon C++ i osobno kerneli `.cu` przez `nvcc`, a potem linkuje je w jedna binarke `.node`. To jest krytyczny element mostu Node.js <-> CUDA.
