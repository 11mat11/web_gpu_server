# Notatki robocze - AI Manager i API (MLP + CNN)

To jest notatka techniczna do pracy magisterskiej. Opisuje warstwe orkiestracji AI po stronie serwera: zarzadzanie modelami, backendami, cyklem zycia i kontraktami API.

## 1) Rola `AiManager`

`AiManager.ts` jest centralnym orchestrator'em dla dwoch modeli:

- `mlp` (MNIST 128x128),
- `cnn` (CIFAR-10 Advanced VGG).

Kazdy model ma niezalezny stan backendow:

- `webgpu`,
- `cuda`.

Dzieki temu mozliwe jest ladowanie/zwalnianie selektywne (model i backend), co jest istotne np. na hostach bez NVIDIA.

## 2) Cykle zycia modeli

`AiManager` utrzymuje globalny stan lifecycle:

- `idle`,
- `loading`,
- `unloading`.

To zapobiega konfliktom rownoleglych operacji `load/unload`.

Publiczne operacje:

- `loadModel(options?)`,
- `unloadModel(options?)`,
- `getStatus()`,
- `predictMlp(backend, input)`,
- `predictCnn(backend, input)`.

## 3) Sciezki wag

Wagi sa czytane bezposrednio przez `AiManager`:

- MLP: `src/ai/mega_mnist_weights.bin`,
- CNN: `src/ai/cifar10_weights.bin`.

Podczas `load` manager sprawdza zgodnosc rozmiaru pliku z `mlpLayout.totalWeightCount` lub `cnnLayout.totalWeightCount`.

## 4) Multi-backend i fallback

Przy `load`:

- backendy wybierane sa flagami `webgpu`/`cuda`,
- model wybierany jest przez `model: 'mlp' | 'cnn'`,
- gdy `model` nie jest podany, manager probuje zaladowac oba modele,
- gdy backend CUDA jest niedostepny, status zawiera `reason` i mozliwy jest tryb WebGPU-only.

## 5) Predykcja i kontrakt wynikow

### MLP

`predictMlp` oczekuje wejscia `16384` float.

Zwraca:

- `prediction`,
- `probabilities`,
- `gpuDurationMs`,
- `totalDurationMs`,
- `timingSource`.

### CNN

`predictCnn` oczekuje wejscia `49152` float (`3x128x128`, CHW).

Dodatkowo zwraca:

- `predictionLabel` (etykieta CIFAR-10),
- `memoryEstimate` modelu CNN.

Softmax i argmax sa liczone po stronie Node.js na podstawie logitow zwroconych z backendu.

## 6) Etykiety CIFAR-10

Mapowanie etykiet jest trzymane w `AiManager`:

- `airplane`, `automobile`, `bird`, `cat`, `deer`,
- `dog`, `frog`, `horse`, `ship`, `truck`.

## 7) Memory estimate

`AiManager` raportuje pamiec na dwoch poziomach:

- per model (`mlp`, `cnn`),
- globalnie dla calego pipeline AI.

Dla kazdego modelu sa pola:

- `hostAllocatedBytes` / `MiB`,
- `totalGpuAllocatedBytes` / `MiB`,
- breakdown `webgpu` i `cuda`.

To pozwala analizowac koszt ladowania modeli osobno i lacznie.

## 8) Warstwa HTTP

Kontrolery i routing sa w:

- `ai.controller.ts`,
- `ai.route.ts`.

Endpointy:

- `GET /ai/status`,
- `POST /ai/load`,
- `POST /ai/unload`,
- `POST /ai/predict/mlp`,
- `POST /ai/predict/cnn`.

Swagger zawiera schemy request/response, przyklady backendow i kody bledow.

## 9) Obsluga bledow

Wewnatrz managera i kontrolerow stosowany jest `AiManagerError` (`code`, `statusCode`).

Przyklady kodow:

- `ai_busy`,
- `invalid_input`,
- `invalid_weights_file`,
- `model_not_loaded`,
- `backend_unavailable`,
- `ai_load_failed`, `ai_predict_failed`, `ai_unload_failed`.

To upraszcza diagnostyke i pozwala utrzymac spojny kontrakt API.

