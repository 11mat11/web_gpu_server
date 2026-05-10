# WebGPU Thesis Server

Serwer REST (Fastify, Node.js 22, ES Modules) do porównywania wydajności **WebGPU** i **CUDA** w ramach pracy magisterskiej.

Swagger UI: `http://localhost:3000/docs`

## Wymagania

- **Node.js 22+**
- GPU/sterowniki z obsługą **WebGPU** (headless; zwykle przez backend Vulkan/Metal/DX12 zależnie od platformy)
- (Opcjonalnie, dla CUDA) **NVIDIA GPU + CUDA Toolkit**
- (Opcjonalnie, dla budowania natywnego dodatku na Windows) **Visual Studio Build Tools** (C/C++)

## Instalacja

```bash
npm install
npm run build
npm run start
```

Domyślnie serwer startuje pod: `http://localhost:3000`

## Uruchamianie bez CUDA (auto-detekcja)

Backend CUDA jest **opcjonalny**. Serwer potrafi działać dalej na WebGPU/CPU nawet jeśli CUDA nie jest dostępne.

Mechanizm auto-detekcji CUDA:

- sprawdza, czy da się uruchomić `nvidia-smi` (czyli czy jest wykrywalna karta NVIDIA), oraz
- czy istnieje zbudowany natywny addon: `build/Release/cuda_matrix_addon.node` (lub `build/Debug/...`).

Jeśli którykolwiek warunek nie jest spełniony, CUDA jest oznaczone jako niedostępne i endpointy wymagające CUDA powinny zwracać błąd walidacyjny/`400` (zależnie od trasy).

Wymuszone wyłączenie CUDA:

- ustaw w `.env`:

```ini
CUDA_ENABLED=false
```

## Struktura projektu (najważniejsze katalogi)

- `src/gpu/shaders/` – shadery **WebGPU** w formacie `.wgsl`
- `dist/gpu/shaders/` – shadery **muszą znaleźć się po buildzie** (runtime Node.js ładuje je z katalogu `dist/`)
- `src/cuda/` – kod C++/CUDA i warstwa integracji (m.in. `addon.cpp`, `matrix_kernels.cu`, wrapper TS)
- `build/Release/` – artefakty natywnego addonu (`cuda_matrix_addon.node`) po kompilacji

## Skrypty

Poniżej kluczowe skrypty (zgodne z `package.json`):

- `dev`: `tsx watch src/index.ts`
- `build`: `tsc && copyfiles -u 1 "src/gpu/shaders/*.wgsl" dist/`
- `start`: `node dist/index.js`

Jeśli potrzebujesz uruchomić serwer bez watch, możesz też odpalić jednorazowo: `tsx src/index.ts`.

## Endpointy

| Metoda | Ścieżka | Opis |
| ------ | ------- | ---- |
| GET | `/health` | Status serwera |
| GET | `/gpu/info` | Diagnostyka WebGPU |
| GET | `/gpu/stress` | Test obciążeniowy GPU |
| POST | `/image/filter` | Filtrowanie obrazów |
| POST | `/matrix/multiply` | Mnożenie macierzy (WebGPU/CUDA/CPU) |
| GET | `/ai/status` | Status modeli AI i pamięci |
| POST | `/ai/load` | Ładowanie modeli MLP/CNN |
| POST | `/ai/predict/mlp` | Inferencja MLP |
| POST | `/ai/predict/cnn` | Inferencja CNN (VGG) |
| POST | `/ai/unload` | Zwalnianie zasobów AI |
| POST | `/video/init` | Inicjalizacja potoku wideo |
| POST | `/video/process` | Przetwarzanie klatek (Downscaling) |
| POST | `/video/histogram` | Obliczanie histogramu wideo |
| POST | `/video/unload` | Zamykanie potoku wideo |
| POST | `/render` | Renderowanie scen SDF |
