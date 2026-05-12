# WebGPU Thesis Server

Serwer REST (Fastify, Node.js 25+, ES Modules) do porównywania wydajności **WebGPU** i **CUDA** w ramach pracy magisterskiej.

Swagger UI: `http://localhost:3000/docs`

## Wymagania

- **Node.js 25+**
- GPU/sterowniki z obsługą **WebGPU** (headless; zwykle przez backend Vulkan/Metal/DX12 zależnie od platformy)
- (Opcjonalnie, dla CUDA) **NVIDIA GPU + CUDA Toolkit**
- (Opcjonalnie, dla budowania natywnego dodatku na Windows) **Visual Studio Build Tools** (C/C++)

## Instalacja

1) Instalacja zależności:

```bash
npm install
```

2) Utworzenie pliku środowiskowego `.env` na podstawie przykładu:

Linux/macOS:

```bash
cp .env.example .env
```

Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

3) Build i uruchomienie:

```bash
npm run build
npm run start
```

Domyślnie serwer startuje pod: `http://localhost:3000`

## 🚀 Instalacja i Uruchomienie (Windows & Native Linux)

Ta sekcja dotyczy środowisk, w których **natywny kod C++/CUDA jest kompilowany lokalnie** (w locie) przez **node-gyp**.

Projekt zawiera natywny addon (`binding.gyp` → `cuda_matrix_addon.node`). W praktyce oznacza to, że **`npm install` może uruchomić kompilację C++/CUDA**. Jeżeli w systemie brakuje wymaganych narzędzi (kompilatora C++ i/lub CUDA Toolkit), instalacja zakończy się typowym „czerwonym” błędem kompilacji.

### Wymagania wstępne (dla obu systemów)

- **Node.js 25+**
- **Python 3.x** — **wymóg absolutny** dla `node-gyp` (bez Pythona kompilacja addonu nie ruszy).

Weryfikacja wersji:

```bash
node -v
npm -v
python --version
```

> Na części dystrybucji Linux polecenie może nazywać się `python3` zamiast `python`.

### Konfiguracja — Windows

#### Krok 1: Visual Studio Build Tools

1. Zainstaluj **Visual Studio Build Tools**.
2. W instalatorze zaznacz workload: **Desktop development with C++**.

To zapewnia kompilator MSVC oraz narzędzia wymagane przez `node-gyp`.

#### Krok 2: NVIDIA CUDA Toolkit

1. Zainstaluj najnowszy **NVIDIA CUDA Toolkit** ze strony producenta.
2. Upewnij się, że `nvcc` jest dostępny w systemie (lub że ustawiona jest zmienna `CUDA_PATH`).

Szybka diagnostyka:

```bash
nvidia-smi
nvcc --version
```

### Konfiguracja — Natywny Linux (np. Ubuntu Dual-Boot)

Instrukcja dotyczy systemów **bare-metal (nie WSL2)**, gdzie sterowniki GPU i CUDA działają natywnie.

#### Instalacja kompilatora i Pythona

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3
```

#### Krok kluczowy: własnościowe sterowniki NVIDIA

Zanim zainstalujesz CUDA, zainstaluj **własnościowe sterowniki NVIDIA**:

- Ubuntu: *Software & Updates* → *Additional Drivers* → wybierz sterownik NVIDIA (proprietary) → zastosuj zmiany → restart.

Po restarcie sprawdź:

```bash
nvidia-smi
```

#### Instalacja CUDA Toolkit

```bash
sudo apt-get install -y nvidia-cuda-toolkit
```

Weryfikacja:

```bash
nvcc --version
```

### Budowanie projektu

0) Skopiuj konfigurację środowiskową (jeśli jeszcze nie masz `.env`):

Linux/macOS:

```bash
cp .env.example .env
```

Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

1) Instalacja zależności (oraz kompilacja natywnego addonu przez `node-gyp`):

```bash
npm install
```

2) Build TypeScript + kopiowanie assetów do `dist/`:

```bash
npm run build
```

Skrypt `build` wykonuje:

- kompilację TypeScript (`tsc`),
- kopiowanie shaderów **`.wgsl`** oraz plików binarnych **`.bin`** z `src/` do `dist/`.

### Uruchamianie

Wersja produkcyjna (uruchamia pliki z `dist/`):

```bash
npm start
```

Wersja developerska (watch / hot reload):

```bash
npm run dev
```

> Uwaga: nawet jeśli CUDA nie jest dostępne w runtime, serwer nadal może działać na WebGPU/CPU dzięki auto-detekcji. Jeśli chcesz wymusić brak ścieżek CUDA, ustaw `CUDA_ENABLED=false` w `.env`.

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

Nawet bez CUDA możesz uruchamiać serwer zarówno w trybie produkcyjnym, jak i developerskim:

```bash
npm start
npm run dev
```

## Struktura projektu (najważniejsze katalogi)

- `src/gpu/shaders/` – shadery **WebGPU** w formacie `.wgsl`
- `dist/gpu/shaders/` – shadery **muszą znaleźć się po buildzie** (runtime Node.js ładuje je z katalogu `dist/`)
- `src/cuda/` – kod C++/CUDA i warstwa integracji (m.in. `addon.cpp`, `matrix_kernels.cu`, wrapper TS)
- `build/Release/` – artefakty natywnego addonu (`cuda_matrix_addon.node`) po kompilacji

## Skrypty

Poniżej kluczowe skrypty (zgodne z `package.json`):

- `dev`: `tsx watch src/index.ts`
- `build`: `tsc && copyfiles -u 1 "src/**/*.wgsl" "src/**/*.bin" dist/`
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
