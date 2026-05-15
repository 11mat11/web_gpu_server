# WebGPU & CUDA Thesis Server

Serwer REST (Fastify, Node.js 25+, ES Modules) do porównywania wydajności **WebGPU** i **CUDA** w ramach pracy magisterskiej.

Dokumentacja interfejsu (Swagger UI): `http://localhost:3000/docs`

## Spis treści
1. [Wymagania](#wymagania)
2. [Konfiguracja Architektury GPU](#konfiguracja-architektury-gpu)
3. [Instalacja](#instalacja)
4. [Budowanie](#budowanie)
5. [Uruchamianie](#uruchamianie)
6. [Środowiska bez wsparcia C++ i CUDA](#środowiska-bez-wsparcia-c-i-cuda)
7. [Zarządzanie Pamięcią i Rozwiązywanie Problemów](#zarządzanie-pamięcią-i-rozwiązywanie-problemów)
8. [Struktura Projektu](#struktura-projektu)
9. [Endpointy](#endpointy)

## Wymagania

- **Node.js 25+**
- Kompatybilny układ graficzny z obsługą **WebGPU** (działający w trybie headless, poprzez backend Vulkan/Metal/DX12).
- **Python 3.x** – absolutnie wymagany przez `node-gyp` do kompilacji modułów natywnych.
- Do pełnej obsługi ścieżek CUDA: **NVIDIA GPU** oraz zainstalowany **NVIDIA CUDA Toolkit**.

## Konfiguracja Architektury GPU

Przed przystąpieniem do kompilacji (`npm run build`) i właściwego operowania na danych, konieczne jest dostosowanie flag sprzętowych pod architekturę docelowej karty graficznej.

W tym celu w pliku `binding.gyp` należy ustawić flagę `-arch=sm_XX` zgodną z układem z którego korzysta aplikacja. Tabela mapowania kluczowych architektur:

| Rodzina GPU | Flaga architektury |
|---|---|
| RTX 40xx (np. 4050) | `sm_89` |
| RTX 30xx | `sm_86` |
| RTX 20xx / GTX 16xx | `sm_75` |
| GTX 10xx | `sm_61` |

Poprawna definicja architektury gwarantuje wygenerowanie optymalnego kodu natywnego (PTX/SASS) dla posiadanego akceleratora.

## Instalacja

Projekt wykorzystuje natywny dodatek C++/CUDA (`binding.gyp` → `cuda_matrix_addon.node`), który kompilowany jest podczas instalacji zależności.

### Instalacja - Linux (Ubuntu)

W przypadku systemów bare-metal (np. Ubuntu Dual-Boot), niezbędne jest uprzednie zainstalowanie własnościowych sterowników NVIDIA oraz narzędzi kompilacyjnych.

1. Zainstaluj wymagane pakiety systemowe w jednym kroku:
```bash
sudo apt-get update && sudo apt-get install -y build-essential python3 make nvidia-cuda-toolkit
```
2. Pobierz zależności projektu (proces ten uruchomi kompilację kodu C++ przez `node-gyp`):
```bash
npm install
```

### Instalacja - Windows

1. Zainstaluj **Visual Studio Build Tools** (wymagane obciążenie robocze: *Desktop development with C++*) w celu zapewnienia kompilatora MSVC niezbędnego dla `node-gyp`.
2. Zainstaluj najnowszą wersję **NVIDIA CUDA Toolkit**. Upewnij się, że polecenie `nvcc` jest dostępne w zmiennych środowiskowych (lub ustawiona jest zmienna `CUDA_PATH`).
3. Pobierz zależności projektu:
```bash
npm install
```

## Budowanie

Po instalacji zależności należy stworzyć plik konfiguracyjny z dostępnego wzorca, a następnie zbudować projekt.

1. Utworzenie pliku środowiskowego `.env`:
   - Linux/macOS: `cp .env.example .env`
   - Windows (PowerShell): `Copy-Item .env.example .env`
2. Generowanie builda:
```bash
npm run build
```
Skrypt uruchamia kompilator TypeScript (`tsc`) i zapewnia przeniesienie zasobów niezbędnych w runtime (shadery **`.wgsl`** i pliki binarne **`.bin`**) ze ścieżek `src/` do `dist/`.

## Uruchamianie

Aplikacja domyślnie startuje na porcie 3000 i udostępnia usługi pod adresem: `http://localhost:3000`

- **Środowisko produkcyjne** (wymaga wcześniejszego wykonania `npm run build`):
```bash
npm start
```
- **Środowisko deweloperskie** (wspierające hot-reload za pośrednictwem biblioteki `tsx`):
```bash
npm run dev
```

## Środowiska bez wsparcia C++ i CUDA

W sytuacji, gdy docelowe środowisko (np. laptop deweloperski) nie posiada karty graficznej z rodziny NVIDIA, albo w systemie brak pakietów C++ / CUDA Toolkit, aplikacja posiada zintegrowane mechanizmy fallbacku w celu bezproblemowej pracy na WebGPU lub CPU.

Działanie mechanizmu auto-detekcji środowiska:
- Jeżeli w systemie wywołanie `nvidia-smi` kończy się błędem,
- lub w repozytorium brak natywnego artefaktu kompilacji (np. `build/Release/cuda_matrix_addon.node`),

to tryb sprzętowy CUDA automatycznie ustawi się w pozycję niedostępną. Endpointy celujące bezpośrednio w kernele CUDA zwrócą status błędu HTTP `400` z odpowiednim komunikatem.

Można również wyłączyć to jawnie z poziomu zmiennych konfiguracyjnych w pliku `.env`:
```ini
CUDA_ENABLED=false
```
Bez CUDA zarówno budowanie (`npm run build`), jak i dewelopment przebiegają standardowym cyklem.

## Zarządzanie Pamięcią i Rozwiązywanie Problemów

- **Zunifikowana pamięć (Unified Memory)**: Natywny kod C++ serwera zarządza pamięcią z wykorzystaniem mechanizmu `cudaMallocManaged`. Rozwiązanie to zabezpiecza aplikację przed nagłymi awariami procesów (crashami) wynikającymi ze zbyt obciążających alokacji. Gdy limit pamięci VRAM zostaje przekroczony, karta graficzna ma prawo wypożyczać na własne potrzeby zasoby ze standardowej pamięci operacyjnej RAM (page migration).
- **Obsługa Timeout Detection and Recovery (TDR / Device Hung)**: W przypadkach skrajnego obciążenia układów graficznych może dochodzić do chwilowych blokad (Device Hung). Jeżeli system bazowy zrzuci proces CUDA na skutek interwencji TDR, logika w serwerze zarejestruje incydent, spróbuje bezpiecznie zwolnić kontekst i dokonać automatycznego ponownego nawiązania komunikacji z układem graficznym.

## Struktura Projektu

Architektura i organizacja kluczowych podkatalogów:

- `src/gpu/shaders/` – Shadery oparte na technologii **WebGPU** (pliki w standardzie `.wgsl`).
- `dist/gpu/shaders/` – Skompilowany i przeniesiony kod shaderów; z tego katalogu Node.js ładuje je podczas pracy środowiska docelowego.
- `src/cuda/` – Moduły C++/CUDA oraz niezbędna warstwa integracji (m.in. wrapper TypeScript, plik `addon.cpp`, kod kernelów `matrix_kernels.cu`).
- `build/Release/` – Rezultat udanej pracy `node-gyp`; z tego miejsca po kompilacji ładowany jest binarny plik `.node` (artefakt wykonawczy).

## Endpointy

Pełny wykaz tras API zaimplementowanych w kontrolerach Fastify:

| Metoda | Ścieżka            | Opis                                |
| ------ | ------------------ | ----------------------------------- |
| GET    | `/health`          | Status serwera                      |
| GET    | `/gpu/info`        | Diagnostyka WebGPU                  |
| GET    | `/gpu/stress`      | Test obciążeniowy GPU               |
| POST   | `/image/filter`    | Filtrowanie obrazów                 |
| POST   | `/matrix/multiply` | Mnożenie macierzy (WebGPU/CUDA/CPU) |
| GET    | `/ai/status`       | Status modeli AI i pamięci          |
| POST   | `/ai/load`         | Ładowanie modeli MLP/CNN            |
| POST   | `/ai/predict/mlp`  | Inferencja MLP                      |
| POST   | `/ai/predict/cnn`  | Inferencja CNN (VGG)                |
| POST   | `/ai/unload`       | Zwalnianie zasobów AI               |
| POST   | `/video/init`      | Inicjalizacja potoku wideo          |
| POST   | `/video/process`   | Przetwarzanie klatek (Downscaling)  |
| POST   | `/video/histogram` | Obliczanie histogramu wideo         |
| POST   | `/video/unload`    | Zamykanie potoku wideo              |
| POST   | `/render`          | Renderowanie scen SDF               |