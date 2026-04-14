# Notatki robocze - CUDA (porownanie z WebGPU)

To jest notatka przypominajaca do pracy magisterskiej, opisujaca jak dziala moja sciezka CUDA od warstwy Node.js do kerneli i jak interpretowac czasy oraz GFLOPS. To nie jest gotowy tekst do rozdzialu, tylko techniczny opis kodu w formie ciaglego streszczenia.

`cudaBackend.ts` jest punktem wejscia po stronie TypeScript. Ten plik buduje obiekt parametrow, dba o sensowne domyslne wartosci (np. seed, zakres losowania, readback), a potem wywoluje funkcje natywna `multiplyMatrixCuda` z dodatku `.node`. Kluczowe jest to, ze `cudaBackend.ts` nie liczy nic na GPU samodzielnie - on tylko tlumaczy dane JavaScript/TypeScript na kontrakt z addonem C++ i zwraca wynik w tym samym ksztalcie, jakiego oczekuje warstwa HTTP.

`addon.cpp` jest faktycznym mostem Node.js <-> CUDA. Ten plik korzysta z N-API i eksportuje pojedyncza funkcje `multiplyMatrixCuda` do swiata JavaScript. Wejscie z JS jest parsowane do struktury `CudaMatrixRequest`: rozmiar, tryb random/custom, flaga optimized, opcjonalny readback i dane macierzy przy trybie custom. Jezeli wejscie jest bledne, addon zwraca blad JS. Jezeli wejscie jest poprawne, tworzony jest obiekt worker i zadanie trafia do kolejki asynchronicznej.

Najwazniejszym elementem w `addon.cpp` jest klasa robocza dzialajaca asynchronicznie (na bazie `AsyncWorker`). Dzieki temu event loop Node.js nie jest blokowany, mimo ze w tle wykonuje sie `cudaMalloc`, uruchamianie kerneli, synchronizacja eventow i ewentualne kopiowanie wyniku na hosta. Schemat jest prosty: JavaScript dostaje Promise, worker wykonuje obliczenia w `Execute`, a po sukcesie wynik jest budowany jako obiekt JS z polami `output`, `generationDurationMs`, `multiplyDurationMs`, `totalDurationMs`, `timingSource` i `memoryEstimate`.

`cuda_worker.h` definiuje ten kontrakt niskopoziomowy: strukture zadania, pola do uchwytow GPU, eventow czasowych i wyniku, a takze makro sprawdzania bledow CUDA. Ta warstwa jest wazna, bo porzadkuje odpowiedzialnosci: parser argumentow i logika API sa po stronie addona, a stan wykonywania i cleanup zasobow sa zamkniete w workerze. W praktyce to ogranicza ryzyko wyciekow i ulatwia jednoznaczna interpretacje, kiedy mierzony jest czas i jakie dane sa odczytywane do odpowiedzi.

`matrix_kernels.h` to cienki interfejs pomiedzy kodem hosta C++ i plikiem `.cu`. Zawiera deklaracje launcherow kerneli: losowania, mnozenia naiwniego i mnozenia tiled. Dzieki temu `addon.cpp` nie musi znac szczegolow implementacji kerneli - tylko podpis funkcji i parametry.

`matrix_kernels.cu` zawiera implementacje obliczen na GPU. W wariancie random jest kernel oparty o haszowanie (`hash32`) i mapowanie do zakresu `[minValue, maxValue]`, co daje szybkie i deterministyczne generowanie danych na urzadzeniu. Wariantu mnozenia sa dwa: naive i tiled. Naive liczy jeden element wyniku na watek i wykonuje petle po `k`, czyli klasyczny GEMM O(N^3), dobry jako baseline. Tiled wykorzystuje pamiec wspoldzielona (`__shared__`), laduje dane kafelkami i liczy 2x2 wyniki na watek (`sum00`, `sum01`, `sum10`, `sum11`), co zmniejsza liczbe odczytow z global memory i poprawia lokalnosc dostepu. W obu wariantach sa warunki brzegowe dla macierzy, ktorych rozmiar nie musi byc idealna wielokrotnoscia bloku.

Pomiar czasu CUDA jest realizowany bezposrednio w `addon.cpp` przez eventy CUDA (`cudaEventRecord`, `cudaEventElapsedTime`). Dla trybu random mierzony jest osobno etap generacji danych i osobno etap mnozenia. Dla trybu custom etap generacji jest pusty (`generationDurationMs = null`), bo dane przychodza z hosta. `multiplyDurationMs` to czas samego kernela mnozenia, a `totalDurationMs` to suma etapu generacji i mnozenia. W odpowiedzi ustawiane jest `timingSource = 'gpu-timestamp'`, bo metryki pochodza z eventow GPU, a nie z zegara CPU.

`smoke.ts` jest prostym testem dymnym tej integracji. Uruchamia wywolanie CUDA z losowymi danymi, wypisuje czasy i rozmiar wyniku, wiec sluzy do szybkiej weryfikacji: czy addon sie laduje, czy kernel startuje i czy wynik wraca do Node.js. To nie jest benchmark naukowy, tylko kontrola poprawnosci pipeline'u.

`binding.gyp` jest plikiem budowania dodatku natywnego i to on technicznie spina Node.js z CUDA podczas kompilacji. Definiuje target `cuda_matrix_addon`, wskazuje zrodlo C++ (`addon.cpp`), include path do `node-addon-api` i CUDA oraz zaleznosci wymagane przez node-gyp. Najwazniejsze jest to, ze kompilacja kerneli `.cu` jest zrobiona jako osobna akcja `compile_cuda_kernels`: `nvcc` tworzy plik obiektowy, ktory potem jest linkowany razem z addonem.

W `binding.gyp` sa osobne warunki dla Windows i systemow Unix. Na Windows akcja uruchamia `nvcc.exe`, generuje `matrix_kernels.obj` i linkuje `cudart.lib`. Na Unix kompilowany jest `matrix_kernels.o`, a przy linkowaniu dolaczane sa `-lcudart` i sciezka do biblioteki CUDA. To sprawia, ze ten sam kod zrodlowy ma rozne szczegoly narzedziowe zalezne od platformy, ale ten sam kontrakt API po stronie Node.js.

Istotne sa tez flagi kompilacji w `binding.gyp`: optymalizacja `-O3`, architektura `-arch=sm_89`, opcjonalny `-use_fast_math` sterowany zmienna srodowiskowa, oraz ustawienia kompilatora hosta C++17. Te flagi maja bezposredni wplyw na wydajnosc i porownywalnosc wynikow, wiec przy porownaniu z WebGPU trzeba je jawnie odnotowac. Szczegolnie `sm_89` oznacza, ze binarka jest profilowana pod konkretna klase GPU, a fast-math moze zmieniac kompromis miedzy dokladnoscia i szybkoscia.

`matrix.ts` (warstwa route) oblicza GFLOPS jednakowo dla backendow, z klasycznego wzoru `2 * N^3 / t`. Czas `t` bierze z `multiplyDurationMs` (a gdy brak - z fallbackowego czasu serwera), wiec interpretacja GFLOPS zalezy od tego, czy porownujemy taki sam rodzaj czasu. W praktyce to oznacza, ze porownanie CUDA vs WebGPU powinno byc robione na tej samej definicji metryki: ten sam rozmiar, ten sam tryb danych, ten sam wariant kernela i ta sama polityka readback.

Najkrotsze podsumowanie calej sciezki jest takie: Node.js przyjmuje parametry, `cudaBackend.ts` pakuje je do kontraktu, `addon.cpp` i `cuda_worker.h` uruchamiaja asynchroniczna prace natywna, `matrix_kernels.cu` wykonuje losowanie i/lub GEMM na GPU, a wynik wraca do JavaScript jako Promise z metrykami czasu i pamieci. `binding.gyp` nie wykonuje obliczen, ale jest krytyczny, bo bez niego ten most kompilacyjny i linkowanie CUDA z addonem Node.js w ogole by nie dzialaly.

## Dodatkowy blok - skad bierze sie `memoryEstimate`

W CUDA te wartosci sa liczone bezposrednio w `addon.cpp`, a nie pobierane z zewnetrznego profilera. Punktem startowym jest `matrixBytes = N * N * sizeof(float)`. Na tej podstawie `gpuAllocatedBytes` ustawiane jest jako `matrixBytes * 3`, bo zawsze alokowane sa trzy bufory urzadzenia: `dMatrixA`, `dMatrixB`, `dMatrixC`.

`hostAllocatedBytes` zalezy od trybu wejscia i readback: dla `custom` host trzyma dwie macierze wejsciowe (`A` i `B`), wiec dochodzi `matrixBytes * 2`; dla `random` ten skladnik jest liczony jako 0, bo dane powstaja na GPU. Do tego dochodzi opcjonalnie `matrixBytes` na wynik, jesli wlaczony jest `readback`. W skrocie: `hostAllocatedBytes = (randomInput ? 0 : 2 * matrixBytes) + (readback ? matrixBytes : 0)`.

`gpuAllocatedMiB` i `hostAllocatedMiB` powstaja przez przeliczenie bajtow na MiB (`/ 1024 / 1024`) i zaokraglenie do 3 miejsc po przecinku. To znaczy, ze metryka pamieci w odpowiedzi API jest estymacja oparta o jawnie tworzone bufory, a nie pomiarem calkowitego zuzycia pamieci procesu/GPU na poziomie sterownika.

