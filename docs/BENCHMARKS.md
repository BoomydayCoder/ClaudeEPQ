# EPQ benchmarks

Host: Darwin arm64, Python 3.9.6.  Spawn overhead (/usr/bin/true): min 1.46 ms, median 1.61 ms.
Wall = min / median over N runs, end-to-end (spawn + parse + run), stdout to /dev/null.  Self = engine-reported `Execution time` (`-t`), min.  Speedup = baseline wall min / engine wall min.

| benchmark | baseline wall min / med (N) | baseline self | jit wall min / med (N) | jit self | aot wall min / med (N) | aot self | jit speedup (wall) | jit speedup (self) | aot speedup (wall) | aot speedup (self) |
|---|---|---|---|---|---|---|---|---|---|---|
| original/bubblesort | 1.259 s / 1.274 s (7) | 1.256 s | 20.9 ms / 21.2 ms (15) | 16.9 ms | 11.5 ms / 11.7 ms (12) | 7.11 ms | 60.21x | 74.17x | 109.59x | 176.54x |
| original/dijkstra | 583.2 ms / 615.7 ms (7) | 572.4 ms | 12.0 ms / 12.5 ms (15) | 7.52 ms | 11.6 ms / 12.0 ms (13) | 6.79 ms | 48.64x | 76.13x | 50.20x | 84.36x |
| original/eratosthenes | 421.0 ms / 460.3 ms (9) | 415.8 ms | 9.71 ms / 11.7 ms (15) | 5.53 ms | 10.3 ms / 13.2 ms (15) | 5.68 ms | 43.36x | 75.24x | 41.04x | 73.25x |
| original/fibonacci | 3.65 ms / 4.13 ms (15) | 0.02 ms | 4.14 ms / 4.96 ms (15) | 0.01 ms | 4.55 ms / 5.72 ms (13) | 0.01 ms | 0.88x | 2.20x | 0.80x | 1.83x |
| original/heapsort | 20.8 ms / 22.6 ms (15) | 16.9 ms | 5.62 ms / 6.22 ms (15) | 0.32 ms | 6.07 ms / 6.80 ms (12) | 0.30 ms | 3.71x | 52.20x | 3.43x | 57.14x |
| original/mergesort | 17.2 ms / 18.5 ms (15) | 12.9 ms | 5.23 ms / 6.37 ms (15) | 0.48 ms | 5.45 ms / 6.64 ms (11) | 0.44 ms | 3.30x | 26.99x | 3.17x | 29.19x |
| extra/bintree | 392.6 ms / 420.6 ms (8) | 368.5 ms | 17.4 ms / 19.2 ms (15) | 13.0 ms | 17.1 ms / 18.9 ms (12) | 12.3 ms | 22.61x | 28.43x | 22.90x | 30.06x |
| extra/bubblesort_big | 640.7 ms / 683.0 ms (7) | 636.7 ms | 12.9 ms / 16.4 ms (15) | 8.35 ms | 9.96 ms / 13.5 ms (15) | 5.18 ms | 49.52x | 76.29x | 64.32x | 122.96x |
| extra/collatz | 800.7 ms / 818.5 ms (7) | 797.2 ms | 17.7 ms / 17.9 ms (15) | 13.7 ms | 19.2 ms / 19.5 ms (11) | 15.0 ms | 45.35x | 58.29x | 41.62x | 53.04x |
| extra/dijkstra_big | 628.6 ms / 645.3 ms (7) | 612.8 ms | 18.6 ms / 18.9 ms (15) | 13.5 ms | 15.9 ms / 16.2 ms (14) | 10.6 ms | 33.86x | 45.41x | 39.64x | 58.02x |
| extra/dynamic | 345.7 ms / 392.4 ms (10) | 323.1 ms | 14.5 ms / 18.2 ms (15) | 9.45 ms | 14.7 ms / 16.9 ms (11) | 9.43 ms | 23.78x | 34.19x | 23.51x | 34.25x |
| extra/fib_rec | 410.2 ms / 446.7 ms (8) | 407.0 ms | 3.84 ms / 4.33 ms (15) | 0.03 ms | 3.97 ms / 4.55 ms (15) | 0.02 ms | 106.72x | 15652.77x | 103.22x | 23939.53x |
| extra/heapsort_big | 595.6 ms / 638.5 ms (7) | 592.0 ms | 15.3 ms / 15.6 ms (15) | 11.0 ms | 14.2 ms / 14.4 ms (14) | 9.40 ms | 39.02x | 53.92x | 42.05x | 62.96x |
| extra/large_output | 232.1 ms / 252.8 ms (15) | 225.2 ms | 9.77 ms / 10.1 ms (15) | 4.91 ms | 10.3 ms / 10.6 ms (14) | 5.07 ms | 23.74x | 45.84x | 22.47x | 44.47x |
| extra/lcg_hash | 622.3 ms / 705.6 ms (7) | 618.3 ms | 17.6 ms / 19.2 ms (15) | 13.0 ms | 15.7 ms / 17.3 ms (7) | 10.7 ms | 35.27x | 47.60x | 39.52x | 57.96x |
| extra/matmul | 530.0 ms / 542.8 ms (7) | 525.6 ms | 9.67 ms / 10.1 ms (15) | 4.39 ms | 9.65 ms / 10.1 ms (7) | 4.25 ms | 54.79x | 119.61x | 54.94x | 123.72x |
| extra/mergesort_big | 630.6 ms / 696.2 ms (7) | 604.8 ms | 25.5 ms / 27.3 ms (15) | 19.7 ms | 24.7 ms / 25.9 ms (7) | 18.8 ms | 24.71x | 30.65x | 25.53x | 32.25x |
| extra/nqueens | 1.089 s / 1.193 s (7) | 1.085 s | 42.5 ms / 45.0 ms (15) | 37.5 ms | 42.0 ms / 44.6 ms (13) | 36.7 ms | 25.65x | 28.94x | 25.91x | 29.55x |
| extra/pairsort | 487.0 ms / 492.2 ms (8) | 464.2 ms | 24.7 ms / 25.6 ms (15) | 19.2 ms | 22.1 ms / 22.4 ms (13) | 16.2 ms | 19.68x | 24.13x | 22.05x | 28.63x |
| extra/primes_trial | 807.9 ms / 828.3 ms (7) | 803.2 ms | 13.1 ms / 14.0 ms (15) | 8.45 ms | 14.1 ms / 14.8 ms (14) | 9.09 ms | 61.71x | 95.09x | 57.40x | 88.36x |
| extra/quicksort | 693.6 ms / 713.6 ms (7) | 688.9 ms | 20.4 ms / 21.4 ms (15) | 15.3 ms | 19.7 ms / 21.0 ms (14) | 14.4 ms | 33.98x | 45.03x | 35.19x | 48.00x |
| extra/sieve_big | 1.094 s / 1.124 s (7) | 1.087 s | 17.6 ms / 18.7 ms (15) | 12.6 ms | 18.7 ms / 19.3 ms (12) | 13.6 ms | 62.34x | 85.99x | 58.46x | 79.63x |
| extra/tak | 190.1 ms / 198.6 ms (15) | 186.0 ms | 14.4 ms / 15.1 ms (15) | 9.67 ms | 13.5 ms / 13.7 ms (15) | 8.70 ms | 13.22x | 19.24x | 14.04x | 21.38x |
| extra/toploop | 567.7 ms / 584.0 ms (7) | 563.0 ms | 16.3 ms / 17.1 ms (15) | 11.7 ms | 18.2 ms / 19.0 ms (14) | 13.0 ms | 34.73x | 48.09x | 31.14x | 43.31x |

| engine | geomean speedup (wall min) | geomean speedup (self) | verified |
|---|---|---|---|
| jit | 25.86x | 54.21x | 24/24 |
| aot | 26.74x | 60.29x | 24/24 |

Geometric mean of baseline wall min over 24 benchmarks: 343.6 ms
