# EPQ benchmarks

Host: Darwin arm64, Python 3.9.6.  Spawn overhead (/usr/bin/true): min 1.93 ms, median 2.31 ms.
Wall = min / median over N runs, end-to-end (spawn + parse + run), stdout to /dev/null.  Self = engine-reported `Execution time` (`-t`), min.  Speedup = baseline wall min / engine wall min.

| benchmark | baseline wall min / med (N) | baseline self | jit wall min / med (N) | jit self | aot wall min / med (N) | aot self | jit speedup (wall) | jit speedup (self) | aot speedup (wall) | aot speedup (self) |
|---|---|---|---|---|---|---|---|---|---|---|
| original/bubblesort | 1.400 s / 1.410 s (7) | 1.395 s | 22.5 ms / 23.4 ms (15) | 18.1 ms | 12.3 ms / 12.6 ms (11) | 7.52 ms | 62.24x | 76.97x | 114.00x | 185.58x |
| original/dijkstra | 630.3 ms / 688.1 ms (7) | 618.7 ms | 12.6 ms / 13.9 ms (15) | 8.07 ms | 12.0 ms / 13.7 ms (10) | 7.24 ms | 49.93x | 76.63x | 52.31x | 85.48x |
| original/eratosthenes | 393.7 ms / 463.4 ms (8) | 388.3 ms | 9.48 ms / 10.1 ms (15) | 5.49 ms | 10.1 ms / 11.1 ms (7) | 6.03 ms | 41.52x | 70.74x | 39.07x | 64.40x |
| original/fibonacci | 2.78 ms / 2.99 ms (15) | 0.02 ms | 2.92 ms / 3.15 ms (15) | 0.01 ms | 3.24 ms / 3.39 ms (15) | 0.01 ms | 0.95x | 2.11x | 0.86x | 2.38x |
| original/heapsort | 16.7 ms / 17.1 ms (15) | 13.6 ms | 4.40 ms / 4.58 ms (15) | 0.26 ms | 4.22 ms / 4.82 ms (14) | 0.23 ms | 3.79x | 51.54x | 3.95x | 58.15x |
| original/mergesort | 14.2 ms / 15.1 ms (15) | 10.6 ms | 4.37 ms / 4.67 ms (15) | 0.41 ms | 4.17 ms / 4.69 ms (12) | 0.36 ms | 3.26x | 25.75x | 3.42x | 29.45x |
| extra/bintree | 382.7 ms / 447.8 ms (9) | 358.7 ms | 17.1 ms / 19.5 ms (15) | 12.8 ms | 16.4 ms / 18.8 ms (7) | 12.0 ms | 22.43x | 27.93x | 23.29x | 29.78x |
| extra/bubblesort_big | 641.1 ms / 697.1 ms (7) | 636.1 ms | 13.2 ms / 13.4 ms (15) | 8.82 ms | 10.0 ms / 10.4 ms (7) | 5.44 ms | 48.66x | 72.13x | 64.00x | 116.95x |
| extra/collatz | 753.4 ms / 856.1 ms (7) | 750.1 ms | 18.2 ms / 18.5 ms (15) | 14.5 ms | 20.0 ms / 20.2 ms (7) | 16.0 ms | 41.44x | 51.79x | 37.64x | 47.02x |
| extra/dijkstra_big | 661.3 ms / 678.6 ms (7) | 641.9 ms | 19.0 ms / 19.2 ms (15) | 14.3 ms | 16.2 ms / 16.6 ms (13) | 11.4 ms | 34.84x | 44.80x | 40.70x | 56.48x |
| extra/dynamic | 362.4 ms / 393.1 ms (10) | 338.1 ms | 14.4 ms / 15.2 ms (15) | 9.92 ms | 14.9 ms / 15.2 ms (11) | 9.84 ms | 25.16x | 34.09x | 24.33x | 34.34x |
| extra/fib_rec | 430.1 ms / 433.0 ms (9) | 427.1 ms | 3.15 ms / 3.84 ms (15) | 0.03 ms | 3.41 ms / 3.92 ms (15) | 0.02 ms | 136.48x | 15252.00x | 126.23x | 26691.00x |
| extra/heapsort_big | 622.1 ms / 643.2 ms (7) | 617.8 ms | 15.8 ms / 16.3 ms (15) | 11.6 ms | 14.6 ms / 14.9 ms (15) | 9.90 ms | 39.33x | 53.39x | 42.59x | 62.40x |
| extra/large_output | 243.9 ms / 266.5 ms (15) | 236.6 ms | 9.89 ms / 10.1 ms (15) | 5.19 ms | 10.1 ms / 10.7 ms (13) | 5.22 ms | 24.66x | 45.57x | 24.04x | 45.36x |
| extra/lcg_hash | 733.6 ms / 750.3 ms (7) | 728.7 ms | 19.3 ms / 20.1 ms (15) | 14.3 ms | 17.3 ms / 17.7 ms (13) | 11.7 ms | 37.94x | 50.89x | 42.40x | 62.11x |
| extra/matmul | 528.2 ms / 547.7 ms (7) | 523.2 ms | 8.91 ms / 9.59 ms (15) | 4.37 ms | 9.43 ms / 9.68 ms (15) | 4.25 ms | 59.26x | 119.62x | 56.03x | 123.08x |
| extra/mergesort_big | 629.7 ms / 661.7 ms (7) | 605.4 ms | 26.1 ms / 26.5 ms (15) | 20.8 ms | 24.2 ms / 24.7 ms (13) | 18.5 ms | 24.17x | 29.07x | 26.06x | 32.73x |
| extra/nqueens | 1.110 s / 1.130 s (7) | 1.105 s | 44.5 ms / 44.6 ms (15) | 39.6 ms | 44.0 ms / 44.5 ms (12) | 38.9 ms | 24.96x | 27.89x | 25.21x | 28.42x |
| extra/pairsort | 494.3 ms / 516.0 ms (7) | 472.6 ms | 25.4 ms / 27.1 ms (15) | 20.0 ms | 23.1 ms / 24.1 ms (13) | 17.2 ms | 19.43x | 23.67x | 21.37x | 27.40x |
| extra/primes_trial | 831.7 ms / 859.9 ms (7) | 826.9 ms | 13.3 ms / 13.9 ms (15) | 8.94 ms | 14.4 ms / 14.7 ms (14) | 9.59 ms | 62.30x | 92.47x | 57.64x | 86.26x |
| extra/quicksort | 718.4 ms / 736.4 ms (7) | 713.3 ms | 20.8 ms / 21.0 ms (15) | 16.1 ms | 20.2 ms / 20.6 ms (13) | 15.2 ms | 34.58x | 44.20x | 35.49x | 46.98x |
| extra/sieve_big | 1.149 s / 1.174 s (7) | 1.142 s | 17.8 ms / 19.1 ms (15) | 13.2 ms | 19.3 ms / 20.0 ms (15) | 14.4 ms | 64.71x | 86.74x | 59.58x | 79.25x |
| extra/tak | 200.4 ms / 207.9 ms (15) | 195.9 ms | 14.5 ms / 14.9 ms (15) | 10.2 ms | 13.8 ms / 14.0 ms (15) | 9.01 ms | 13.82x | 19.18x | 14.52x | 21.74x |
| extra/toploop | 590.6 ms / 618.4 ms (7) | 585.9 ms | 17.3 ms / 17.9 ms (15) | 12.4 ms | 18.6 ms / 19.7 ms (14) | 13.7 ms | 34.21x | 47.23x | 31.72x | 42.63x |

| engine | geomean speedup (wall min) | geomean speedup (self) | verified |
|---|---|---|---|
| jit | 26.46x | 53.16x | 24/24 |
| aot | 27.54x | 60.50x | 24/24 |

Geometric mean of baseline wall min over 24 benchmarks: 343.4 ms
