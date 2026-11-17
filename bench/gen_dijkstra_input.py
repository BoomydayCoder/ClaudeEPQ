#!/usr/bin/env python3
"""Deterministically generate the input for bench/original/dijkstra.epq:
a connected random undirected weighted graph.

  bench/gen_dijkstra_input.py [--vertices 20000] [--edges 100000] [--seed 12345] > dijkstra.in

Format (what dijkstra.epq reads): V E, then E lines "u v cost".
Connectivity: the first V-1 edges form a random spanning tree (vertex i>0 is
joined to a random earlier vertex); the rest are random pairs.  Costs 1..1000.
Uses its own LCG so the output never depends on the Python version.
"""
import argparse
import sys


def lcg(seed):
    s = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield s >> 33


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertices", type=int, default=20000)
    ap.add_argument("--edges", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    V, E = a.vertices, a.edges
    if E < V - 1:
        sys.exit("need at least V-1 edges for connectivity")
    r = lcg(a.seed)
    out = ["%d %d" % (V, E)]
    for v in range(1, V):
        u = next(r) % v
        out.append("%d %d %d" % (u, v, 1 + next(r) % 1000))
    for _ in range(E - (V - 1)):
        u = next(r) % V
        v = next(r) % V
        out.append("%d %d %d" % (u, v, 1 + next(r) % 1000))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
