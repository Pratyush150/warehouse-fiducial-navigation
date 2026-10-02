"""Warehouse floor: 1 m grid, shelf blocks, aisles, floor fiducials, A* planning."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

CELL = 1.0
SHELF_HALF = 0.45          # shelf footprint is 0.9 m square inside its cell
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass
class Layout:
    shelf: np.ndarray       # bool (nx, ny); True = shelf cell
    markers: np.ndarray     # (M, 2) marker positions (cell centres), id = row index

    @property
    def nx(self):
        return self.shelf.shape[0]

    @property
    def ny(self):
        return self.shelf.shape[1]

    def free(self, c):
        i, j = c
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.shelf[i, j]

    def stations(self):
        """Pick faces: free cells next to a shelf, with the direction to face it."""
        out = []
        for i in range(self.nx):
            for j in range(self.ny):
                if self.shelf[i, j]:
                    continue
                for k, (di, dj) in enumerate(DIRS):
                    a, b = i + di, j + dj
                    if 0 <= a < self.nx and 0 <= b < self.ny and self.shelf[a, b]:
                        out.append(((i, j), k * np.pi / 2))
                        break
        return out

    def clearance(self, xy):
        """Distance from a point to the nearest shelf footprint or the outer wall."""
        x, y = xy
        d = min(x + 0.5, y + 0.5, self.nx - 0.5 - x, self.ny - 0.5 - y)
        i0, j0 = int(round(x)), int(round(y))
        for i in range(i0 - 1, i0 + 2):
            for j in range(j0 - 1, j0 + 2):
                if 0 <= i < self.nx and 0 <= j < self.ny and self.shelf[i, j]:
                    dx = max(abs(x - i) - SHELF_HALF, 0.0)
                    dy = max(abs(y - j) - SHELF_HALF, 0.0)
                    d = min(d, np.hypot(dx, dy) if (dx or dy) else -min(SHELF_HALF - abs(x - i),
                                                                         SHELF_HALF - abs(y - j)))
        return d

    def astar(self, start, goal, turn_cost=0.5):
        """4-connected A* over (cell, heading); turning costs like moving half a cell."""
        if not (self.free(start) and self.free(goal)):
            return None
        h = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])  # noqa: E731
        openq = [(h(start), 0.0, start, -1, None)]
        best, parent = {}, {}
        while openq:
            f, g, c, d, par = heapq.heappop(openq)
            if (c, d) in best:
                continue
            best[(c, d)] = g
            parent[(c, d)] = par
            if c == goal:
                path, node = [], (c, d)
                while node is not None:
                    path.append(node[0])
                    node = parent[node]
                return path[::-1]
            for k, (di, dj) in enumerate(DIRS):
                n = (c[0] + di, c[1] + dj)
                if not self.free(n):
                    continue
                g2 = g + 1 + (turn_cost if d not in (-1, k) else 0.0)
                if (n, k) not in best:
                    heapq.heappush(openq, (g2 + h(n), g2, n, k, (c, d)))
        return None


def corners(path):
    """Keep only the cells where the path changes direction (plus the ends)."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    out.append(path[-1])
    return out


def make_layout(marker_spacing=1, nx=21, ny=14, worn=0.0, seed=0):
    """Three shelf blocks per row band, two shelves deep, one-cell aisles.

    worn: fraction of floor codes that are unreadable (scuffed, covered),
    removed at random but never the start cell's code."""
    shelf = np.zeros((nx, ny), bool)
    for j in range(1, ny - 1):
        if j % 3 == 0:
            continue                    # east-west aisle
        for i0 in (2, 8, 14):
            shelf[i0:i0 + 5, j] = True
    mk = [(i, j) for i in range(0, nx, marker_spacing) for j in range(0, ny, marker_spacing)
          if not shelf[i, j]]
    if worn > 0:
        rng = np.random.default_rng(seed)
        keep = rng.random(len(mk)) >= worn
        mk = [m for m, k in zip(mk, keep) if k or m == (0, 0)]
    return Layout(shelf, np.array(mk, float))
