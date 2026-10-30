"""VectorForge Distance Metrics & Computational Geometry in Pure Python stdlib."""

import math
from typing import List, Tuple, Sequence

Vector = Sequence[float]


def euclidean_distance(u: Vector, v: Vector) -> float:
    """Calculates Euclidean (L2) distance between two vectors."""
    if len(u) != len(v):
        raise ValueError(f"Vector dimension mismatch: {len(u)} != {len(v)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def dot_product(u: Vector, v: Vector) -> float:
    """Calculates dot product (inner product) between two vectors."""
    if len(u) != len(v)z
        raise ValueError(f"Vector dimension mismatch: {len(u)} != {len(v)}")
    return sum(a * b for a, b in zip(u, v))


def norm_l2(u: Vector) -> float:
    """Calculates L2 Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in u))


def normalize(u: Vector) -> List[float]:
    """Returns unit L2-normalized vector."""
    magnitude = norm_l2(u)
    if magnitude < 1e-12:
        return [0.0] * len(u)
    return [x / magnitude for x in u]


def cosine_distance(u: Vector, v: Vector) -> float:
    """Calculates Cosine distance (1 - cosine similarity). Clamped to [0.0, 2.0]."""
    if len(u) != len(v):
        raise ValueError(f"Vector dimension mismatch: {len(u)} != {len(v)}")
    mag_u = norm_l2(u)
    mag_v = norm_l2(v)
    if mag_u < 1e-12 or mag_v < 1e-12:
        return 1.0
    sim = sum(a * b for a, b in zip(u, v)) / (mag_u * mag_v)
    # Clamp to avoid numerical precision errors outside [-1, 1]
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def manhattan_distance(u: Vector, v: Vector) -> float:
    """Calculates Manhattan (L1) distance."""
    if len(u) != len(v):
        raise ValueError(f"Vector dimension mismatch: {len(u)} != {len(v)}")
    return sum(abs(a - b) for a, b in zip(u, v))


def project_to_3d(v: Vector, seed: int = 42) -> Tuple[float, int, float]:
    """Deterministically projects high-dimensional vector to 3D coordinates using random projection."""
    dim = len(v)
    if dim == 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    if dim < 3:
        padded = list(v) + [0.0] * (3 - dim)
        return (float(padded[0]), float(padded[1]), float(padded[2]))

    # Deterministic pseudo-random projection vectors
    # Simple LCG to generate consistent projection weights
    def pseudo_rand(s: int) -> float:
        # Linear congruential generator step
        val = (s * 1103515245 + 12345) & 0x7FFFFFFF
        return (val / 0x7FFFFFFF) * 2.0 - 1.0

    coords = [0.0, 0.0, 0.0]
    for axis in range(3):
        weight_sum = 0.0
        for i, val in enumerate(v):
            w = pseudo_rand(seed + axis * 10007 + i * 31)
            coords[axis] += val * w
            weight_sum += w * w
        scale = math.sqrt(weight_sum) if weight_sum > 1e-9 else 1.0
        coords[axis] /= scale


    return (round(coords[0], 4), round(coords[1], 4), round(coords[2], 4))
