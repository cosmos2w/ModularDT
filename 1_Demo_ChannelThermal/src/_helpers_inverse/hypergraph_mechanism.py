from __future__ import annotations

"""Task-agnostic hypergraph mechanism features and atlas clustering.

The atlas here is an unsupervised mechanism-discovery prior, like clustering
internal physical behavior modes.  A mechanism is represented by a canonical
hypergraph organization plus a compact field-behavior descriptor; it is not a
valid-layout prior and it does not sample raw layouts directly.

The representation remains task-agnostic but physically meaningful because its
features are built from forward-HONF-extracted hypergraphs and behavior
descriptors.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass
class MechanismFeatureConfig:
    hypergraph_weight: float = 1.0
    behavior_weight: float = 0.5
    include_behavior: bool = True
    include_count_descriptor: bool = True
    count_weight: float = 0.25
    normalize_eps: float = 1e-6
    num_clusters: int = 24
    kmeans_iterations: int = 100
    kmeans_seed: int = 0


@dataclass
class MechanismAtlasState:
    feature_config: Dict[str, Any]
    hypergraph_mean: list
    hypergraph_std: list
    behavior_mean: list
    behavior_std: list
    count_mean: float
    count_std: float
    cluster_centers: list
    cluster_counts: list
    assignments: list
    feature_dim: int
    hypergraph_dim: int
    behavior_dim: int


def _as_2d_float(arr: Optional[np.ndarray], *, n: int = 0) -> np.ndarray:
    if arr is None:
        return np.zeros((int(n), 0), dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out[:, None]
    return np.nan_to_num(out.reshape(out.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _mean_std(arr: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    if arr.size == 0 or arr.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.mean(clean, axis=0).astype(np.float32)
    std = np.std(clean, axis=0).astype(np.float32)
    std = np.where(std < float(eps), 1.0, std).astype(np.float32)
    return mean, std


def _stats_array(stats: Mapping[str, Any], key: str, dim: int, default: float) -> np.ndarray:
    raw = stats.get(key)
    if raw is None:
        return np.full((int(dim),), float(default), dtype=np.float32)
    out = np.asarray(raw, dtype=np.float32).reshape(-1)
    if out.size < dim:
        out = np.pad(out, (0, dim - out.size), constant_values=float(default))
    return out[:dim].astype(np.float32)


def build_mechanism_features(
    hypergraph_vec: np.ndarray,
    behavior_vec: Optional[np.ndarray],
    count_vec: Optional[np.ndarray],
    config: MechanismFeatureConfig,
    stats: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build normalized mechanism feature rows from hypergraph/behavior parts."""

    hg = _as_2d_float(hypergraph_vec)
    n = int(hg.shape[0])
    behavior = _as_2d_float(behavior_vec, n=n)
    if behavior.shape[0] != n:
        behavior = behavior[:1].repeat(n, axis=0) if behavior.shape[0] == 1 else behavior[:n]
    count = _as_2d_float(count_vec, n=n)
    if count.shape[0] != n:
        count = count[:1].repeat(n, axis=0) if count.shape[0] == 1 else count[:n]
    if count.shape[1] > 1:
        count = count[:, :1]

    eps = float(config.normalize_eps)
    if stats is None:
        hg_mean, hg_std = _mean_std(hg, eps)
        bh_mean, bh_std = _mean_std(behavior, eps)
        if count.shape[1] > 0:
            cnt_mean_arr, cnt_std_arr = _mean_std(count, eps)
            count_mean = float(cnt_mean_arr[0]) if cnt_mean_arr.size else 0.0
            count_std = float(cnt_std_arr[0]) if cnt_std_arr.size else 1.0
        else:
            count_mean, count_std = 0.0, 1.0
    else:
        hg_mean = _stats_array(stats, "hypergraph_mean", hg.shape[1], 0.0)
        hg_std = np.maximum(_stats_array(stats, "hypergraph_std", hg.shape[1], 1.0), eps)
        bh_mean = _stats_array(stats, "behavior_mean", behavior.shape[1], 0.0)
        bh_std = np.maximum(_stats_array(stats, "behavior_std", behavior.shape[1], 1.0), eps)
        count_mean = float(stats.get("count_mean", 0.0))
        count_std = max(float(stats.get("count_std", 1.0)), eps)

    parts = []
    if hg.shape[1] > 0:
        parts.append(((hg - hg_mean[None, :]) / hg_std[None, :]) * float(config.hypergraph_weight))
    if bool(config.include_behavior) and behavior.shape[1] > 0:
        parts.append(((behavior - bh_mean[None, :]) / bh_std[None, :]) * float(config.behavior_weight))
    if bool(config.include_count_descriptor) and count.shape[1] > 0:
        parts.append(((count[:, :1] - count_mean) / count_std) * float(config.count_weight))
    features = np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((n, 0), dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out_stats: Dict[str, Any] = {
        "feature_config": asdict(config),
        "hypergraph_mean": hg_mean.astype(float).tolist(),
        "hypergraph_std": hg_std.astype(float).tolist(),
        "behavior_mean": bh_mean.astype(float).tolist(),
        "behavior_std": bh_std.astype(float).tolist(),
        "count_mean": float(count_mean),
        "count_std": float(count_std),
        "feature_dim": int(features.shape[1]),
        "hypergraph_dim": int(hg.shape[1]),
        "behavior_dim": int(behavior.shape[1]),
    }
    return features, out_stats


def fit_kmeans_numpy(features, num_clusters, *, iterations=100, seed=0) -> Dict[str, Any]:
    """Small NumPy k-means implementation with robust initialization."""

    x = _as_2d_float(np.asarray(features, dtype=np.float32))
    n, dim = int(x.shape[0]), int(x.shape[1])
    if n == 0:
        return {
            "centers": np.zeros((0, dim), dtype=np.float32),
            "assignments": np.zeros((0,), dtype=np.int64),
            "inertia": 0.0,
            "cluster_counts": np.zeros((0,), dtype=np.int64),
        }
    k = min(max(int(num_clusters), 1), n)
    rng = np.random.default_rng(int(seed))
    centers = np.empty((k, dim), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = x[first]
    closest_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(np.sum(closest_sq))
        if not np.isfinite(total) or total <= 0.0:
            idx = int(rng.integers(0, n))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n, p=probs))
        centers[c] = x[idx]
        closest_sq = np.minimum(closest_sq, np.sum((x - centers[c]) ** 2, axis=1))

    assignments = np.zeros((n,), dtype=np.int64)
    for _ in range(max(int(iterations), 1)):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(distances, axis=1).astype(np.int64)
        new_centers = centers.copy()
        for c in range(k):
            members = x[new_assignments == c]
            if members.size:
                new_centers[c] = np.mean(members, axis=0)
            else:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                new_centers[c] = x[farthest]
                new_assignments[farthest] = c
        if np.array_equal(assignments, new_assignments) and np.allclose(centers, new_centers):
            centers = new_centers.astype(np.float32)
            assignments = new_assignments
            break
        centers = new_centers.astype(np.float32)
        assignments = new_assignments

    distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1).astype(np.int64)
    inertia = float(np.sum(distances[np.arange(n), nearest]))
    counts = np.bincount(nearest, minlength=k).astype(np.int64)
    return {"centers": centers, "assignments": nearest, "inertia": inertia, "cluster_counts": counts}


class HypergraphMechanismAtlas:
    """Cluster-first mechanism prior over hypergraph/behavior descriptors.

    This is the unsupervised internal-structure model:
        mechanism = hypergraph organization + field-behavior descriptor.
    It is not a valid-layout prior.
    """

    def __init__(self, config: Optional[MechanismFeatureConfig] = None) -> None:
        self.config = config or MechanismFeatureConfig()
        self.stats: Dict[str, Any] = {}
        self.cluster_centers = np.zeros((0, 0), dtype=np.float32)
        self.cluster_counts = np.zeros((0,), dtype=np.int64)
        self.assignments = np.zeros((0,), dtype=np.int64)
        self.training_features = np.zeros((0, 0), dtype=np.float32)
        self.training_counts = np.zeros((0, 1), dtype=np.float32)

    def fit(self, hypergraph_vec, behavior_vec=None, count_vec=None) -> "HypergraphMechanismAtlas":
        features, stats = build_mechanism_features(hypergraph_vec, behavior_vec, count_vec, self.config)
        km = fit_kmeans_numpy(
            features,
            self.config.num_clusters,
            iterations=self.config.kmeans_iterations,
            seed=self.config.kmeans_seed,
        )
        self.stats = stats
        self.cluster_centers = np.asarray(km["centers"], dtype=np.float32)
        self.cluster_counts = np.asarray(km["cluster_counts"], dtype=np.int64)
        self.assignments = np.asarray(km["assignments"], dtype=np.int64)
        self.training_features = np.asarray(features, dtype=np.float32)
        self.training_counts = _as_2d_float(count_vec, n=features.shape[0])[:, :1]
        return self

    def encode(self, hypergraph_vec, behavior_vec=None, count_vec=None) -> np.ndarray:
        features, _ = build_mechanism_features(hypergraph_vec, behavior_vec, count_vec, self.config, stats=self.stats)
        return features.astype(np.float32)

    def nearest_cluster(self, features) -> np.ndarray:
        x = _as_2d_float(np.asarray(features, dtype=np.float32))
        if self.cluster_centers.size == 0:
            return np.zeros((x.shape[0],), dtype=np.int64)
        distances = np.sum((x[:, None, :] - self.cluster_centers[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1).astype(np.int64)

    def prior_energy(self, features) -> np.ndarray:
        x = _as_2d_float(np.asarray(features, dtype=np.float32))
        if self.cluster_centers.size == 0:
            return np.zeros((x.shape[0],), dtype=np.float32)
        distances = np.sum((x[:, None, :] - self.cluster_centers[None, :, :]) ** 2, axis=2)
        return np.sqrt(np.min(distances, axis=1)).astype(np.float32)

    def _eligible_clusters(self, count_range: Optional[Tuple[float, float]]) -> np.ndarray:
        k = int(self.cluster_centers.shape[0])
        eligible = np.arange(k, dtype=np.int64)
        if count_range is None or self.training_counts.size == 0 or self.assignments.size == 0:
            return eligible
        lo, hi = float(count_range[0]), float(count_range[1])
        keep = []
        counts = self.training_counts.reshape(-1)
        for cid in eligible:
            members = counts[self.assignments == cid]
            if members.size and np.any((members >= lo) & (members <= hi)):
                keep.append(int(cid))
        return np.asarray(keep if keep else eligible, dtype=np.int64)

    def sample_features(
        self,
        num_samples,
        *,
        rng,
        count_range: Optional[Tuple[float, float]] = None,
        cluster_ids=None,
        jitter_std: float = 0.05,
    ) -> Dict[str, Any]:
        if self.cluster_centers.size == 0:
            raise RuntimeError("HypergraphMechanismAtlas must be fit or loaded before sampling.")
        n = int(num_samples)
        rng_obj = rng if hasattr(rng, "choice") else np.random.default_rng(rng)
        if cluster_ids is None:
            eligible = self._eligible_clusters(count_range)
            weights = self.cluster_counts[eligible].astype(np.float64)
            weights = weights / weights.sum() if weights.sum() > 0.0 else np.full_like(weights, 1.0 / max(weights.size, 1))
            chosen = rng_obj.choice(eligible, size=n, replace=True, p=weights).astype(np.int64)
        else:
            chosen = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
            if chosen.size == 1 and n != 1:
                chosen = np.repeat(chosen, n)
            elif chosen.size != n:
                chosen = np.resize(chosen, n)

        rows = []
        for cid in chosen:
            members = np.where(self.assignments == int(cid))[0]
            if members.size and self.training_features.shape[0] > int(np.max(members)):
                base = self.training_features[int(rng_obj.choice(members))]
            else:
                base = self.cluster_centers[int(cid)]
            rows.append(base)
        features = np.asarray(rows, dtype=np.float32)
        if float(jitter_std) > 0.0:
            features = features + rng_obj.normal(0.0, float(jitter_std), size=features.shape).astype(np.float32)
        decoded = self.decode_feature_to_parts(features)
        return {"features": features.astype(np.float32), "cluster_ids": chosen.astype(np.int64), **decoded}

    def decode_feature_to_parts(self, features) -> Dict[str, np.ndarray]:
        x = _as_2d_float(np.asarray(features, dtype=np.float32))
        cursor = 0
        hg_dim = int(self.stats.get("hypergraph_dim", 0))
        bh_dim = int(self.stats.get("behavior_dim", 0))
        out: Dict[str, np.ndarray] = {}
        if hg_dim > 0:
            hg_norm = x[:, cursor : cursor + hg_dim] / max(float(self.config.hypergraph_weight), 1.0e-12)
            hg_mean = _stats_array(self.stats, "hypergraph_mean", hg_dim, 0.0)
            hg_std = _stats_array(self.stats, "hypergraph_std", hg_dim, 1.0)
            out["hypergraph_vec"] = (hg_norm * hg_std[None, :] + hg_mean[None, :]).astype(np.float32)
            cursor += hg_dim
        else:
            out["hypergraph_vec"] = np.zeros((x.shape[0], 0), dtype=np.float32)
        if bool(self.config.include_behavior) and bh_dim > 0:
            bh_norm = x[:, cursor : cursor + bh_dim] / max(float(self.config.behavior_weight), 1.0e-12)
            bh_mean = _stats_array(self.stats, "behavior_mean", bh_dim, 0.0)
            bh_std = _stats_array(self.stats, "behavior_std", bh_dim, 1.0)
            out["behavior_vec"] = (bh_norm * bh_std[None, :] + bh_mean[None, :]).astype(np.float32)
            cursor += bh_dim
        else:
            out["behavior_vec"] = np.zeros((x.shape[0], 0), dtype=np.float32)
        if bool(self.config.include_count_descriptor) and cursor < x.shape[1]:
            cnt_norm = x[:, cursor : cursor + 1] / max(float(self.config.count_weight), 1.0e-12)
            out["count_descriptor"] = (cnt_norm * float(self.stats.get("count_std", 1.0)) + float(self.stats.get("count_mean", 0.0))).astype(np.float32)
        return out

    def to_state(self) -> dict:
        state = MechanismAtlasState(
            feature_config=asdict(self.config),
            hypergraph_mean=list(self.stats.get("hypergraph_mean", [])),
            hypergraph_std=list(self.stats.get("hypergraph_std", [])),
            behavior_mean=list(self.stats.get("behavior_mean", [])),
            behavior_std=list(self.stats.get("behavior_std", [])),
            count_mean=float(self.stats.get("count_mean", 0.0)),
            count_std=float(self.stats.get("count_std", 1.0)),
            cluster_centers=np.asarray(self.cluster_centers, dtype=float).tolist(),
            cluster_counts=np.asarray(self.cluster_counts, dtype=int).tolist(),
            assignments=np.asarray(self.assignments, dtype=int).tolist(),
            feature_dim=int(self.stats.get("feature_dim", self.cluster_centers.shape[1] if self.cluster_centers.ndim == 2 else 0)),
            hypergraph_dim=int(self.stats.get("hypergraph_dim", 0)),
            behavior_dim=int(self.stats.get("behavior_dim", 0)),
        )
        return asdict(state)

    @classmethod
    def from_state(cls, state) -> "HypergraphMechanismAtlas":
        payload = dict(state)
        config = MechanismFeatureConfig(**dict(payload.get("feature_config", {})))
        atlas = cls(config)
        atlas.stats = {
            "feature_config": asdict(config),
            "hypergraph_mean": list(payload.get("hypergraph_mean", [])),
            "hypergraph_std": list(payload.get("hypergraph_std", [])),
            "behavior_mean": list(payload.get("behavior_mean", [])),
            "behavior_std": list(payload.get("behavior_std", [])),
            "count_mean": float(payload.get("count_mean", 0.0)),
            "count_std": float(payload.get("count_std", 1.0)),
            "feature_dim": int(payload.get("feature_dim", 0)),
            "hypergraph_dim": int(payload.get("hypergraph_dim", 0)),
            "behavior_dim": int(payload.get("behavior_dim", 0)),
        }
        atlas.cluster_centers = np.asarray(payload.get("cluster_centers", []), dtype=np.float32)
        if atlas.cluster_centers.ndim == 1:
            atlas.cluster_centers = atlas.cluster_centers.reshape(0, int(atlas.stats.get("feature_dim", 0)))
        atlas.cluster_counts = np.asarray(payload.get("cluster_counts", []), dtype=np.int64)
        atlas.assignments = np.asarray(payload.get("assignments", []), dtype=np.int64)
        atlas.training_features = np.zeros((0, int(atlas.stats.get("feature_dim", 0))), dtype=np.float32)
        atlas.training_counts = np.zeros((0, 1), dtype=np.float32)
        return atlas
