"""Memory-mapped access to the ragged wind-farm volume dataset.

The source package contains a large ``family_tensor.npz`` convenience bundle
and an even larger ragged full-volume export.  This module addresses only the
individual ``family_volume/*.npy`` arrays by default.  Every large array is
opened with ``mmap_mode='r'`` and individual runs are exposed as views, so
loading metadata or plotting one case does not read the complete dataset.

If a future copy of the dataset contains compact individual ``.npy`` files,
``compact_metadata`` can read the small metadata arrays from those files.  The
optional NPZ fallback is explicitly opt-in and never reads field arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOLUME_ARRAYS: tuple[str, ...] = (
    "U",
    "p",
    "k",
    "epsilon",
    "run_cell_offsets",
    "run_shape",
    "run_x_offsets",
    "run_y_offsets",
    "run_z_offsets",
    "x_cell_m",
    "y_cell_m",
    "z_cell_m",
    "case",
    "layout_index",
    "wd_deg",
    "source_time",
    "completed",
)

COMPACT_METADATA_ARRAYS: tuple[str, ...] = (
    "case",
    "layout",
    "layout_index",
    "wd_deg",
    "n_turbines",
    "turbine_xy_D",
    "wake_loss_pct",
    "descriptors",
    "descriptor_names",
    "sampled_params",
    "sampled_param_names",
    "domain_inside_window",
    "gap_filled_nodes",
    "vert_plane_y_m",
    "calib_ratio",
    "U_ref",
    "D_m",
    "hub_height_m",
)


class DatasetLayoutError(ValueError):
    """Raised when the expected wind-farm array layout is not present."""


def _load_npy(path: Path, *, mmap: bool = True) -> np.ndarray:
    """Load one array without permitting object deserialisation."""

    if not path.is_file():
        raise DatasetLayoutError(f"Missing required dataset array: {path}")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def discover_volume_root(root: str | Path) -> Path:
    """Return the directory containing the ragged ``family_volume`` arrays.

    ``root`` may point directly at ``family_volume`` or at the dataset root
    containing that directory.  Symlinks are resolved only after the candidate
    has been selected, preserving a useful path in error messages.
    """

    candidate = Path(root).expanduser()
    direct = candidate / "run_shape.npy"
    nested = candidate / "family_volume" / "run_shape.npy"
    if direct.is_file():
        return candidate.resolve()
    if nested.is_file():
        return (candidate / "family_volume").resolve()
    raise DatasetLayoutError(
        f"Could not find family_volume arrays below {candidate}. Pass the dataset root or the family_volume directory."
    )


def _slice_from_offsets(offsets: np.ndarray, index: int, *, name: str) -> slice:
    if offsets.ndim != 1 or len(offsets) < 2:
        raise DatasetLayoutError(f"{name} must be a one-dimensional offsets array")
    if not 0 <= index < len(offsets) - 1:
        raise IndexError(f"Run index {index} is outside [0, {len(offsets) - 2}]")
    start = int(offsets[index])
    stop = int(offsets[index + 1])
    return slice(start, stop)


@dataclass(frozen=True)
class RunView:
    """A single ragged run and its exact cell-centre axes."""

    index: int
    case: str
    layout_index: int
    wind_direction_deg: float
    source_time: float
    shape_nxyz: tuple[int, int, int]
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    U: np.ndarray
    p: np.ndarray
    k: np.ndarray
    epsilon: np.ndarray

    @property
    def nx(self) -> int:
        return self.shape_nxyz[0]

    @property
    def ny(self) -> int:
        return self.shape_nxyz[1]

    @property
    def nz(self) -> int:
        return self.shape_nxyz[2]

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def U_structured(self) -> np.ndarray:
        """Velocity view with shape ``(nz, ny, nx, 3)``."""

        return self.U.reshape(self.nz, self.ny, self.nx, 3)

    @property
    def p_structured(self) -> np.ndarray:
        return self.p.reshape(self.nz, self.ny, self.nx)

    @property
    def k_structured(self) -> np.ndarray:
        return self.k.reshape(self.nz, self.ny, self.nx)

    @property
    def epsilon_structured(self) -> np.ndarray:
        return self.epsilon.reshape(self.nz, self.ny, self.nx)


class WindFarmDataset:
    """Read-only, memory-mapped view of a ``family_volume`` dataset."""

    def __init__(self, root: str | Path):
        self.volume_root = discover_volume_root(root)
        self.root = self.volume_root.parent
        self._arrays: dict[str, np.ndarray] = {
            name: _load_npy(self.volume_root / f"{name}.npy") for name in VOLUME_ARRAYS
        }
        self._validate_metadata_shapes()

    def _validate_metadata_shapes(self) -> None:
        n = len(self._arrays["case"])
        for name in ("layout_index", "wd_deg", "source_time", "completed"):
            if self._arrays[name].shape != (n,):
                raise DatasetLayoutError(f"{name}.npy has shape {self._arrays[name].shape}, expected {(n,)}")
        if self._arrays["run_shape"].shape != (n, 3):
            raise DatasetLayoutError(f"run_shape.npy has shape {self._arrays['run_shape'].shape}, expected {(n, 3)}")
        for name in ("run_cell_offsets", "run_x_offsets", "run_y_offsets", "run_z_offsets"):
            if self._arrays[name].shape != (n + 1,):
                raise DatasetLayoutError(f"{name}.npy has shape {self._arrays[name].shape}, expected {(n + 1,)}")

    @property
    def n_runs(self) -> int:
        return len(self._arrays["case"])

    @property
    def arrays(self) -> Mapping[str, np.ndarray]:
        """Mapping of volume array names to read-only mmap-backed arrays."""

        return self._arrays

    def array(self, name: str) -> np.ndarray:
        try:
            return self._arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown volume array {name!r}") from exc

    def run_bounds(self, index: int) -> slice:
        return _slice_from_offsets(self._arrays["run_cell_offsets"], index, name="run_cell_offsets")

    def axis(self, index: int, axis: str) -> np.ndarray:
        if axis not in "xyz":
            raise ValueError(f"axis must be one of x, y, z; got {axis!r}")
        values = self._arrays[f"{axis}_cell_m"]
        offsets = self._arrays[f"run_{axis}_offsets"]
        return values[_slice_from_offsets(offsets, index, name=f"run_{axis}_offsets")]

    def shape(self, index: int) -> tuple[int, int, int]:
        return tuple(int(value) for value in self._arrays["run_shape"][index])  # type: ignore[return-value]

    def run(self, index: int) -> RunView:
        """Return one run as mmap-backed flat field views plus coordinate axes."""

        bounds = self.run_bounds(index)
        shape = self.shape(index)
        expected = int(np.prod(shape, dtype=np.int64))
        if bounds.stop - bounds.start != expected:
            raise DatasetLayoutError(
                f"Run {index} cell offset length {bounds.stop - bounds.start} does not match "
                f"run_shape product {expected}"
            )
        return RunView(
            index=index,
            case=str(self._arrays["case"][index]),
            layout_index=int(self._arrays["layout_index"][index]),
            wind_direction_deg=float(self._arrays["wd_deg"][index]),
            source_time=float(self._arrays["source_time"][index]),
            shape_nxyz=shape,
            x_m=self.axis(index, "x"),
            y_m=self.axis(index, "y"),
            z_m=self.axis(index, "z"),
            U=self._arrays["U"][bounds],
            p=self._arrays["p"][bounds],
            k=self._arrays["k"][bounds],
            epsilon=self._arrays["epsilon"][bounds],
        )

    def iter_runs(self, indices: Iterable[int] | None = None):
        """Yield :class:`RunView` objects in deterministic row order."""

        selected = range(self.n_runs) if indices is None else sorted(int(i) for i in indices)
        for index in selected:
            yield self.run(index)

    def compact_metadata(self, *, allow_npz_fallback: bool = False) -> dict[str, np.ndarray]:
        """Read compact metadata without touching large field arrays.

        Individual compact ``.npy`` arrays are preferred.  The source copy
        currently provides only ``family_tensor.npz``; when
        ``allow_npz_fallback`` is true, only the small metadata keys requested
        by this method are decompressed.  Hub/vertical field keys are never
        read here.
        """

        search_roots = (self.root / "family_tensor", self.volume_root.parent / "family_tensor")
        result: dict[str, np.ndarray] = {}
        for name in COMPACT_METADATA_ARRAYS:
            for directory in search_roots:
                path = directory / f"{name}.npy"
                if path.is_file():
                    result[name] = _load_npy(path)
                    break
        missing = [name for name in COMPACT_METADATA_ARRAYS if name not in result]
        if not missing and result:
            return result
        if not allow_npz_fallback:
            return result

        candidates = (self.root / "family_tensor.npz", self.volume_root / "family_tensor.npz")
        bundle = next((path for path in candidates if path.is_file()), None)
        if bundle is None:
            return result
        with np.load(bundle, allow_pickle=False) as archive:
            for name in missing:
                if name in archive.files:
                    # Metadata keys are bounded by the compact dataset schema;
                    # copying closes the ZIP file without retaining a handle.
                    result[name] = np.asarray(archive[name])
        return result
