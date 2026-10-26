"""Self-contained physical training archives and process-local HDF5 reading.

Model features use a strict input whitelist. Split labels, generator provenance,
solver diagnostics and target-derived statistics never enter ``inputs``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .environment import RENDERER, control_records
from .io import canonical_json, sha256_file, sha256_json
from .processing import FIELD_ORDER, derive_fields
from .resources import require_budget, require_owned_root

SCHEMA = "hetmf_training_h5_v1"
ENVIRONMENT_COLUMNS = ("center_x", "center_y", "spacing_x", "spacing_y", "sampling_area", "s", "kappa", "beta")
CONTACT_COLUMNS = ("x", "y", "r", "gamma", "u")
SPLITS = ("train", "dev", "test", "panel_new_environment", "panel_new_layout", "panel_new_pair", "panel_extrapolation", "audit")
STR = h5py.string_dtype("utf-8")


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _digest_dataset(dataset, digest):
    digest.update(dataset.name.encode())
    digest.update(canonical_json([list(dataset.shape), str(dataset.dtype)]).encode())
    for start in range(0, dataset.shape[0] if dataset.ndim else 1, 16):
        value = dataset[start:start + 16] if dataset.ndim else dataset[()]
        if h5py.check_string_dtype(dataset.dtype) is not None:
            digest.update(canonical_json(np.asarray(value).astype(str).tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(value).tobytes())


def _payload_hash(handle):
    digest = hashlib.sha256()
    attrs = {k: v.item() if isinstance(v, np.generic) else v for k, v in handle.attrs.items() if k != "payload_sha256"}
    digest.update(canonical_json(attrs).encode())
    names = []
    def collect(name, obj):
        group_attrs = {k: v.item() if isinstance(v, np.generic) else v for k, v in obj.attrs.items()}
        digest.update(canonical_json([name, group_attrs]).encode())
        if isinstance(obj, h5py.Dataset):
            names.append(name)
    handle.visititems(collect)
    for name in sorted(names):
        _digest_dataset(handle[name], digest)
    return digest.hexdigest()


def _case_hash(environment, contacts, mask, fields):
    digest = hashlib.sha256()
    for value in (environment, contacts, mask, fields):
        a = np.ascontiguousarray(value)
        digest.update(canonical_json([list(a.shape), str(a.dtype)]).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


class _Moments:
    def __init__(self, width):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def add(self, values):
        v = np.asarray(values, dtype=np.float64).reshape(-1, len(self.mean))
        n = len(v)
        if not n:
            return
        mean = v.mean(axis=0)
        m2 = np.square(v - mean).sum(axis=0)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += m2 + np.square(delta) * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def std(self):
        # Constant geometry columns retain a neutral scale rather than divide by zero.
        std = np.sqrt(self.m2 / self.count)
        return np.where(std < 1e-12, 1.0, std)


def _normalization(handle):
    stats = {"fields": _Moments(5), "environment_records": _Moments(8), "contacts": _Moments(5)}
    rows = handle["splits/train"][:]
    for i in rows:
        stats["fields"].add(handle["targets/fields"][i])
        stats["environment_records"].add(handle["inputs/environment_records"][i])
        stats["contacts"].add(handle["inputs/contacts"][i][handle["inputs/contact_mask"][i]])
    fingerprint = sha256_json({"indices": rows.tolist(), "state_ids": [_text(handle["metadata/state_id"][i]) for i in rows],
                               "input_hashes": [_text(handle["metadata/input_hash"][i]) for i in rows]})
    return stats, fingerprint


def export_training_h5(release_root, output_path, *, retained_budget_bytes,
                       temporary_budget_bytes, minimum_free_bytes=50 * 2**30):
    """Validate FP64 sources, then atomically export a standalone physical archive.

    The retained bound covers this new output only; temporary bound covers its
    complete atomic staging file. Existing release files are never rewritten.
    """
    from . import formal
    root = require_owned_root(Path(release_root))
    output = require_owned_root(Path(output_path))
    if not root.is_dir() or output.exists():
        raise FileExistsError("release missing or output already exists; verify existing archives explicitly")
    context = formal.load_release_context(root)
    plan, approval = context["plan"], context["approval"]
    rows = plan["states"]
    grids = {formal._grid(row, approval) for row in rows}
    if len(grids) != 1:
        raise ValueError("single training archive requires one native grid")
    grid = grids.pop()
    n = len(rows)
    upper = n * (grid * grid * 5 * 4 + 144 * 8 * 8 + 16 * 5 * 8 + 16384) + 2 * len(canonical_json(plan).encode()) + 16 * 2**20
    if retained_budget_bytes < upper or temporary_budget_bytes < upper:
        raise RuntimeError(f"training archive requires conservative retained and staging bounds >= {upper}")
    reservation = require_budget(retained_bytes=retained_budget_bytes, temporary_bytes=temporary_budget_bytes,
                                 minimum_free_bytes=minimum_free_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with h5py.File(temporary, "x") as h:
            h.attrs.update(schema=SCHEMA, accepted=False, profile=plan["profile"], plan_hash=plan["plan_hash"],
                           field_order=",".join(FIELD_ORDER), environment_columns=",".join(ENVIRONMENT_COLUMNS),
                           contact_columns=",".join(CONTACT_COLUMNS), renderer=RENDERER, units="dimensionless", grid_n=grid,
                           normalization_policy="population moments over core TRAIN cases only; constant std=1",
                           source_release=str(root), source_approval_sha256=sha256_json(approval))
            inputs = h.create_group("inputs")
            envds = inputs.create_dataset("environment_records", (n, 144, 8), dtype="f8", chunks=(1,144,8), compression="lzf")
            cds = inputs.create_dataset("contacts", (n,16,5), dtype="f8", chunks=(1,16,5), compression="lzf")
            masks = inputs.create_dataset("contact_mask", (n,16), dtype="?", chunks=(1,16), compression="lzf")
            fields = h.create_dataset("targets/fields", (n,grid,grid,5), dtype="f4", chunks=(1,grid,grid,5), compression="lzf")
            xy = (np.arange(grid, dtype=np.float64) + .5) / grid
            xx, yy = np.meshgrid(xy, xy)
            h.create_dataset("queries/xy", data=np.stack((xx, yy), -1).reshape(-1,2))
            h.create_dataset("queries/cell_area", data=np.full(grid * grid, 1 / grid**2))
            h.create_dataset("queries/environment_sampling_area", data=np.full(144, 1 / 144))
            h.create_dataset("provenance/formal_plan_json", data=canonical_json(plan), dtype=STR)
            h.create_dataset("provenance/formal_approval_json", data=canonical_json(approval), dtype=STR)
            meta = h.create_group("metadata")
            keys = ("state_id", "split", "environment_root_id", "layout_root_id", "input_hash", "challenge", "quartet_id", "quartet_endpoint", "source_sha256", "source_key", "solver_sha256", "source_input_sha256", "source_approval_sha256")
            mds = {k: meta.create_dataset(k, (n,), dtype=STR) for k in keys}
            contact_ids = h.create_dataset("provenance/contact_id", (n,16), dtype=STR)
            audits = h.create_dataset("provenance/source_physical_audit_json", (n,), dtype=STR)
            inheritance_ds = h.create_dataset("provenance/source_inheritance_json", (n,), dtype=STR)
            hashes = h.create_dataset("integrity/case_sha256", (n,), dtype=STR)
            for split in SPLITS:
                h.create_dataset(f"splits/{split}", data=np.asarray([i for i,row in enumerate(rows) if row["split"] == split], dtype=np.int64))
            for i,row in enumerate(rows):
                validated = formal.resolve_release_row(root, row, context=context)
                scene = validated["scene"]
                source = Path(validated["raw_path"])
                with h5py.File(source, "r") as raw:
                    target = derive_fields(raw)
                    if not np.isfinite(target).all():
                        raise RuntimeError("nonfinite derived target")
                    solver_hash = _text(raw.attrs["solver_sha256"])
                    source_input_hash = _text(raw.attrs["input_sha256"])
                records = np.asarray([[r[k] for k in ENVIRONMENT_COLUMNS] for r in control_records(scene["environment"])], dtype=np.float64)
                contacts = np.zeros((16,5), dtype=np.float64)
                mask = np.zeros(16, dtype=np.bool_)
                count = len(scene["contacts"])
                if count > 16:
                    raise ValueError("archive supports at most 16 contacts")
                contacts[:count] = [[contact[k] for k in CONTACT_COLUMNS] for contact in scene["contacts"]]
                mask[:count] = True
                envds[i], cds[i], masks[i], fields[i] = records, contacts, mask, target
                contact_ids[i] = [str(c["id"]) for c in scene["contacts"]] + [""] * (16-count)
                audits[i] = canonical_json(validated["physical_audit"])
                inheritance_ds[i] = canonical_json(validated.get("inheritance"))
                values = {**{k: str(row.get(k, "")) for k in keys}, "source_sha256": validated["raw_sha256"],
                          "source_key": source.name, "solver_sha256": solver_hash, "source_input_sha256": source_input_hash,
                          "source_approval_sha256": validated["approval_hash"]}
                for k in keys:
                    mds[k][i] = values[k]
                hashes[i] = _case_hash(records, contacts, mask, target)
            stats, fingerprint = _normalization(h)
            norm = h.create_group("normalization")
            norm.attrs.update(fit_split="train", train_membership_sha256=fingerprint, variance="population", constant_std_floor=1e-12)
            for name,moments in stats.items():
                norm.create_dataset(f"{name}/mean", data=moments.mean)
                norm.create_dataset(f"{name}/std", data=moments.std())
                norm.create_dataset(f"{name}/count", data=moments.count)
            h.attrs["accepted"] = True
            h.attrs["payload_sha256"] = _payload_hash(h)
            h.flush()
        verification = verify_training_h5(temporary)
        actual = temporary.stat().st_size
        if actual > retained_budget_bytes or actual > temporary_budget_bytes:
            raise RuntimeError("training archive exceeded declared byte budget")
        # Atomic no-clobber publication: a concurrent publisher owns any existing name.
        os.link(temporary, output)
        temporary.unlink()
        return {"path": str(output), "bytes": actual, "sha256": sha256_file(output), **verification,
                "storage_reservation": reservation, "source_cases_revalidated": n}
    finally:
        temporary.unlink(missing_ok=True)



def _verify_frozen_semantics(h):
    """Bind portable archive records to the embedded frozen physical inputs.

    Original numerical solver hashes are retained verbatim: they must not be
    rederived from whichever implementation happens to read an archive later.
    """
    from .study_design import hash_plan, resolve_state
    from .formal import validate_formal_plan, _safe_scene
    plan = json.loads(_text(h["provenance/formal_plan_json"][()]))
    approval = json.loads(_text(h["provenance/formal_approval_json"][()]))
    try:
        validate_formal_plan(plan)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("embedded frozen input plan invalid") from error
    approval_hash = sha256_json(approval)
    if (hash_plan(plan) != h.attrs["plan_hash"] or plan["profile"] != h.attrs["profile"] or
            approval.get("plan_hash") != plan["plan_hash"] or
            approval_hash != h.attrs["source_approval_sha256"] or
            approval.get("disk_root") != h.attrs["source_release"]):
        raise RuntimeError("embedded plan/approval identity mismatch")
    if len(plan["states"]) != len(h["targets/fields"]):
        raise RuntimeError("embedded plan state count mismatch")
    for i,row in enumerate(plan["states"]):
        for key in ("state_id", "split", "environment_root_id", "layout_root_id", "input_hash", "challenge", "quartet_id", "quartet_endpoint"):
            if _text(h[f"metadata/{key}"][i]) != str(row.get(key, "")):
                raise RuntimeError(f"archive metadata differs from frozen plan: {key}")
        try:
            scene = resolve_state(plan,row)
            source_scene = _safe_scene(plan,row)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("embedded physical input hash mismatch") from error
        records = np.asarray([[r[k] for k in ENVIRONMENT_COLUMNS] for r in control_records(scene["environment"])], dtype=np.float64)
        contacts = np.zeros((16,5), dtype=np.float64)
        count = len(scene["contacts"])
        if count > 16:
            raise RuntimeError("frozen contacts exceed archive capacity")
        contacts[:count] = [[c[k] for k in CONTACT_COLUMNS] for c in scene["contacts"]]
        mask = np.arange(16) < count
        literals = [str(c["id"]) for c in scene["contacts"]] + [""] * (16-count)
        if (not np.array_equal(h["inputs/environment_records"][i],records) or
                not np.array_equal(h["inputs/contacts"][i],contacts) or
                not np.array_equal(h["inputs/contact_mask"][i],mask) or
                h["provenance/contact_id"].asstr()[i].tolist() != literals):
            raise RuntimeError("archive physical inputs differ from frozen plan")
        source_key = f"{source_scene['state_id']}_n{h.attrs['grid_n']}.h5"
        if (_text(h["metadata/source_key"][i]) != source_key or
                _text(h["metadata/source_input_sha256"][i]) != sha256_json(source_scene)):
            raise RuntimeError("archive source input lineage mismatch")
        inherited = json.loads(_text(h["provenance/source_inheritance_json"][i]))
        source_approval = _text(h["metadata/source_approval_sha256"][i])
        if inherited is None:
            if source_approval != approval_hash:
                raise RuntimeError("archive source approval mismatch")
        else:
            spec = approval.get("inheritance")
            if (not isinstance(spec,dict) or not isinstance(inherited,dict) or
                    any(inherited.get(k) != spec.get(k) for k in ("root","plan_hash","approval_hash")) or
                    source_approval != spec.get("approval_hash") or
                    inherited.get("source_raw_path") != str(Path(spec["root"])/"raw"/source_key) or
                    inherited.get("source_raw_sha256") != _text(h["metadata/source_sha256"][i])):
                raise RuntimeError("archive inherited source lineage mismatch")
        audit = json.loads(_text(h["provenance/source_physical_audit_json"][i]))
        if audit.get("accepted") is not True:
            raise RuntimeError("archive source physical audit was not accepted")


def verify_training_h5(path, *, verify_payload=True):
    """Check the full partition, input whitelist, source binding and payload seal."""
    with h5py.File(path, "r") as h:
        if h.attrs.get("schema") != SCHEMA or not h.attrs.get("accepted"):
            raise RuntimeError("unaccepted or unsupported training archive")
        required_attrs = {"field_order": ",".join(FIELD_ORDER), "environment_columns": ",".join(ENVIRONMENT_COLUMNS),
                          "contact_columns": ",".join(CONTACT_COLUMNS), "renderer": RENDERER, "units": "dimensionless"}
        if any(h.attrs.get(k) != v for k,v in required_attrs.items()):
            raise RuntimeError("training canonical columns/renderer/units mismatch")
        n,ny,nx,nf = h["targets/fields"].shape
        if ny != nx or nf != 5 or h["inputs/environment_records"].shape != (n,144,8) or h["inputs/contacts"].shape != (n,16,5) or h["inputs/contact_mask"].shape != (n,16):
            raise RuntimeError("training array dimensions inconsistent")
        if set(h["inputs"]) != {"environment_records", "contacts", "contact_mask"} or set(h["splits"]) != set(SPLITS):
            raise RuntimeError("training input whitelist or split set mismatch")
        if int(h.attrs.get("grid_n", -1)) != ny or h["targets/fields"].dtype != np.dtype("float32"):
            raise RuntimeError("training grid or target precision mismatch")
        xy = (np.arange(ny, dtype=np.float64) + .5) / ny
        xx,yy = np.meshgrid(xy,xy)
        expected_xy = np.stack((xx,yy),-1).reshape(-1,2)
        if (not np.array_equal(h["queries/xy"][:],expected_xy) or
                not np.array_equal(h["queries/cell_area"][:],np.full(ny*ny,1/ny**2)) or
                not np.array_equal(h["queries/environment_sampling_area"][:],np.full(144,1/144))):
            raise RuntimeError("training query geometry/measure mismatch")
        norm = h["normalization"]
        if norm.attrs.get("fit_split") != "train" or norm.attrs.get("variance") != "population":
            raise RuntimeError("training normalizer policy mismatch")
        for name,width in (("fields",5),("environment_records",8),("contacts",5)):
            mean,std = (h[f"normalization/{name}/{k}"][:] for k in ("mean","std"))
            if mean.shape != (width,) or std.shape != (width,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0) or h[f"normalization/{name}/count"][()] <= 0:
                raise RuntimeError("training normalizer dimensions or values invalid")
        if any(v.shape != (n,) for v in h["metadata"].values()):
            raise RuntimeError("training metadata dimensions inconsistent")
        state_ids = h["metadata/state_id"].asstr()[:]
        if len(set(state_ids)) != n:
            raise RuntimeError("duplicate state IDs")
        split_values = h["metadata/split"].asstr()[:]
        indices = []
        for split in SPLITS:
            ids = h[f"splits/{split}"][:]
            if not np.array_equal(ids, np.flatnonzero(split_values == split)):
                raise RuntimeError(f"invalid split membership: {split}")
            indices.extend(ids.tolist())
        if sorted(indices) != list(range(n)):
            raise RuntimeError("split memberships do not partition archive")
        for a,b in (("train","dev"),("train","test"),("dev","test")):
            for key in ("environment_root_id", "layout_root_id", "input_hash"):
                values = h[f"metadata/{key}"].asstr()[:]
                if set(values[h[f"splits/{a}"][:]]) & set(values[h[f"splits/{b}"][:]]):
                    raise RuntimeError(f"core split leakage: {key}")
        if verify_payload:
            _verify_frozen_semantics(h)
            if _payload_hash(h) != h.attrs.get("payload_sha256"):
                raise RuntimeError("training payload checksum mismatch")
            for i in range(n):
                records, contacts, mask, fields = (h[k][i] for k in ("inputs/environment_records", "inputs/contacts", "inputs/contact_mask", "targets/fields"))
                if not all(np.isfinite(v).all() for v in (records, contacts, fields)) or (contacts[~mask] != 0).any():
                    raise RuntimeError("nonfinite training arrays or invalid contact padding")
                if _case_hash(records, contacts, mask, fields) != _text(h["integrity/case_sha256"][i]):
                    raise RuntimeError("training case checksum mismatch")
            stats, fingerprint = _normalization(h)
            if fingerprint != h["normalization"].attrs.get("train_membership_sha256"):
                raise RuntimeError("normalization train membership mismatch")
            for name,moments in stats.items():
                for key,value in (("mean",moments.mean),("std",moments.std()),("count",moments.count)):
                    if not np.array_equal(h[f"normalization/{name}/{key}"][()], value):
                        raise RuntimeError("normalizer differs from TRAIN-only recomputation")
        return {"schema": SCHEMA, "states": n, "grid": ny, "profile": _text(h.attrs["profile"]),
                "plan_hash": _text(h.attrs["plan_hash"]), "payload_sha256": _text(h.attrs["payload_sha256"]),
                "split_counts": {k: len(h[f"splits/{k}"]) for k in SPLITS}, "payload_verified": bool(verify_payload)}


class HETMFDataset:
    """Lazy, pickle and fork safe case loader with deterministic native queries.

    ``normalize`` changes targets only. Inputs always retain canonical physical
    values. Query weights are domain-area Monte Carlo weights (1/Q); the native
    finite-volume cell area remains available in metadata.
    """
    def __init__(self, path, split="train", query_count=None, seed=0, normalize=False):
        self.path = str(Path(path).resolve())
        if split not in SPLITS:
            raise ValueError(f"unknown split: {split}")
        verify_training_h5(self.path, verify_payload=False)
        with h5py.File(self.path, "r") as h:
            self.indices = h[f"splits/{split}"][:]
            self.grid = int(h.attrs["grid_n"])
            self.dataset_payload_sha256 = _text(h.attrs["payload_sha256"])
            self.dataset_plan_hash = _text(h.attrs["plan_hash"])
            self.normalization_train_membership_sha256 = _text(h["normalization"].attrs["train_membership_sha256"])
            self.normalization = {name: {k: h[f"normalization/{name}/{k}"][()] for k in ("mean","std","count")}
                                  for name in ("fields","environment_records","contacts")}
        if query_count is not None and (not isinstance(query_count, int) or not 1 <= query_count <= self.grid**2):
            raise ValueError("query_count outside native grid")
        self.query_count, self.seed, self.normalize = query_count, int(seed), bool(normalize)
        self.split, self.epoch = split, 0
        self._handle, self._pid = None, None

    def __len__(self):
        return len(self.indices)

    def _open(self):
        if self._handle is None or self._pid != os.getpid():
            self.close()
            opened = h5py.File(self.path, "r")
            try:
                identity = (_text(opened.attrs["payload_sha256"]), _text(opened.attrs["plan_hash"]),
                            _text(opened["normalization"].attrs["train_membership_sha256"]))
                expected = (self.dataset_payload_sha256, self.dataset_plan_hash,
                            self.normalization_train_membership_sha256)
                if identity != expected:
                    raise RuntimeError("training archive replaced since dataset initialization")
            except BaseException:
                opened.close()
                raise
            self._handle = opened
            self._pid = os.getpid()
        return self._handle

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
        self._handle, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"], state["_pid"] = None, None
        return state

    def __del__(self):
        self.close()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        i = int(self.indices[index])
        h = self._open()
        total = self.grid**2
        ids = np.arange(total)
        if self.query_count is not None:
            seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.epoch}:{_text(h['metadata/state_id'][i])}:{_text(h['metadata/input_hash'][i])}".encode()).digest()[:8], "little")
            ids = np.random.default_rng(seed).choice(total, self.query_count, replace=False)
        fields = h["targets/fields"][i].reshape(total,5)[ids]
        if self.normalize:
            fields = ((fields - self.normalization["fields"]["mean"]) / self.normalization["fields"]["std"]).astype(np.float32)
        xy = h["queries/xy"][:][ids]
        inputs = {k: h[f"inputs/{k}"][i] for k in ("environment_records","contacts","contact_mask")}
        metadata = {k: _text(h[f"metadata/{k}"][i]) for k in h["metadata"]}
        metadata.update(native_cell_area=1/total, native_cell_ids=ids, fields_normalized=self.normalize,
                        contact_ids=tuple(h["provenance/contact_id"].asstr()[i].tolist()),
                        dataset_payload_sha256=self.dataset_payload_sha256, dataset_plan_hash=self.dataset_plan_hash,
                        normalization_train_membership_sha256=self.normalization_train_membership_sha256)
        return {"inputs": inputs, "query_xy": xy, "query_weight": np.full(len(ids),1/len(ids), dtype=np.float64),
                "fields": fields, "metadata": metadata}
