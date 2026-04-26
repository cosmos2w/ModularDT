from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Dict, Optional

from schemas import DesignRequest
from settings import settings


INDEX_PATH = settings.cache_dir / "cache_index.json"


def _model_dump(model) -> Dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def canonical_request_payload(request: DesignRequest) -> Dict:
    payload = _model_dump(request)
    payload["cylinders"] = sorted(
        [{"x": round(float(c["x"]), 10), "y": round(float(c["y"]), 10)} for c in payload["cylinders"]],
        key=lambda item: (item["x"], item["y"]),
    )
    payload["re"] = round(float(payload["re"]), 10)
    return payload


def request_hash(request: DesignRequest) -> str:
    encoded = json.dumps(canonical_request_payload(request), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_index() -> Dict[str, str]:
    if not INDEX_PATH.exists():
        return {}
    try:
        with INDEX_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _write_index(index: Dict[str, str]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    tmp.replace(INDEX_PATH)


def find_cached_job(req_hash: str) -> Optional[str]:
    job_id = _read_index().get(req_hash)
    if not job_id:
        return None
    result_path = settings.cache_dir / job_id / "result.json"
    return job_id if result_path.exists() else None


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]


def register_cache(req_hash: str, job_id: str) -> None:
    index = _read_index()
    index[req_hash] = job_id
    _write_index(index)
