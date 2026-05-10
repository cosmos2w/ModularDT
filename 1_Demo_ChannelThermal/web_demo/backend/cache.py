from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from settings import settings


INDEX_PATH = settings.cache_dir / "cache_index.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def request_hash(payload: Any) -> str:
    text = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_job_id() -> str:
    return uuid.uuid4().hex[:16]


def _read_index() -> Dict[str, str]:
    if not INDEX_PATH.exists():
        return {}
    try:
        with INDEX_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in data.items()}


def find_cached_job(hash_value: str) -> Optional[str]:
    job_id = _read_index().get(str(hash_value))
    if not job_id:
        return None
    if (settings.cache_dir / job_id / "result.json").exists():
        return job_id
    return None


def register_cache(hash_value: str, job_id: str) -> None:
    index = _read_index()
    index[str(hash_value)] = str(job_id)
    with INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
