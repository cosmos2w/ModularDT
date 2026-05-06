from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend_dir: Path
    web_demo_dir: Path
    demo_root: Path
    src_dir: Path
    storage_dir: Path
    manifest_path: Path
    inverse_manifest_path: Path
    inverse_target_presets_dir: Path
    cache_dir: Path
    inverse_jobs_dir: Path
    device: str
    cors_origins: tuple[str, ...]
    max_inverse_n_samples: int
    max_inverse_verify_top_k: int
    max_inverse_save_verified_top_k: int
    max_concurrent_simulation_jobs: int


def build_settings() -> Settings:
    backend_dir = Path(__file__).resolve().parent
    web_demo_dir = backend_dir.parent
    demo_root = web_demo_dir.parent
    src_dir = demo_root / "src"
    storage_dir = web_demo_dir / "storage"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    storage_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = storage_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    inverse_target_presets_dir = storage_dir / "inverse_target_presets"
    inverse_target_presets_dir.mkdir(parents=True, exist_ok=True)
    inverse_jobs_dir = cache_dir / "inverse_jobs"
    inverse_jobs_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        backend_dir=backend_dir,
        web_demo_dir=web_demo_dir,
        demo_root=demo_root,
        src_dir=src_dir,
        storage_dir=storage_dir,
        manifest_path=storage_dir / "model_manifest.json",
        inverse_manifest_path=storage_dir / "inverse_model_manifest.json",
        inverse_target_presets_dir=inverse_target_presets_dir,
        cache_dir=cache_dir,
        inverse_jobs_dir=inverse_jobs_dir,
        device=os.environ.get("MODULARDT_WEB_DEMO_DEVICE", "auto"),
        cors_origins=(
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ),
        max_inverse_n_samples=int(os.environ.get("MODULARDT_WEB_DEMO_MAX_INVERSE_N_SAMPLES", "512")),
        max_inverse_verify_top_k=int(os.environ.get("MODULARDT_WEB_DEMO_MAX_INVERSE_VERIFY_TOP_K", "64")),
        max_inverse_save_verified_top_k=int(os.environ.get("MODULARDT_WEB_DEMO_MAX_INVERSE_SAVE_TOP_K", "16")),
        max_concurrent_simulation_jobs=int(os.environ.get("MODULARDT_WEB_DEMO_MAX_SIMULATION_JOBS", "1")),
    )


settings = build_settings()
