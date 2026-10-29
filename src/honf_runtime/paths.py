"""Path anchors used by the project runtime.

Paths in committed configuration use an explicit URI-like prefix whenever the
anchor would otherwise be ambiguous.  This avoids depending on the shell's
current working directory and avoids embedding developer-machine locations.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(
    value: str | Path,
    *,
    source_dir: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve an absolute, ``project://``, ``config://``, or relative path."""

    raw = str(value)
    root = Path(project_root).expanduser().resolve()
    if raw.startswith("project://"):
        return (root / raw.removeprefix("project://")).resolve()
    if raw.startswith("config://"):
        if source_dir is None:
            raise ValueError("config:// paths require source_dir.")
        return (Path(source_dir).expanduser().resolve() / raw.removeprefix("config://")).resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    if source_dir is not None:
        candidate = (Path(source_dir).expanduser().resolve() / path).resolve()
        if candidate.exists():
            return candidate
    return (root / path).resolve()


def project_relative(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> str:
    """Return a stable project-relative display path when possible."""

    resolved = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)
