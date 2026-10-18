"""Standard-library-only validation of a complete immutable source closure."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def validate_source_manifest(manifest_path,manifest_sha256,source_root):
    manifest=Path(manifest_path).resolve(strict=True);root=Path(source_root).resolve(strict=True)
    if _sha(manifest)!=manifest_sha256:raise ValueError('Frozen source manifest bytes changed.')
    payload=json.loads(manifest.read_text());files=payload.get('files')
    if not isinstance(files,dict) or not files:raise ValueError('Frozen source manifest must map every repository-relative file to SHA256.')
    for relative,expected in files.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not isinstance(expected,str) or len(expected)!=64:
            raise ValueError('Invalid frozen source file identity.')
        actual=root/path
        if actual.is_symlink() or not actual.is_file() or not actual.resolve().is_relative_to(root) or _sha(actual)!=expected:
            raise ValueError(f'Frozen source file is missing or changed: {relative}')
    actual={str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() and not path.is_symlink()
            and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo')}
    if actual!=set(files):raise ValueError('Frozen source closure has added or omitted files outside its manifest.')
    for relative,target in payload.get('links',{}).items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid frozen source link identity.')
        actual=root/path
        if not actual.is_symlink() or str(actual.resolve(strict=True))!=str(Path(target).resolve(strict=True)):
            raise ValueError(f'Frozen source external link changed: {relative}')
    return dict(manifest_path=str(manifest),manifest_sha256=manifest_sha256,source_root=str(root),
                validated_files=len(files),validated_external_links=len(payload.get('links',{})))
