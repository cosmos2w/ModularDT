"""Source file/link inventory is checked before a scientific GPU launch."""
import hashlib
import json
import pytest
from honf_runtime.source_closure import validate_source_manifest


def seal(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'core.py').write_text('value=1\n')
    payload={'files':{'core.py':hashlib.sha256((root/'core.py').read_bytes()).hexdigest()},'links':{}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(payload))
    return root,path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_manifest_rejects_mutation_addition_and_changed_manifest(tmp_path):
    root,path,sha=seal(tmp_path)
    assert validate_source_manifest(path,sha,root)['validated_files']==1
    (root/'new.py').write_text('unexpected=1')
    with pytest.raises(ValueError,match='added or omitted'):validate_source_manifest(path,sha,root)
    (root/'new.py').unlink();(root/'core.py').write_text('value=2\n')
    with pytest.raises(ValueError,match='missing or changed'):validate_source_manifest(path,sha,root)
    path.write_text('{}')
    with pytest.raises(ValueError,match='manifest bytes'):validate_source_manifest(path,sha,root)


def test_source_manifest_rejects_external_source_file_link(tmp_path):
    root,path,sha=seal(tmp_path);external=tmp_path/'outside.py';external.write_text('value=1\n')
    (root/'core.py').unlink();(root/'core.py').symlink_to(external)
    with pytest.raises(ValueError,match='missing or changed'):validate_source_manifest(path,sha,root)
