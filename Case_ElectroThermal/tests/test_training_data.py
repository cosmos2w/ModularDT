"""Archive corruption, partition/normalization and process ownership boundaries."""
import json
import multiprocessing
import pickle
import shutil

import h5py
import numpy as np
import pytest

from hetmf import formal
from hetmf.environment import generate_environment
from hetmf.training_data import HETMFDataset, SPLITS, export_training_h5, verify_training_h5


def _child_sample(dataset, queue):
    sample = dataset[0]
    queue.put((sample["fields"].tolist(), dataset._pid))
    dataset.close()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "release"
    (root / "inputs").mkdir(parents=True)
    (root / "raw").mkdir()
    from hetmf.io import sha256_json
    from hetmf.study_design import hash_plan, resolve_state
    from hetmf.interventions import physical_input_hash
    import hashlib
    rows, scenes, ledger, environments, layouts = [], {}, {}, {}, {}
    for i,split in enumerate(SPLITS):
        sid = f"state{i}"
        environment=generate_environment(100+i,"uniform")
        environment['root_id']=f'e{i}'
        contacts=[{"id":"pad0","x":.2,"y":.3,"r":.07,"gamma":2.+i*.01,"u":.5},
                  {"id":"pad1","x":.7,"y":.6,"r":.08,"gamma":3.,"u":-.5}]
        environments[f'e{i}']=environment
        layouts[f'l{i}']={'root_id':f'l{i}','family':'dispersed','voltage_family':'balanced','contacts':contacts}
        row = {"state_id":sid,"split":split,"environment_root_id":f"e{i}","layout_root_id":f"l{i}",
               "input_hash":physical_input_hash({'environment':environment,'contacts':contacts})}
        rows.append(row)
    plan={"schema":"hetmf_input_plan_v1","kind":"formal_proposal","formal_generation_authorized":False,
          "states":rows,"profile":"standard","counts":{"total_states":len(rows)},
          "split_counts":{split:1 for split in SPLITS},"environments":environments,"layouts":layouts}
    plan['plan_hash']=hash_plan(plan)
    approval={'plan_hash':plan['plan_hash'],'disk_root':str(root)}
    for i,row in enumerate(rows):
        scene=resolve_state(plan,row)
        sid=row['state_id']
        scene['formal_state_id']=sid
        scene['state_id']='f_'+hashlib.sha256(sid.encode()).hexdigest()[:24]
        scenes[sid]=scene
        source=root/'raw'/f"{scene['state_id']}_n8.h5"
        with h5py.File(source,"w") as h:
            h.attrs["solver_sha256"]="solver"
            h.attrs["input_sha256"]=sha256_json(scene)
            for k in ("potential","temperature_rise","q_sheet","q_contact"):
                h[k]=np.arange(64).reshape(8,8)*.01+i
            h["jx_face"]=np.full((8,9),i+.25)
            h["jy_face"]=np.full((9,8),i-.25)
        ledger[(scene['state_id'],8)]={"solver_sha256":"solver"}
    (root / "inputs/formal_plan.json").write_text(json.dumps(plan))
    (root / "inputs/formal_approval.json").write_text(json.dumps(approval))
    monkeypatch.setattr(formal,"load_release_context",lambda r:{"plan":plan,"approval":approval,"root":root})
    monkeypatch.setattr(formal,"resolve_release_row",lambda r,row,context: {**formal.validate_release_source(),
                      "scene":scenes[row["state_id"]],"raw_path":str(root/"raw"/f"{scenes[row['state_id']]['state_id']}_n8.h5")})
    # Eight synthetic rows isolate archive mechanics from full formal panel policy.
    # The production validator is exercised against the actual release separately.
    def fixture_plan(payload):
        if payload.get("plan_hash")!=hash_plan(payload):
            raise ValueError("fixture plan hash mismatch")
        return {}
    monkeypatch.setattr(formal,"validate_formal_plan",fixture_plan)
    monkeypatch.setattr(formal,"_grid",lambda r,a:8)
    monkeypatch.setattr(formal,"_load_accepted_ledger",lambda r:ledger)
    monkeypatch.setattr(formal,"_load_inheritance",lambda p,a:None)
    monkeypatch.setattr(formal,"validate_release_source",lambda *a,**k:{"raw_sha256":"source","physical_audit":{"accepted":True},"approval_hash":sha256_json(approval)},raising=False)
    import hetmf.training_data as td
    monkeypatch.setattr(td,"require_owned_root",lambda p:p)
    monkeypatch.setattr(td,"require_budget",lambda **kwargs:kwargs)
    path = tmp_path / "training.h5"
    receipt=export_training_h5(root,path,retained_budget_bytes=64*2**20,temporary_budget_bytes=64*2**20)
    assert receipt['source_cases_revalidated']==8
    return path


def test_physical_inputs_exact_train_only_stats_and_partition(archive):
    result=verify_training_h5(archive)
    assert result['states']==8 and result['payload_verified']
    with h5py.File(archive) as h:
        assert set(h['inputs'])=={'contacts','contact_mask','environment_records'}
        np.testing.assert_array_equal(h['normalization/fields/mean'][:],h['targets/fields'][0].astype('f8').mean((0,1)))
        np.testing.assert_array_equal(h['normalization/contacts/count'][()],2)
        assert h['normalization'].attrs['fit_split']=='train'
        for i,split in enumerate(SPLITS):
            np.testing.assert_array_equal(h[f'splits/{split}'][:],[i])
        assert h['provenance/contact_id'].asstr()[0,0]=='pad0'
    sample=HETMFDataset(archive)[0]
    assert sample['fields'].shape==(64,5)
    assert sample['inputs']['environment_records'].shape==(144,8)
    assert sample['query_weight'].sum()==1
    assert sample['metadata']['native_cell_area']==1/64


@pytest.mark.parametrize('key', ['targets/fields','inputs/contacts','metadata/state_id','splits/train','normalization/fields/mean'])
def test_changed_payload_rejected(archive,key):
    with h5py.File(archive,'r+') as h:
        d=h[key]
        if key=='metadata/state_id': d[0]='tampered'
        elif key=='splits/train': d[0]=1
        else:
            index=(0,)*d.ndim
            d[index]=d[index]+1
    with pytest.raises(RuntimeError): verify_training_h5(archive)


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_query_sampling_pickle_and_fork_safe_handles(archive,start_method):
    dataset=HETMFDataset(archive,query_count=11,seed=123)
    first=dataset[0]
    np.testing.assert_array_equal(first['metadata']['native_cell_ids'],dataset[0]['metadata']['native_cell_ids'])
    dataset.set_epoch(1)
    assert not np.array_equal(first['metadata']['native_cell_ids'],dataset[0]['metadata']['native_cell_ids'])
    clone=pickle.loads(pickle.dumps(dataset))
    assert clone._handle is None
    np.testing.assert_array_equal(dataset[0]['fields'],clone[0]['fields'])
    parent_pid=dataset._pid
    context=multiprocessing.get_context(start_method)
    queue=context.Queue();process=context.Process(target=_child_sample,args=(dataset,queue));process.start()
    values,child_pid=queue.get(timeout=15);process.join(timeout=15)
    assert process.exitcode==0 and child_pid!=parent_pid
    np.testing.assert_array_equal(values,dataset[0]['fields'])
    dataset.close();clone.close()


def test_export_refuses_budget_and_existing_output(archive):
    with pytest.raises(FileExistsError):
        export_training_h5(archive.parent/'release',archive,retained_budget_bytes=1,temporary_budget_bytes=1)


@pytest.mark.parametrize('attribute,value', [('environment_columns','permuted'),('contact_columns','permuted'),('renderer','wrong'),('units','wrong'),('grid_n',42)])
def test_loader_rejects_incompatible_schema_without_payload_scan(archive,attribute,value):
    with h5py.File(archive,'r+') as h: h.attrs[attribute]=value
    with pytest.raises(RuntimeError): HETMFDataset(archive)


def test_budget_and_failed_source_leave_no_archive(archive, monkeypatch):
    output=archive.parent/'new.h5'
    with pytest.raises(RuntimeError,match='bounds'):
        export_training_h5(archive.parent/'release',output,retained_budget_bytes=1,temporary_budget_bytes=1)
    def rejected(*args,**kwargs): raise RuntimeError('source checksum mismatch')
    monkeypatch.setattr(formal,'validate_release_source',rejected)
    with pytest.raises(RuntimeError,match='source checksum'):
        export_training_h5(archive.parent/'release',output,retained_budget_bytes=64*2**20,temporary_budget_bytes=64*2**20)
    assert not output.exists() and not list(output.parent.glob('new.h5.tmp-*'))


def test_queries_bind_physical_state_across_row_positions(archive):
    from hetmf.training_data import _payload_hash
    other=archive.with_name('reordered.h5')
    shutil.copyfile(archive,other)
    with h5py.File(other,'r+') as h:
        names=[]
        h.visititems(lambda name,obj:names.append(name) if isinstance(obj,h5py.Dataset) and obj.ndim and obj.shape[0]==8 and name.split("/")[0] in {"metadata","inputs","targets","integrity","provenance"} else None)
        for name in names:
            d=h[name];a,b=d[0],d[1];d[0],d[1]=b,a
        h['splits/train'][0]=1;h['splits/dev'][0]=0
        h.attrs['payload_sha256']=_payload_hash(h)
    a,b=HETMFDataset(archive,query_count=17,seed=42),HETMFDataset(other,query_count=17,seed=42)
    assert a.indices[0]!=b.indices[0]
    aa,bb=a[0],b[0]
    np.testing.assert_array_equal(aa['metadata']['native_cell_ids'],bb['metadata']['native_cell_ids'])
    np.testing.assert_array_equal(aa['fields'],bb['fields'])
    assert len(aa['metadata']['dataset_payload_sha256'])==64
    assert len(aa['metadata']['normalization_train_membership_sha256'])==64
    a.close();b.close()


def test_torch_dataloader_two_workers_cpu_only(archive):
    torch=pytest.importorskip('torch')
    dataset=HETMFDataset(archive,query_count=7)
    # Open a parent handle first: forked workers must replace it safely.
    expected=dataset[0]['fields']
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=2)
    batch=next(iter(loader))
    np.testing.assert_array_equal(batch['fields'][0].numpy(),expected)
    assert not batch['fields'].is_cuda
    dataset.close()


def test_loader_literal_ids_follow_complete_records_and_cpu_predictions(archive):
    import copy
    torch=pytest.importorskip('torch')
    from hetmf.honf_adapter import ElectroThermalHONFAdapter
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    dataset=HETMFDataset(archive,query_count=11,seed=31)
    try:
        sample=dataset[0]
        changed=copy.deepcopy(sample)
        permutation=np.asarray([15,1,8,0,*[i for i in range(16) if i not in {15,1,8,0}]])
        for key in ('contacts','contact_mask'):
            changed['inputs'][key]=changed['inputs'][key][permutation]
        changed['metadata']['contact_ids']=tuple(np.asarray(sample['metadata']['contact_ids'])[permutation])
        adapter=ElectroThermalHONFAdapter()
        before,after=adapter.collate([sample]),adapter.collate([changed])
        # Active pads appear in order pad1,pad0 after the complete record move.
        torch.testing.assert_close(after.scene.module_ids[0],before.scene.module_ids[0,[1,0]],atol=0,rtol=0)
        assert before.scene.module_ids[0,0]!=before.scene.module_ids[0,1]
        model=adapter.build_model(context_dim=16,num_heads=2).eval()
        with torch.no_grad():
            original=model.decode(model.prepare(before.scene),before.query).nonlinear_field
            permuted=model.decode(model.prepare(after.scene),after.query).nonlinear_field
        torch.testing.assert_close(original,permuted,atol=2e-6,rtol=2e-5)
    finally:
        dataset.close()
        torch.set_num_threads(old_threads)


def test_concurrent_final_publisher_cannot_be_overwritten(archive,monkeypatch):
    import os
    from pathlib import Path
    output=archive.with_name('race.h5')
    real_link=os.link
    incumbent=b'other publisher retained output'
    def race(source,destination):
        if Path(destination)==output:
            output.write_bytes(incumbent)
        return real_link(source,destination)
    monkeypatch.setattr(os,'link',race)
    with pytest.raises(FileExistsError):
        export_training_h5(archive.parent/'release',output,retained_budget_bytes=64*2**20,temporary_budget_bytes=64*2**20)
    assert output.read_bytes()==incumbent
    assert not list(output.parent.glob('race.h5.tmp-*'))


def test_loader_rejects_same_path_replacement_with_another_valid_archive(archive):
    import os
    from hetmf.training_data import _case_hash,_payload_hash
    dataset=HETMFDataset(archive)
    dataset[0]
    dataset.close()
    replacement=archive.with_name('replacement.h5')
    shutil.copyfile(archive,replacement)
    with h5py.File(replacement,'r+') as h:
        h['targets/fields'][1,0,0,0]+=1
        h['integrity/case_sha256'][1]=_case_hash(h['inputs/environment_records'][1],h['inputs/contacts'][1],
                                               h['inputs/contact_mask'][1],h['targets/fields'][1])
        h.attrs['payload_sha256']=_payload_hash(h)
    assert verify_training_h5(replacement)['payload_verified']
    os.replace(replacement,archive)
    with pytest.raises(RuntimeError,match='replaced'):
        dataset[0]
    assert dataset._handle is None and dataset._pid is None
    dataset.close()


@pytest.mark.parametrize('tamper',['physical_inputs','metadata','embedded_plan'])
def test_resealed_semantic_mismatch_is_rejected(archive,tamper):
    from hetmf.training_data import _payload_hash,_case_hash
    with h5py.File(archive,'r+') as h:
        if tamper=='physical_inputs':
            h['inputs/contacts'][1,0,0]+=.01
            h['integrity/case_sha256'][1]=_case_hash(h['inputs/environment_records'][1],h['inputs/contacts'][1],
                                                   h['inputs/contact_mask'][1],h['targets/fields'][1])
        elif tamper=='metadata':
            h['metadata/environment_root_id'][1]='different-root'
        else:
            plan=json.loads(h['provenance/formal_plan_json'].asstr()[()])
            plan['states'][1]['input_hash']='0'*64
            h['provenance/formal_plan_json'][()]=json.dumps(plan)
        h.attrs['payload_sha256']=_payload_hash(h)
    with pytest.raises(RuntimeError): verify_training_h5(archive)
