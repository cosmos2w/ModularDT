#!/usr/bin/env python3
"""Atomic Round10 accounting and exclusive physical-device leases.

This module deliberately lives outside HONF_Proj's frozen scientific inventory.
Reservations charge worst-case work while live, then actual work at closeout.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def device_processes(uuid):
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
    return [{'pid':int(pid.strip()),'gpu_uuid':gpu.strip(),'used_memory_mib':memory.strip()}
            for line in output.splitlines() if line.strip()
            for pid,gpu,memory in [line.split(',')] if gpu.strip()==uuid]


class Campaign:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.master = read(self.root / 'receipts/master_reservation_v1.json')
        if self.master['schema'] != 'honf_round10_master_reservation_v1' or self.master['root'] != str(self.root):
            raise ValueError('Round10 master identity differs.')
        self.start = datetime.fromisoformat(self.master['created_utc']).timestamp()
        self.limits = self.master['limits']
        self.ledger_path = self.root / 'receipts/execution_ledger_v1.json'

    def check_storage(self):
        mount = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE,FSTYPE', '-T', '/data2'], text=True)
        if ' '.join(mount.split()) != self.master['mount']:
            raise RuntimeError('Actual /data2 mount differs from Round10 reservation.')
        free = shutil.disk_usage('/data2').free
        if free < self.limits['minimum_free_data2_bytes'] + 128 * 1024**2:
            raise OSError('Round10 disk headroom exhausted.')
        retained = temporary = 0
        for path in self.root.rglob('*'):
            if path.is_file() and not path.is_symlink():
                if 'tmp' in path.relative_to(self.root).parts:
                    temporary += path.stat().st_size
                else:
                    retained += path.stat().st_size
        if retained > self.limits['retained_payload_bytes'] or temporary > self.limits['temporary_payload_bytes'] or retained + temporary > self.limits['peak_new_payload_bytes']:
            raise OSError('Round10 payload budget exhausted.')
        return {'retained_bytes': retained, 'temporary_bytes': temporary, 'free_bytes': free}

    @contextmanager
    def transaction(self):
        with (self.root / 'receipts/ledger.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = read(self.ledger_path) if self.ledger_path.exists() else {'schema': 'honf_round10_execution_ledger_v1', 'operations': {}}
            yield ledger
            atomic(self.ledger_path, ledger)

    def time_accounting(self, ledger, *, now=None, running_as_reserved=True):
        """Use reserved live maxima for admission and elapsed live time for monitoring."""
        now=time.time() if now is None else now
        by_kind={'main':0.,'engineering':0.,'evaluation':0.}
        for row in ledger['operations'].values():
            charged=(row['reserved_seconds'] if running_as_reserved else max(0.,now-row['started_unix'])) if row['status']=='running' else row['actual_seconds']
            by_kind[row['kind']]+=charged
        return {'total_seconds':sum(by_kind.values()),'non_evaluation_seconds':by_kind['main']+by_kind['engineering'],
                'evaluation_seconds':by_kind['evaluation']}

    def reserve(self, operation, *, track, kind, updates, seconds, physical_gpu):
        if kind not in ('main', 'engineering', 'evaluation') or updates < 0 or seconds <= 0:
            raise ValueError('Invalid operation allocation.')
        if str(physical_gpu) not in self.master['authorized_gpus']:
            raise ValueError('GPU is outside the Round10 authorization.')
        self.check_storage()
        if time.time() + seconds > self.start + self.limits['wall_seconds']:
            raise RuntimeError('Operation cannot finish within the Round10 wall budget.')
        with self.transaction() as ledger:
            if operation in ledger['operations']:
                raise ValueError('Operation ID already exists; use a new attempt namespace.')
            rows = list(ledger['operations'].values())
            accounting=self.time_accounting(ledger)
            cap = self.limits['gpu_associated_seconds']
            evaluation_cap=self.limits['gpu_associated_seconds_reserved_for_evaluation']
            non_evaluation_cap = cap - evaluation_cap
            if (accounting['evaluation_seconds'] > evaluation_cap
                    or accounting['non_evaluation_seconds'] > non_evaluation_cap
                    or accounting['total_seconds'] + seconds > cap
                    or (kind != 'evaluation' and accounting['non_evaluation_seconds'] + seconds > non_evaluation_cap)
                    or (kind == 'evaluation' and accounting['evaluation_seconds'] + seconds > evaluation_cap)):
                raise RuntimeError('Round10 associated process-time reservation exhausted.')
            if kind != 'evaluation':
                cap = self.limits['main_scientific_updates' if kind == 'main' else 'discarded_engineering_updates']
                used = sum((r['reserved_updates'] if r['status'] == 'running' else r['actual_updates']) for r in rows if r['kind'] == kind)
                if used + updates > cap:
                    raise RuntimeError('Round10 update reservation exhausted.')
            row = dict(operation=operation, track=track, kind=kind, status='running', pid=os.getpid(),
                       physical_gpu=physical_gpu, gpu_uuid=self.master['authorized_gpus'][str(physical_gpu)],
                       reserved_updates=int(updates), reserved_seconds=float(seconds), started_unix=time.time())
            ledger['operations'][operation] = row
        return row

    def finish(self, operation, *, updates, status, error=None):
        with self.transaction() as ledger:
            row = ledger['operations'][operation]
            if row['status'] != 'running':
                raise ValueError('Operation already closed.')
            row.update(status=status, actual_updates=int(updates), actual_seconds=time.time()-row['started_unix'],
                       finished_unix=time.time(), error=error)
            if updates > row['reserved_updates'] or row['actual_seconds'] > row['reserved_seconds']:
                row['budget_violation'] = True
        return row

    @contextmanager
    def device(self, physical_gpu):
        if str(physical_gpu) not in self.master['authorized_gpus']:
            raise ValueError('Unauthorized physical device.')
        with (self.root / f'receipts/gpu{physical_gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            uuid = self.master['authorized_gpus'][str(physical_gpu)]
            if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
                raise ValueError('CUDA_VISIBLE_DEVICES must equal the reserved UUID.')
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
            if (physical_gpu, uuid) not in [(int(a.strip()), b.strip()) for a,b in (line.split(',') for line in inventory.splitlines())]:
                raise ValueError('Physical GPU index/UUID changed.')
            competitors=[row for row in device_processes(uuid) if row['pid']!=os.getpid()]
            if competitors:raise RuntimeError('Reserved GPU already has a competing compute process: '+str(competitors))
            yield uuid


class Monitor:
    def __init__(self, campaign, operation, output, gpu_uuid, interval=2.):
        self.campaign=campaign; self.operation=operation; self.output=Path(output); self.uuid=gpu_uuid
        self.interval=interval; self.event=threading.Event(); self.thread=None

    def sample(self):
        now=time.time(); reasons=[]
        row={'unix':now,'operation':self.operation,'gpu_uuid':self.uuid}
        try:
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5)
            found=False
            for line in inventory.splitlines():
                uuid,free,total=[x.strip() for x in line.split(',')]
                if uuid==self.uuid:
                    found=True; free=int(free)*1024**2; total=int(total)*1024**2
                    floor=max(8*1024**3,math.ceil(.2*total)); margin=6*1024**3
                    row.update(gpu_free_bytes=free,gpu_total_bytes=total,gpu_floor_bytes=floor,preventive_margin_bytes=margin)
                    if free<floor+margin:reasons.append('gpu_floor_plus_margin')
            if not found:reasons.append('gpu_missing')
            row['compute_processes']=device_processes(self.uuid)
            row['unexpected_compute_processes']=[p for p in row['compute_processes'] if p['pid']!=os.getpid()]
            if row['unexpected_compute_processes']:reasons.append('unexpected_gpu_competitor')
            row.update(self.campaign.check_storage())
            ledger=read(self.campaign.ledger_path); operation=ledger['operations'][self.operation]
            accounting=self.campaign.time_accounting(ledger,now=now,running_as_reserved=False)
            row['actual_time_accounting']=accounting
            evaluation_cap=self.campaign.limits['gpu_associated_seconds_reserved_for_evaluation']
            total_cap=self.campaign.limits['gpu_associated_seconds']
            if accounting['total_seconds']>=total_cap:reasons.append('total_process_time_budget')
            non_evaluation=accounting['non_evaluation_seconds'];evaluation=accounting['evaluation_seconds']
            if (non_evaluation>total_cap-evaluation_cap or
                    (operation['kind']!='evaluation' and non_evaluation>=total_cap-evaluation_cap)):
                reasons.append('non_evaluation_process_time_budget')
            if (evaluation>evaluation_cap or
                    (operation['kind']=='evaluation' and evaluation>=evaluation_cap)):
                reasons.append('evaluation_process_time_budget')
            if now-operation['started_unix']+120>=operation['reserved_seconds']:reasons.append('operation_time_reservation')
            if now-self.campaign.start+120>=self.campaign.limits['wall_seconds']:reasons.append('campaign_wall_budget')
            row['observed']=True
        except Exception as error:
            row.update(observed=False,error=repr(error));reasons.append('monitor_gap')
        row['stop_reasons']=reasons
        if reasons:
            target=self.output/'CLEAN_STOP_REQUEST.json'
            try:
                with target.open('x') as stream:json.dump({'request_id':self.operation,'requested_unix':now,'requested_by':'round10_resource_monitor','reasons':reasons},stream)
            except FileExistsError:pass
        with (self.campaign.root/'receipts/resource_samples_v1.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n')
        return row

    def __enter__(self):
        row=self.sample()
        if row['stop_reasons']:raise RuntimeError('Round10 admission failed: '+str(row['stop_reasons']))
        def run():
            while not self.event.wait(self.interval):self.sample()
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.event.set()
        if self.thread:self.thread.join(timeout=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();campaign=Campaign(args.root)
    print(json.dumps({'storage':campaign.check_storage(),'ledger':read(campaign.ledger_path) if campaign.ledger_path.exists() else {}},sort_keys=True))


if __name__=='__main__':main()
