"""Physical GPU/disk/time monitor requesting resumable epoch-boundary stops."""
from __future__ import annotations
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid


class JointResourceMonitor:
    def __init__(self,root,output,reservation,*,interval=1.):
        self.root=Path(root);self.output=Path(output);self.resources=reservation;self.interval=interval
        self.stop_event=threading.Event();self.thread=None;self.last_sample=None
    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True)
        self.thread=threading.Thread(target=self.run,name='round9-resource-monitor',daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=10)
    def request_stop(self,reasons):
        target=self.output/'CLEAN_STOP_REQUEST.json'
        value={'request_id':'round9_resource_'+uuid.uuid4().hex,'requested_unix':time.time(),
               'requested_by':'joint_native_resource_monitor','reasons':reasons}
        try:
            with target.open('x') as stream:json.dump(value,stream,sort_keys=True);stream.write('\n')
        except FileExistsError:pass
    def run(self):
        target=self.root/'receipts/resource_samples.jsonl'
        while not self.stop_event.is_set():
            now=time.time();row={'schema':'round9_resource_sample_v1','unix':now,'output':str(self.output),
                'gap_seconds':None if self.last_sample is None else now-self.last_sample,'interval_seconds':self.interval}
            reasons=[]
            try:
                text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5)
                for line in text.splitlines():
                    fields=[part.strip() for part in line.split(',')]
                    if fields[0]==self.resources['gpu_uuid']:
                        free,total=int(fields[1])*1024**2,int(fields[2])*1024**2
                        floor=max(self.resources['gpu_min_free_bytes'],8*1024**3,math.ceil(.2*total))
                        row.update(gpu_free_bytes=free,gpu_total_bytes=total,gpu_floor_bytes=floor,
                                   preventive_margin_bytes=self.resources['preventive_margin_bytes'])
                        if free<floor+self.resources['preventive_margin_bytes']:reasons.append('gpu_preventive_floor')
                if 'gpu_free_bytes' not in row:reasons.append('gpu_missing')
                row['disk_free_bytes']=shutil.disk_usage('/data2').free
                if row['disk_free_bytes']<self.resources['disk_min_free_bytes']+128*1024**2:reasons.append('disk_headroom')
                if self.last_sample is None or int(now)//10 != int(self.last_sample)//10:
                    retained=temporary=0
                    for path in self.root.rglob('*'):
                        if path.is_file() and not path.is_symlink():
                            size=path.stat().st_size
                            if path.is_relative_to(self.root/'tmp'):temporary+=size
                            else:retained+=size
                    row.update(retained_payload_bytes=retained,temporary_payload_bytes=temporary)
                    if retained>self.resources['retained_disk_limit_bytes']:reasons.append('retained_disk_cap')
                    if temporary>self.resources['temporary_disk_limit_bytes']:reasons.append('temporary_disk_cap')
                    if retained+temporary>self.resources['peak_new_disk_limit_bytes']:reasons.append('peak_disk_cap')
                elapsed=now-self.resources['start_unix'];row['round_elapsed_seconds']=elapsed
                if elapsed+self.resources['closeout_reserved_seconds']>=self.resources['elapsed_limit_seconds']:reasons.append('reserved_closeout')
                row['observed']=True
            except (OSError,subprocess.SubprocessError,ValueError) as error:
                row.update(observed=False,error=str(error));reasons.append('resource_monitor_gap')
            if reasons:self.request_stop(reasons)
            row['clean_stop_reasons']=reasons
            with target.open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
            self.last_sample=now;self.stop_event.wait(self.interval)
