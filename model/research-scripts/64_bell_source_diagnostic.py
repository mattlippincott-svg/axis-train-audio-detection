"""Read-only installed-model diagnostic on two metadata-described train bells."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,importlib
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import read_wav16

def main():
    f=importlib.import_module('06_foundation')
    i=tf.lite.Interpreter(model_path='experiments/wheel-rail-intermodal-gates/intermodal-lab.tflite',num_threads=2)
    i.allocate_tensors();inp=i.get_input_details()[0]['index'];out=i.get_output_details()
    pi=next(o['index'] for o in out if o['shape'][-1]==1)
    si=next(o['index'] for o in out if o['shape'][-1]==521)
    names=[r['display_name'] for r in csv.DictReader(open('yamnet/yamnet_class_map.csv'))]
    results=[]
    sources=[('28777','train',Path('data/clips/fsd50k/dev/28777.wav'),None),
        ('127451','val',Path('data/clips/fsd50k/dev/127451.wav'),None),
        ('ns-061-armed-marked-bell','development',Path('/lab/work/audio-diagnostics/ns-061-armed/capture.wav'),(68,76)),
        ('ns-061-armed-engine-first20','development',Path('/lab/work/audio-diagnostics/ns-061-armed/capture.wav'),(23.23025,43.23025))]
    for sid,split,path,interval in sources:
        w=read_wav16(path)
        if interval:w=w[round(interval[0]*16000):round(interval[1]*16000)]
        original_duration=len(w)/16000;w=w[:16000*60];records=[]
        for start in range(0,len(w)-15600+1,7680):
            x=w[start:start+15600];i.set_tensor(inp,x[None].astype(np.float32));i.invoke()
            s=i.get_tensor(si).ravel();top=np.argsort(s)[-3:][::-1]
            records.append(dict(start_s=start/16000,head=float(i.get_tensor(pi).ravel()[0]),
                bell=float(s[195:198].max()),train_horn=float(s[324:326].max()),
                level_dbfs=float(20*np.log10(max(1e-12,np.sqrt(np.mean(x*x))))),
                top=[dict(name=names[k],score=float(s[k])) for k in top]))
        p=np.array([r['head'] if r['level_dbfs']>=-90 else 0 for r in records])
        results.append(dict(source_id=sid,split=split,original_duration_s=original_duration,
            capture_interval_s=interval,
            analyzed_duration_s=len(w)/16000,sustained_head=f.event_score(p),
            bell_peak=max((r['bell'] for r in records),default=0),records=records))
        print(sid,results[-1]['sustained_head'],results[-1]['bell_peak'],flush=True)
    dest=Path('experiments/bell-audit')
    (dest/'source-diagnostic.json').write_text(json.dumps(dict(
        scope='Existing model; no fitting, test scoring or deployment. Source descriptions are not verified bell intervals. Microphone bell interval is conservatively between user mark and bell-fading messages, with approximate wall-time mapping; overlapping sound classes remain unknown. Engine first20 interval uses existing source alignment. Sustained head calculation excludes independent horn triggers.',
        results=results),indent=2))

if __name__=='__main__':main()
