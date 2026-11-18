"""Untrained engine panel plus source-aligned saved microphone diagnostics."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import json,argparse
from pathlib import Path
from importlib import import_module
import numpy as np
from common import OUT,read_wav16


def main():
    import tensorflow as tf
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=['upper-finetune','engine-anchors','engine-diverse'],default='upper-finetune');args=parser.parse_args()
    root=Path(OUT).parent/'experiments'/args.experiment;panel=Path(OUT).parent/'experiments'/'upper-finetune'/'engine-panel';f=import_module('06_foundation')
    paths={'061':Path(OUT).parent/'experiments'/'channel-robust'/'conditioned.tflite','upper':root/'conditioned.tflite'}
    models={}
    for name,path in paths.items():
        i=tf.lite.Interpreter(model_path=str(path),num_threads=2);i.allocate_tensors();oo=i.get_output_details()
        models[name]=(i,i.get_input_details()[0]['index'],next(o['index'] for o in oo if o['shape'][-1]==1))
    waves={p.stem:read_wav16(p)[:16000*30] for p in panel.glob('*.wav')}
    waves['ns_source_engine']=read_wav16(Path(OUT).parent/'experiments'/'foundation'/'probes'/'ns-locomotives.wav')[:16000*20]
    lab=Path('/lab/work/audio-diagnostics');alignment=json.loads((lab/'ns-061-armed'/'alignment.json').read_text())
    if not alignment['accepted']:raise RuntimeError('Reference capture is not aligned')
    lo,hi=alignment['source_first20_capture_interval_s']
    waves['ns_microphone_engine']=read_wav16(lab/'ns-061-armed'/'capture.wav')[round(lo*16000):round(hi*16000)]
    waves['brake_microphone']=read_wav16(lab/'tim-kahn-535384'/'capture.wav')[:16000*16]
    result={}
    for name,w in waves.items():
        scores={m:[] for m in models};gated={m:[] for m in models}
        for off in range(0,len(w)-15600+1,7680):
            v=w[off:off+15600].astype(np.float32);level=10*np.log10(np.mean(v.astype(np.float64)**2)+1e-12)
            for model,(i,ii,pi) in models.items():
                i.set_tensor(ii,v.reshape(1,15600));i.invoke();p=float(i.get_tensor(pi).ravel()[0]);scores[model].append(p);gated[model].append(p if level>=-65 else 0.)
        result[name]={m:{'peak':max(scores[m]),'ungated_sustained_score':f.event_score(np.array(scores[m])),
                       'gated_sustained_score':f.event_score(np.array(gated[m]))} for m in models}
        print(name,result[name],flush=True)
    (root/'diagnostics.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
