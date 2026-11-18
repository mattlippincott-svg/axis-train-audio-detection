"""Compare exported models on source probes and recorded lab microphone audio."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import json
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
from common import OUT, read_wav16


def main():
    import tensorflow as tf
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['foundation','channel-robust'],default='foundation')
    args=parser.parse_args()
    f=import_module('06_foundation');foundation=Path(OUT).parent/'experiments'/'foundation'
    root=Path(OUT).parent/'experiments'/args.experiment
    models={}
    baseline_path=foundation/'baseline'/'traindet.tflite' if args.experiment=='foundation' else foundation/'conditioned.tflite'
    for name,path in [('baseline',baseline_path),('conditioned',root/'conditioned.tflite')]:
        i=tf.lite.Interpreter(model_path=str(path),num_threads=2);i.allocate_tensors()
        models[name]=(i,i.get_input_details()[0]['index'],next(o['index'] for o in i.get_output_details() if o['shape'][-1]==1))
    paths=list((foundation/'probes').glob('*.wav'))
    lab=Path('/lab/work/audio-diagnostics')
    paths += [lab/p for p in ['brakes-isolated/capture.wav','tim-kahn-535384/capture.wav','colorado-replay/captured.wav'] if (lab/p).exists()]
    result={}
    for path in paths:
        wave=read_wav16(path)
        if path.stem=='ns-locomotives':wave=wave[:16000*20]
        variants={}
        for gain in ([0,-24] if foundation in path.parents else [0]):
            w=wave*10**(gain/20)
            preds={name:[] for name in models}
            for off in range(0,max(1,len(w)-15600+1),7680):
                v=w[off:off+15600]
                if len(v)<15600:v=np.pad(v,(0,15600-len(v)))
                level=10*np.log10(np.mean(v.astype(np.float64)**2)+1e-12)
                for name,(i,ii,pi) in models.items():
                    i.set_tensor(ii,v.astype(np.float32).reshape(1,15600));i.invoke()
                    preds[name].append(float(i.get_tensor(pi).ravel()[0]) if level>=-65 else 0.)
            variants[str(gain)]={name:{'peak':max(p),'sustained_3s_score':f.event_score(np.array(p))} for name,p in preds.items()}
        key=str(path.relative_to(foundation)) if foundation in path.parents else str(path.relative_to(lab))
        result[key]=variants
        print(key,variants,flush=True)
    (root/'lab-probe-comparison.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
