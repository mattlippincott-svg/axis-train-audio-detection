"""Compare the verified shared source/capture interval, with and without raw gating."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import json
from pathlib import Path
from importlib import import_module
import numpy as np
from common import OUT,read_wav16


def main():
    import tensorflow as tf
    f=import_module('06_foundation');root=Path(OUT).parent/'experiments';exp=root/'channel-robust'
    alignment=json.loads((exp/'capture-alignment.json').read_text())
    if not alignment['alignment_accepted']:raise RuntimeError('No accepted source/capture match')
    a=alignment['anchors'][0];offset=int(round(a['source_start_s']*16000))
    source=read_wav16(root/'foundation'/'probes'/'sustained-brakes.wav')[offset:offset+16*16000]
    mic=read_wav16('/lab/work/audio-diagnostics/tim-kahn-535384/capture.wav')[:len(source)]
    source_rms=np.sqrt(np.mean(source.astype(np.float64)**2)+1e-12)
    mic_rms=np.sqrt(np.mean(mic.astype(np.float64)**2)+1e-12)
    waves={'source_native':source,'source_matched_to_capture_rms':source*(mic_rms/source_rms),'microphone':mic}
    paths={'deployed_060':root/'foundation'/'conditioned.tflite'}
    if (exp/'conditioned.tflite').exists():paths['channel_candidate']=exp/'conditioned.tflite'
    report={'source_interval_seconds':[160+a['source_start_s'],160+a['source_start_s']+len(source)/16000],
            'capture_interval_seconds':[0,len(mic)/16000],'models':{}}
    for name,path in paths.items():
        i=tf.lite.Interpreter(model_path=str(path),num_threads=2);i.allocate_tensors()
        ii=i.get_input_details()[0]['index'];pi=next(o['index'] for o in i.get_output_details() if o['shape'][-1]==1)
        stats={}
        for wn,w in waves.items():
            p=[];gated=[]
            for off in range(0,len(w)-15600+1,7680):
                v=w[off:off+15600].astype(np.float32)
                level=10*np.log10(np.mean(v.astype(np.float64)**2)+1e-12)
                i.set_tensor(ii,v.reshape(1,15600));i.invoke();prob=float(i.get_tensor(pi).ravel()[0]);p.append(prob);gated.append(prob if level>=-65 else 0.)
            stats[wn]={'rms_dbfs':float(10*np.log10(np.mean(w.astype(np.float64)**2)+1e-12)),
                       'ungated_3s_score':f.event_score(np.array(p)),'raw_gated_3s_score':f.event_score(np.array(gated))}
        report['models'][name]=stats
    (exp/'aligned-model-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
