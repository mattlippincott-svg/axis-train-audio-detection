"""Inspect raw level, recognition and persistence separately; no fitting."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,pickle,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def longest(mask):
    best=run=0
    for yes in mask:run=run+1 if yes else 0;best=max(best,run)
    return best

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    f=import_module('06_foundation');ch=import_module('13_channel_robust');sigmoid=import_module('rail_fusion').sigmoid
    root=Path('experiments/colorado-window-diagnostic');root.mkdir(exist_ok=True)
    modelroot=Path('experiments/wheel-rail-flatspots-gates');selected=json.loads((modelroot/'results.json').read_text())['selected'];expert=pickle.load(open(modelroot/'experts.pkl','rb'))['rhythm'];base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'));mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001);s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    lab=Path('/lab/work/audio-diagnostics');align=json.loads((lab/'ns-061-armed/alignment.json').read_text());lo,hi=align['source_first20_capture_interval_s'];assert align['accepted']
    examples=[('colorado',read_wav16(lab/'colorado-replay/captured.wav')),('engine',read_wav16(lab/'ns-061-armed/capture.wav')[round(lo*16000):round(hi*16000)])]
    summary=[]
    for name,w in examples:
        windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,level=ch.condition(windows);x=np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)])
        np.savez_compressed(root/f'{name}-features.npz',x=x.astype(np.float16),level=level)
        b=base(x,training=False).numpy().ravel();cp=sigmoid(expert.decision_function(x)+selected['bias']);gate=level>=-65
        p=np.maximum(b,cp)*gate;raw=np.maximum(b,cp);times=np.arange(len(p))*.48
        blocks=[]
        for start in range(0,int(np.ceil(len(w)/16000)),20):
            m=(times>=start)&(times<start+20)
            if not m.any():continue
            blocks.append(dict(start_s=start,median_level=float(np.median(level[m])),above_gate_fraction=float(gate[m].mean()),rhythm_peak=float(cp[m].max()),rhythm_median=float(np.median(cp[m])),rhythm_confident_windows=int((cp[m]>=.869).sum()),windows=int(m.sum()),combined_event_score=f.event_score(p[m])))
        summary.append(dict(name=name,duration_s=len(w)/16000,above_gate_fraction=float(gate.mean()),rhythm_ungated_event_score=f.event_score(cp),combined_ungated_event_score=f.event_score(raw),combined_gated_event_score=f.event_score(p),raw_rhythm_confident_fraction=float((cp>=.869).mean()),gated_rhythm_confident_fraction=float(((cp>=.869)&gate).mean()),longest_raw_rhythm_confident_run=longest(cp>=.869),longest_smoothed_combined_confident_run=longest(f.smooth(p)>=.869),blocks=blocks))
        (root/f'{name}-windows.json').write_text(json.dumps([dict(start_s=float(t),level_dbfs=float(l),base=float(bb),rhythm=float(c),combined=float(pp)) for t,l,bb,c,pp in zip(times,level,b,cp,p)],indent=2))
    (root/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
