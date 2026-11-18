"""Read-only diagnostics on saved lab audio, never capture or deployment.

Colorado replay is a development reference with mixed train sounds, not an
isolated ground-truth clatter interval. NS first twenty source seconds are the
previously aligned engine-only diagnostic. Contaminated 535384 is excluded.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,pickle,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,OUT,read_wav16
from rail_rhythm import rhythm_features
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import yamnet,params

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path(OUT).parent/'experiments'/'temporal-rail'
    report=json.loads((root/'results.json').read_text());bias=report['clatter_bias']
    gate=pickle.load(open(root/'rhythm-gate.pkl','rb'))
    clatter=pickle.load(open(root.parent/'rail-sounds'/'experts.pkl','rb'))['clatter']
    base=models.load_model(root.parent/'channel-robust'/'head.h5',compile=False)
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    f=import_module('06_foundation');channel=import_module('13_channel_robust')
    lab=Path('/lab/work/audio-diagnostics')
    colorado=read_wav16(lab/'colorado-replay'/'captured.wav')
    alignment=json.loads((lab/'ns-061-armed'/'alignment.json').read_text());assert alignment['accepted']
    lo,hi=alignment['source_first20_capture_interval_s']
    engine=read_wav16(lab/'ns-061-armed'/'capture.wav')[round(lo*16000):round(hi*16000)]
    results=[]
    for name,w in [('Colorado mixed freight microphone replay',colorado),('NS aligned engine-only microphone interval',engine)]:
        ends=np.arange(15600,len(w)+1,7680)
        windows=np.stack([w[end-15600:end] for end in ends]);normal,level=channel.condition(windows)
        x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
        bp=base(x,training=False).numpy().ravel();rp=gate.predict_proba(rhythm_features(w,ends))[:,1]
        cp=sigmoid(clatter.decision_function(x)+bias)*rp
        bp[level < -65]=0;cp[level < -65]=0;p=np.maximum(bp,cp)
        # Diagnostic only: class scores aren't independent verification that horns
        # and bells were physically absent. This isolates the new contribution.
        no_cue=x[:,1024:][:,[324,325,195,196,197]].max(1)<.1
        isolated_cp=np.where(no_cue,cp,0)
        r=dict(recording=name,seconds=len(w)/16000,baseline_score=f.event_score(bp),candidate_score=f.event_score(p),
               clatter_score=f.event_score(cp),low_horn_bell_clatter_score=f.event_score(isolated_cp),
               threshold=.869,clatter_peak=float(cp.max()),rhythm_peak=float(rp.max()),
               windows_above_volume_gate=int((level>=-65).sum()),windows=len(p))
        results.append(r)
        safe='colorado' if 'Colorado' in name else 'engine'
        (root/f'microphone-{safe}-windows.json').write_text(json.dumps([dict(end_s=end/16000,base=float(b),clatter=float(c),rhythm=float(r),level=float(l)) for end,b,c,r,l in zip(ends,bp,cp,rp,level)],indent=2))
    (root/'microphone-results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
