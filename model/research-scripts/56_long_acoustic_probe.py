"""Exploratory longer acoustic field, fixed weights; not a qualified model."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,pickle,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/long-acoustic-probe');root.mkdir(exist_ok=True);f=import_module('06_foundation');ch=import_module('13_channel_robust')
    modelroot=Path('experiments/wheel-rail-flatspots-gates');selected=json.loads((modelroot/'results.json').read_text())['selected'];expert=pickle.load(open(modelroot/'experts.pkl','rb'))['rhythm'];base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    lab=Path('/lab/work/audio-diagnostics');a=json.loads((lab/'ns-061-armed/alignment.json').read_text());lo,hi=a['source_first20_capture_interval_s'];assert a['accepted']
    waves={'colorado':read_wav16(lab/'colorado-replay/captured.wav'),'engine':read_wav16(lab/'ns-061-armed/capture.wav')[round(lo*16000):round(hi*16000)]}
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.);results=[]
    for frames in (96,192,288):
        length=400+160*(frames-1);pin=layers.Input((frames,64));s,e=yamnet.yamnet(pin,params.Params(patch_window_seconds=frames*.01));core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
        @tf.function(input_signature=[tf.TensorSpec([None,length],tf.float32)])
        def embed(w):
            lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001);s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
        for name,w in waves.items():
            original=dict(np.load(Path('experiments/colorado-window-diagnostic')/f'{name}-features.npz'));b=base(original['x'].astype(np.float32),training=False).numpy().ravel();level=original['level'];ends=np.arange(15600,len(w)+1,7680)
            windows=np.stack([np.pad(w[max(0,end-length):end],(max(0,length-end),0)) for end in ends]);normal,_=ch.condition(windows)
            x=np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)]);cp=sigmoid(expert.decision_function(x)+selected['bias']);p=np.maximum(b,cp);p[level < -65]=0
            np.savez_compressed(root/f'{name}-{frames}-features.npz',x=x.astype(np.float16),level=level)
            r=dict(name=name,mel_frames=frames,acoustic_seconds=length/16000,score=f.event_score(p),ungated_rhythm_score=f.event_score(cp),rhythm_peak=float(cp.max()),trigger=bool(f.event_score(p)>=.869));results.append(r);print(r,flush=True)
    (root/'results.json').write_text(json.dumps(dict(results=results,limits=['probe-guided development only','existing expert was trained on 96-frame acoustic fields','longer fields require validation and recalibration before any acceptance or deployment','same frozen YAMNet weights; original baseline branch and current raw level gate retained']),indent=2))

if __name__=='__main__':main()
