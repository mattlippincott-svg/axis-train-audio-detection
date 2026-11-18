"""Fixed-model broader diagnostics; no fitting, recalibration or deployment."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import argparse,csv,json,pickle,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,MANIFEST,read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/wheel-rail-support-gates');args=parser.parse_args()
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path(args.root);config=json.loads((root/'results.json').read_text());selected=config['selected']
    experts=pickle.load(open(root/'experts.pkl','rb'));f=import_module('06_foundation');ch=import_module('13_channel_robust')
    base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')));d=dict(np.load('experiments/channel-robust/features.npz'))
    def predict(x,level):
        b=np.concatenate([base(x[j:j+2048],training=False).numpy().ravel() for j in range(0,len(x),2048)]);b[level < -65]=0
        c=sigmoid(experts['rhythm'].decision_function(x)+selected['bias']);wp=experts['wheel'].predict_proba(x)[:,1]
        contribution=np.minimum(1,c+selected['support']*wp*c*(c>=.5));contribution[level < selected['min_level_dbfs']]=0
        return b,np.maximum(b,contribution),contribution
    mask=(d['split']==2)&~np.isin(d['clip'],config['excluded_evaluation_rows']);x=d['x'][mask];clip=d['clip'][mask];version=d['version'][mask]
    b,p,c=predict(x,d['level'][mask]);records=[]
    for ci in np.unique(clip):
        for v in (0,1):
            ids=np.flatnonzero((clip==ci)&(version==v));records.append(dict(clip=int(ci),version=v,label=int(rows[int(ci)]['label']),source=rows[int(ci)]['id'],labels=rows[int(ci)]['labels'],baseline_score=f.event_score(b[ids]),candidate_score=f.event_score(p[ids])))
    y=np.array([bool(r['label']) for r in records]);old=np.array([r['baseline_score'] for r in records]);new=np.array([r['candidate_score'] for r in records])
    report=dict(selected_fixed=selected,evaluation_baseline=f.metrics(y,old,.869),evaluation_candidate=f.metrics(y,new,.869),new_false_combinations=[r for r in records if not r['label'] and r['candidate_score']>=.869 and r['baseline_score']<.869],records=records,microphone_references=[],scope='historically reused development evaluation; fixed model selected before scoring; not independent field acceptance')
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    lab=Path('/lab/work/audio-diagnostics');alignment=json.loads((lab/'ns-061-armed/alignment.json').read_text());assert alignment['accepted'];lo,hi=alignment['source_first20_capture_interval_s']
    examples=[('Colorado mixed freight',read_wav16(lab/'colorado-replay/captured.wav')),('NS engine-only reference',read_wav16(lab/'ns-061-armed/capture.wav')[round(lo*16000):round(hi*16000)])]
    for name,w in examples:
        windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,level=ch.condition(windows);x=np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)])
        b,p,c=predict(x,level)
        report['microphone_references'].append(dict(name=name,seconds=len(w)/16000,baseline_score=f.event_score(b),candidate_score=f.event_score(p),rhythm_score=f.event_score(c),trigger=bool(f.event_score(p)>=.869)))
    (root/'broader-diagnostics.json').write_text(json.dumps(report,indent=2));print('RESULT',report['evaluation_baseline'],report['evaluation_candidate'],'new false',len(report['new_false_combinations']),report['microphone_references'],flush=True)

if __name__=='__main__':main()
