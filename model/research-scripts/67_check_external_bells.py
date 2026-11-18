"""Frozen-model external bell and crossing diagnostics; no training."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import hashlib,importlib,json,pickle,sys
from pathlib import Path
import numpy as np
from common import read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/bell-external-check')
    for p,h in json.loads((root/'frozen-candidate.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    f=importlib.import_module('06_foundation');ch=importlib.import_module('13_channel_robust')
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights('data/yamnet.h5')
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001);s,e=core(lm,training=False)
        return tf.concat([e,s],axis=1)
    base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    rhythm=pickle.load(open('experiments/wheel-rail-intermodal-gates/experts.pkl','rb'))['rhythm']
    rbias=json.loads(Path('experiments/wheel-rail-intermodal-gates/results.json').read_text())['selected']['bias']
    bell=pickle.load(open('experiments/bell-candidate/expert.pkl','rb'));bias=json.loads(Path('experiments/bell-candidate/results.json').read_text())['bias']
    results=[]
    for r in json.loads((root/'catalog.json').read_text())['sources']:
        if not r.get('downloaded'):continue
        path=root/'sources'/str(r['id'])/'audio.wav';w=read_wav16(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['audio_sha256']
        raw=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,level=ch.condition(raw)
        x=np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)])
        bp=base(x,training=False).numpy().ravel();current=np.maximum(bp,sigmoid(rhythm.decision_function(x)+rbias))
        extra=sigmoid(bell.decision_function(x)+bias);candidate=np.maximum(current,extra)
        for p in (current,candidate,extra):p[level < -90]=0
        result=dict(source_id=r['id'],creator=r['creator'],role=r['role'],duration_s=len(w)/16000,
            current_sustained=f.event_score(current),candidate_sustained=f.event_score(candidate),new_expert_sustained=f.event_score(extra),
            threshold=.869,label_status='UNREVIEWED; no interval ground truth inferred from title or model',
            windows=[dict(start_s=j*.48,current=float(current[j]),candidate=float(candidate[j]),new_expert=float(extra[j]),horn=float(x[j,1024+324:1024+326].max())) for j in range(len(x))])
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='windows'}),flush=True)
    (root/'diagnostic.json').write_text(json.dumps(dict(results=results,scope='Frozen candidate versus current head on newly acquired development references. No fitting or acceptance scoring. Train excerpts overlap horns and other sounds; recognition cannot be attributed to bells. Crossing sound is a separate challenge, not a non-train negative. Horn shortcut omitted.'),indent=2))

if __name__=='__main__':main()
