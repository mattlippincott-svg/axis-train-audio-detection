"""Read-only exact user-provided intermodal reference; never fits or deploys."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import argparse,hashlib,json,pickle,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-intermodal',action='store_true');args=parser.parse_args()
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/intermodal-reference');r=json.loads((root/'reference.json').read_text(encoding='utf-8-sig'));path=root/r['local_audio_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['audio_sha256']
    w=read_wav16(path);assert abs(len(w)/16000-50)<.05
    f=import_module('06_foundation');ch=import_module('13_channel_robust')
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'));mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001);s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    base=models.load_model('experiments/channel-robust/head.h5',compile=False);experts={}
    for name in ('wheel-rail-support-gates','wheel-rail-flatspots-gates')+(('wheel-rail-intermodal-gates',) if args.include_intermodal else ()):
        p=Path('experiments')/name;experts[name]=(pickle.load(open(p/'experts.pkl','rb'))['rhythm'],json.loads((p/'results.json').read_text())['selected']['bias'])
    windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();records=[]
    for gain in (0,-24,-48):
        normal,level=ch.condition(windows*10**(gain/20));x=np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)])
        b=base(x,training=False).numpy().ravel();values={'installed_baseline':b}
        for name,(expert,bias) in experts.items():values[name]=np.maximum(b,sigmoid(expert.decision_function(x)+bias))
        scores={}
        for name,p in values.items():
            gated=p.copy();gated[level < -65]=0;scores[name]=dict(sustained_score=f.event_score(gated),ungated_score=f.event_score(p))
        records.append(dict(digital_gain_db=gain,median_raw_dbfs=float(np.median(level)),above_gate_fraction=float((level>=-65).mean()),models=scores))
    report=dict(source_id=r['source_id'],original_interval_s=[35,85],local_duration_s=len(w)/16000,records=records,scope='fixed existing models on a newly acquired development reference; not training or microphone evaluation',limits=['post-recording attenuation is not microphone simulation','source has previously been used in user lab playback; no independent-source claim','mechanical layout is user-described, not a verified diagnosis','overlap classes remain unannotated; no absence inferred from model tags'])
    if args.include_intermodal:report['intermodal_score_scope']='This new model fitted the exact excerpt; its scores are training diagnostics, not generalization evidence.'
    (root/('diagnostic-after-fit.json' if args.include_intermodal else 'diagnostic.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
