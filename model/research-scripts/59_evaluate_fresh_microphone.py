"""Compare pre-frozen candidates on a fresh capture; never fit or recalibrate."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import argparse,hashlib,json,pickle,re,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-id',required=True);parser.add_argument('--posthoc-experiment');args=parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+',args.capture_id)
    lab=Path('/lab');folder=lab/'work/audio-diagnostics'/args.capture_id
    state=json.loads((folder/'capture-status.json').read_text(encoding='utf-8-sig'));assert not state['status'].startswith('CANCELLED')
    q=json.loads((folder/'quality-and-alignment.json').read_text());assert q['alignment_accepted'] and all(s['complete'] for s in q['segments'])
    spec=json.loads((lab/'outputs/fresh-wheel-rail-evaluation-models.json').read_text(encoding='utf-8-sig'));assert spec['playback_sha256']==q['playback_sha256']
    from tf_keras import Model,layers,models
    import tensorflow as tf
    ch=import_module('13_channel_robust');f=import_module('06_foundation');sigmoid=import_module('rail_fusion').sigmoid
    corein=layers.Input((96,64));s,e=yamnet.yamnet(corein,params.Params());core=Model(corein,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'));mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001);s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    w=read_wav16(folder/'capture.wav');windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,level=ch.condition(windows);x=np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)])
    base=models.load_model('experiments/channel-robust/head.h5',compile=False);bp=np.concatenate([base(x[j:j+2048],training=False).numpy().ravel() for j in range(0,len(x),2048)])
    raw={'installed_baseline':bp};configs={'installed_baseline':dict(min_level_dbfs=-65,threshold=.869)}
    for item in spec['models']:
        root=Path('experiments')/item['experiment'];weights=root/'experts.pkl';configfile=root/'results.json'
        assert hashlib.sha256(weights.read_bytes()).hexdigest()==item['weights_sha256']
        assert hashlib.sha256(configfile.read_bytes()).hexdigest()==item['selection_sha256']
        config=json.loads(configfile.read_text())['selected'];assert config['support']==0
        expert=pickle.load(open(weights,'rb'))['rhythm'];raw[item['experiment']]=np.maximum(bp,sigmoid(expert.decision_function(x)+config['bias']));configs[item['experiment']]=dict(min_level_dbfs=config['min_level_dbfs'],threshold=.869)
    if args.posthoc_experiment:
        assert re.fullmatch(r'[a-z0-9-]+',args.posthoc_experiment)
        root=Path('experiments')/args.posthoc_experiment;config=json.loads((root/'results.json').read_text())['selected'];assert config['support']==0
        expert=pickle.load(open(root/'experts.pkl','rb'))['rhythm'];raw[args.posthoc_experiment]=np.maximum(bp,sigmoid(expert.decision_function(x)+config['bias']));configs[args.posthoc_experiment]=dict(min_level_dbfs=config['min_level_dbfs'],threshold=.869)
    predictions={};smoothed={}
    for name,p in raw.items():
        p=p.copy();p[level<configs[name]['min_level_dbfs']]=0;predictions[name]=p;smoothed[name]=f.smooth(p)
    starts=np.arange(len(x))*.48;results=[]
    for seg in q['segments']:
        m=(starts>=seg['capture_start_s'])&(starts+.975<=seg['capture_end_s']);assert m.sum()>=7
        modelscores={}
        for name in predictions:
            ss=smoothed[name][m];score=float(np.lib.stride_tricks.sliding_window_view(ss,7).min(1).max())
            modelscores[name]=dict(sustained_score=score,threshold=configs[name]['threshold'],trigger=bool(score>=configs[name]['threshold']),ungated_interval_score=f.event_score(raw[name][m]))
        results.append(dict(source_id=str(seg['source_id']),expected_class=seg['expected_class'],capture_start_s=seg['capture_start_s'],capture_end_s=seg['capture_end_s'],median_window_dbfs=float(np.median(level[m])),above_gate_fraction=float(np.mean(level[m]>=-65)),above_background_db=seg['above_silence_median_db'],models=modelscores))
    summary={}
    for name in predictions:
        summary[name]=dict(train_passages_detected=sum(r['expected_class']=='train' and r['models'][name]['trigger'] for r in results),train_passages=3,non_train_controls_triggered=sum(r['expected_class']=='non_train' and r['models'][name]['trigger'] for r in results),non_train_controls=2)
    report=dict(capture_id=args.capture_id,capture_sha256=hashlib.sha256((folder/'capture.wav').read_bytes()).hexdigest(),alignment_lag_s=q['lag_s'],corroborating_anchors=q['corroborating_anchors'],silence_median_rms_dbfs=q['silence_median_rms_dbfs'],summary=summary,segments=results,scope='pre-frozen candidate comparison; fresh capture excluded from fitting; train source content fitted',limits=['not unseen-source field acceptance','AAC recording path differs from native PipeWire processing','offline head scoring omits live speech suppression','source labels do not exclude unforeseen ambient interference','weak/gated control exposure cannot establish classifier-level rejection'])
    if args.posthoc_experiment:report.update(scope='Post-hoc development comparison: the added candidate was trained after this known capture, but the capture was not fitted; original frozen candidates remain verified against their saved hashes',posthoc_experiment=args.posthoc_experiment)
    out=Path('experiments')/(args.capture_id+('-posthoc-'+args.posthoc_experiment if args.posthoc_experiment else ''));out.mkdir(exist_ok=True);(out/'results.json').write_text(json.dumps(report,indent=2))
    (out/'windows.json').write_text(json.dumps([dict(start_s=float(t),level_dbfs=float(l),**{name:float(predictions[name][i]) for name in predictions}) for i,(t,l) in enumerate(zip(starts,level))],indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
