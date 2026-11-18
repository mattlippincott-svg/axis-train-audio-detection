"""Train an isolated clatter expert from exact HUMAN_REVIEWED intervals.

No reserved-corpus audio is scored. Rolling-only examples stay diagnostic
train positives, never generic non-train negatives. No deployment.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,pickle,sys,hashlib
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from sklearn.linear_model import LogisticRegression
from common import DATA,OUT,MANIFEST,read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path(OUT).parent/'experiments'/'reviewed-clatter'
CORPUS=ROOT.parent/'rail-corpus-v2'
CHANNEL=ROOT.parent/'channel-robust'

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    f=import_module('06_foundation');channel=import_module('13_channel_robust')
    ROOT.mkdir(exist_ok=True)
    labels_bytes=(CORPUS/'human-interval-labels.json').read_bytes()
    labels=json.loads(labels_bytes)['labels'];catalog=json.loads((CORPUS/'catalog-audited.json').read_text())
    canonical={r['id']:r for r in catalog['sources']}
    assert all(r['review_status']=='HUMAN_REVIEWED' and canonical[r['source_id']]['role']=='train_review' for r in labels)
    assert 535384 not in [r['source_id'] for r in labels]
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    d=dict(np.load(CHANNEL/'features.npz'));rng=np.random.default_rng(20261106)
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    label_hash=hashlib.sha256(labels_bytes).hexdigest()
    cache=ROOT/'reviewed-features.npz'
    if cache.exists() and json.loads((ROOT/'cache-info.json').read_text())['label_sha256']!=label_hash:raise RuntimeError('Labels changed: use a new experiment directory')
    if not cache.exists():
        arrays={k:[] for k in ('x','source','version','level')}
        for r in labels:
            sid=r['source_id'];w=read_wav16(CORPUS/'sources'/str(sid)/'audio.wav')[round(r['start_s']*16000):round(r['end_s']*16000)]
            assert len(w)>=15600
            windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy()
            for v in range(8):
                raw=windows.copy()
                if v:
                    raw=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),raw,axis=1).astype(np.float32)
                    if v>1:raw=lfilter([1,-rng.uniform(.3,.8)],[1],raw,axis=1).astype(np.float32)
                    raw*=10**(-(24 if v==1 else rng.uniform(8,28))/20)
                normal,level=channel.condition(raw);x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
                for k,value in dict(x=x.astype(np.float16),source=np.full(len(x),sid),version=np.full(len(x),v),level=level).items():arrays[k].append(value)
        np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in arrays.items()})
        (ROOT/'cache-info.json').write_text(json.dumps({'label_sha256':label_hash,'scope':'only exact reviewed intervals; original unboosted audio'},indent=2))
    reviewed=dict(np.load(cache));positive_ids=[r['source_id'] for r in labels if r['rhythmic_clatter']=='present']
    positive=(np.isin(reviewed['source'],positive_ids))&(reviewed['level']>=-70)
    pw=np.zeros(positive.sum());ps=reviewed['source'][positive]
    # Two pure reviewed clatter sources each receive twice the mixed-source mass.
    for sid in positive_ids:
        r=next(r for r in labels if r['source_id']==sid);mass=.5 if r['rail_squeal']=='present' else 1.
        pw[ps==sid]=mass/max(1,(ps==sid).sum())
    old=dict(np.load(ROOT.parent/'rail-sounds'/'external-features.npz'))
    tags=old['x'][:,1024:];weak=np.isin(old['source'],[769890,199536])&(old['level']>=-65)&(tags[:,[324,325,195,196,197]].max(1)<.1)&(tags[:,327]<.2)&(tags[:,:6].max(1)<.2)
    weak_w=np.zeros(weak.sum());ws=old['source'][weak]
    for sid in (769890,199536):weak_w[ws==sid]=.25/max(1,(ws==sid).sum())
    px=np.concatenate([reviewed['x'][positive],old['x'][weak]]);pw=np.r_[pw,weak_w];pw/=pw.sum()
    excluded_train=catalog['future_training_excluded_legacy_rows']
    neg=(d['split']==0)&(d['label']==0)&(d['level']>=-70)&~np.isin(d['clip'],excluded_train)
    nx=d['x'][neg];nc=d['clip'][neg];nw=np.ones(len(nx),np.float64)
    for c in np.unique(nc):
        m=nc==c;nw[m]/=m.sum()
        if any(t.lower() in rows[int(c)]['labels'].lower() for t in ('Skateboard','Engine','Truck','Bus','Wind','Water','Mechanism','Siren','Bicycle')):nw[m]*=3
    nw/=nw.sum()
    # Earlier TRAIN engine anchors supply non-clatter confusers, not held-out panel.
    engine=dict(np.load(ROOT.parent/'low-band'/'expert-data.npz'));em=engine['y']==1
    ew=engine['weight'][em].astype(np.float64);ew/=ew.sum()
    nx=np.concatenate([nx,engine['x'][em,:1545]]);nw=np.r_[nw*.8,ew*.2]
    tx=np.concatenate([nx,px]).astype(np.float32);ty=np.r_[np.zeros(len(nx)),np.ones(len(px))]
    weights=np.r_[nw,pw]*len(ty)/2
    expert=LogisticRegression(C=.03,max_iter=750,solver='lbfgs',random_state=20261106)
    expert.fit(tx,ty,sample_weight=weights)
    with open(ROOT/'clatter.pkl','wb') as h:pickle.dump(expert,h)
    base=models.load_model(CHANNEL/'head.h5',compile=False)
    def base_predict(x):return np.concatenate([base(x[i:i+2048],training=False).numpy().ravel() for i in range(0,len(x),2048)])
    evaluation_exclusions=catalog['future_evaluation_excluded_legacy_rows']
    selected=(d['split']!=0)&~np.isin(d['clip'],evaluation_exclusions)
    x=d['x'][selected];clip=d['clip'][selected];version=d['version'][selected];level=d['level'][selected];split=d['split'][selected]
    bp=base_predict(x);bp[level < -65]=0;logits=expert.decision_function(x)
    groups=[]
    for c in np.unique(clip):
        for v in (0,1):
            ids=np.flatnonzero((clip==c)&(version==v));groups.append((int(c),v,int(rows[int(c)]['label']),int(split[ids[0]]),ids))
    old_false={(c,v) for c,v,l,s,ids in groups if s==1 and l==0 and f.event_score(bp[ids])>=.869}
    def records(bias):
        cp=sigmoid(logits+bias);cp[level < -65]=0;p=np.maximum(bp,cp)
        return [dict(clip=c,version=v,label=l,split=s,score=f.event_score(p[ids]),base_score=f.event_score(bp[ids]),clatter_score=f.event_score(cp[ids])) for c,v,l,s,ids in groups]
    lo,hi=-30.,6.
    for _ in range(25):
        mid=(lo+hi)/2;rr=records(mid)
        false={(r['clip'],r['version']) for r in rr if r['split']==1 and r['label']==0 and r['score']>=.869}
        if false<=old_false:lo=mid
        else:hi=mid
    bias=lo-.5;rr=records(bias)
    report={'threshold':.869,'bias':bias,'label_sha256':label_hash,'reviewed_positive_sources':positive_ids,
        'mixed_positive_source':216642,'pure_reviewed_clatter_sources':[76078,636899],'weak_positive_sources':[769890,199536],
        'rolling_only_reference_sources':[592835,640306],'negative_recordings':len(np.unique(nc)),
        'excluded_legacy_evaluation_rows':evaluation_exclusions,
        'reserved_source_ids_unscored':[r['id'] for r in catalog['sources'] if r['role']=='reserved_review'],
        'limits':['only twenty seconds of reviewed clatter without squeal','reviewed examples are training/development, not held-out acceptance',
                  'two pure reviewed sources are passenger trains, not US freight','frozen baseline may have learned related creators',
                  'historically reused evaluation','head-only evaluation omits live speech suppression','new branch has no squeal bonus']}
    for name,s in [('validation',1),('evaluation',2)]:
        r=[r for r in rr if r['split']==s];y=np.array([r['label'] for r in r])
        report[name]=f.metrics(y,np.array([r['score'] for r in r]),.869)
        report[name+'_baseline']=f.metrics(y,np.array([r['base_score'] for r in r]),.869)
    rb=base_predict(reviewed['x']);rc=sigmoid(expert.decision_function(reviewed['x'])+bias)
    rb[reviewed['level']<-65]=0;rc[reviewed['level']<-65]=0
    report['reviewed_development_results']=[]
    for r in labels:
        for v in (0,1):
            m=(reviewed['source']==r['source_id'])&(reviewed['version']==v)
            report['reviewed_development_results'].append(dict(source=r['source_id'],role='fit positive' if r['source_id'] in positive_ids else 'rolling diagnostic only',version=v,
                baseline_score=f.event_score(rb[m]),candidate_score=f.event_score(np.maximum(rb[m],rc[m])),clatter_score=f.event_score(rc[m])))
    # Saved microphone audio is a development diagnostic only, never fitted here.
    lab=Path('/lab/work/audio-diagnostics');alignment=json.loads((lab/'ns-061-armed'/'alignment.json').read_text());assert alignment['accepted']
    lo_s,hi_s=alignment['source_first20_capture_interval_s']
    engine=read_wav16(lab/'ns-061-armed'/'capture.wav')[round(lo_s*16000):round(hi_s*16000)]
    report['microphone']=[]
    for name,w in [('Colorado mixed freight',read_wav16(lab/'colorado-replay'/'captured.wav')),('NS engine-only',engine)]:
        windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,lv=channel.condition(windows)
        mx=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
        mb=base_predict(mx);mc=sigmoid(expert.decision_function(mx)+bias);mb[lv < -65]=0;mc[lv < -65]=0
        report['microphone'].append(dict(name=name,baseline_score=f.event_score(mb),candidate_score=f.event_score(np.maximum(mb,mc)),clatter_score=f.event_score(mc)))
    (ROOT/'records.json').write_text(json.dumps(rr,indent=2))
    (ROOT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
