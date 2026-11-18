"""Fit a causal rhythm gate, retaining frozen 0.6.1 and earlier clatter expert.

Training uses only original training negatives and two clatter source anchors.
Validation sets operating bias; historical evaluation is development evidence.
Does not change or deploy the live application.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,pickle,argparse
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import DATA,OUT,MANIFEST,read_wav16
from rail_rhythm import rhythm_features
from rail_fusion import sigmoid

ROOT=Path(OUT).parent/'experiments'/'temporal-rail'
CHANNEL=ROOT.parent/'channel-robust'
SOURCES=ROOT.parent/'rail-sounds'

def variant(w,version):
    if not version:return w
    w=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w).astype(np.float32)
    if version>1:w=lfilter([1,-.65],[1],w).astype(np.float32)
    return w*10**(-24/20)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--continuous-context',action='store_true');args=parser.parse_args()
    suffix='-continuous' if args.continuous_context else ''
    from tf_keras import models
    f=import_module('06_foundation')
    ROOT.mkdir(exist_ok=True)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    d=dict(np.load(CHANNEL/'features.npz'));ext=dict(np.load(SOURCES/'external-features.npz'))
    source_report=json.loads((SOURCES/'results.json').read_text())
    authors=set(source_report['reserved_authors']);meta={}
    for part in ('dev','eval'):meta.update(json.load(open(Path(DATA)/'meta'/f'{part}_clips_info_FSD50K.json')))
    sid=lambda r:r['id'].split('/')[-1].replace('.wav','')
    negative_clips=[int(c) for c in np.unique(d['clip'][(d['split']==0)&(d['label']==0)])
        if meta.get(sid(rows[int(c)]),{}).get('uploader') not in authors and sid(rows[int(c)])!='535384']
    cache=ROOT/'training-rhythm.npz'
    if not cache.exists():
        tx=[];ty=[];tw=[]
        for count,c in enumerate(negative_clips):
            w=read_wav16(Path(DATA)/rows[c]['path'])[:16000*30]
            if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
            ends=np.unique(np.linspace(15600,len(w),min(8,1+(len(w)-15600)//7680)).astype(int))
            for v in range(3):
                tx.append(rhythm_features(variant(w,v),ends));ty.extend([0]*len(ends));tw.extend([1/(3*len(ends))]*len(ends))
            if count%200==0:print('negative rhythm',count,len(negative_clips),flush=True)
        # Same source families as the short-window experiment; no evaluation audio
        # enters training. Temporal augmentation does not claim measured mic response.
        for s in (769890,199536):
            w=read_wav16(SOURCES/f'{s}.wav')[:16000*120]
            for v in range(8):
                m=(ext['source']==s)&(ext['version']==v);tags=ext['x'][m,1024:]
                clean=(tags[:,[324,325,195,196,197]].max(1)<.1)&(tags[:,:6].max(1)<.2)&(tags[:,327]<.2)&(ext['level'][m]>=-65)
                ends=np.round(ext['time'][m]*16000+15600).astype(int)
                # Early partial context is retained, matching causal cold startup.
                a=rhythm_features(variant(w,v),ends)[clean]
                tx.append(a);ty.extend([1]*len(a));tw.extend([len(negative_clips)/(2*8*max(1,len(a)))]*len(a))
        np.savez_compressed(cache,x=np.concatenate(tx),y=ty,weight=tw)
    train=dict(np.load(cache));gate=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000,random_state=20261104))
    gate.fit(train['x'],train['y'],logisticregression__sample_weight=train['weight'])
    with open(ROOT/'rhythm-gate.pkl','wb') as h:pickle.dump(gate,h)
    # Dense evaluation waveform context matches the existing feature cache hop order.
    cache=ROOT/f'evaluation-rhythm{suffix}.npz'
    if not cache.exists():
        selected=d['split']!=0;rx=np.zeros((selected.sum(),train['x'].shape[1]),np.float32)
        dc=d['clip'][selected];dv=d['version'][selected]
        pools={s:np.unique(d['clip'][(d['split']==s)&(d['label']==0)]) for s in (1,2)}
        for count,c in enumerate(np.unique(dc)):
            w=read_wav16(Path(DATA)/rows[int(c)]['path'])[:16000*30]
            if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
            ends=np.arange(15600,len(w)+1,7680)
            if args.continuous_context:
                s=int(d['split'][np.flatnonzero(d['clip']==c)[0]])
                pool=pools[s];previous=int(pool[count%len(pool)])
                if previous==c:previous=int(pool[(count+1)%len(pool)])
                bed=read_wav16(Path(DATA)/rows[previous]['path'])[-64000:]
                if len(bed)<64000:bed=np.tile(bed,int(np.ceil(64000/max(1,len(bed)))))[:64000]
                w=np.r_[bed,w];ends=ends+64000
            for v in (0,1):
                ids=np.flatnonzero((dc==c)&(dv==v));assert len(ids)==len(ends),(c,v,len(ids),len(ends))
                rx[ids]=rhythm_features(variant(w,v),ends)
            if count%300==0:print('evaluation rhythm',count,len(np.unique(dc)),flush=True)
        np.savez_compressed(cache,x=rx,clip=dc,version=dv)
    rd=dict(np.load(cache));selected=d['split']!=0
    assert np.array_equal(rd['clip'],d['clip'][selected]) and np.array_equal(rd['version'],d['version'][selected])
    x=d['x'][selected];level=d['level'][selected];split=d['split'][selected]
    clatter=pickle.load(open(SOURCES/'experts.pkl','rb'))['clatter']
    logits=clatter.decision_function(x);rhythm=gate.predict_proba(rd['x'])[:,1]
    base=models.load_model(CHANNEL/'head.h5',compile=False)
    bp=np.concatenate([base(x[i:i+2048],training=False).numpy().ravel() for i in range(0,len(x),2048)])
    bp[level < -65]=0
    groups=[]
    for c in np.unique(rd['clip']):
        for v in (0,1):
            ids=np.flatnonzero((rd['clip']==c)&(rd['version']==v))
            groups.append((int(c),v,int(rows[int(c)]['label']),int(split[ids[0]]),ids))
    def score(bias):
        # No squeal bonus in this experiment: isolate temporal clatter contribution.
        cp=sigmoid(logits+bias)*rhythm
        cp[level < -65]=0
        p=np.maximum(bp,cp)
        return [dict(clip=c,version=v,label=l,split=s,score=f.event_score(p[ids]),base_score=f.event_score(bp[ids]),
                     clatter_score=f.event_score(cp[ids])) for c,v,l,s,ids in groups]
    old_false={(c,v) for c,v,l,s,ids in groups if s==1 and l==0 and f.event_score(bp[ids])>=.869}
    lo,hi=-30.,30.
    for _ in range(25):
        mid=(lo+hi)/2;rr=score(mid)
        false={(r['clip'],r['version']) for r in rr if r['split']==1 and r['label']==0 and r['score']>=.869}
        if false<=old_false:lo=mid
        else:hi=mid
    bias=lo-.5;rr=score(bias)
    report={'context_seconds':4,'continuous_context':args.continuous_context,
            'prefix_policy':'preceding negative recording from same split; sharp join stress test' if args.continuous_context else 'zero cold-start padding',
            'clatter_bias':bias,'threshold':.869,'training_negative_recordings':len(negative_clips),
            'training_windows':len(train['y']),'training_positive_sources':[769890,199536],
            'policy':'clatter probability times rhythm probability; max with frozen baseline; no squeal bonus',
            'calibration':'no new negative validation combinations, then subtract 0.5 logit margin',
            'limits':['two weakly labeled positive source families','cold-start context padding','historically reused evaluation',
                      'waveform rhythm and acoustic augmentations not exactly identical','head-only evaluation omits live speech suppression',
                      'requires streaming and microphone validation before any deployment']}
    for name,s in [('validation',1),('evaluation',2)]:
        r=[r for r in rr if r['split']==s];y=np.array([r['label'] for r in r])
        report[name]=f.metrics(y,np.array([r['score'] for r in r]),.869)
        report[name+'_baseline']=f.metrics(y,np.array([r['base_score'] for r in r]),.869)
    report['targeted']=[dict(r,source=sid(rows[r['clip']])) for r in rr if sid(rows[r['clip']]) in {'316616','232178','382586','420655','87376','95067'}]
    (ROOT/f'records{suffix}.json').write_text(json.dumps(rr,indent=2))
    (ROOT/f'results{suffix}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
