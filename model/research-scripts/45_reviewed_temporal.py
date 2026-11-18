"""Four-second reviewed-clatter experiment; no deployment or held-out fitting.

New temporal branch has no squeal bonus. Inherited baseline policy is unchanged.
Exact reviewed intervals only; partial context cannot trigger this branch.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,pickle,hashlib
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import DATA,MANIFEST,read_wav16
from rail_rhythm import rhythm_features
from rail_fusion import sigmoid

def main():
    root=Path('experiments/reviewed-temporal');root.mkdir(exist_ok=True)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    cat=json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text())
    excluded=set(cat['future_training_excluded_legacy_rows'])
    excluded.update(json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())['future_training_excluded_legacy_rows'])
    d=dict(np.load('experiments/channel-robust/features.npz'));rng=np.random.default_rng(20261106)
    pool=np.unique(d['clip'][(d['split']==0)&(d['label']==0)&~np.isin(d['clip'],list(excluded))])
    chosen=np.sort(rng.choice(pool,min(600,len(pool)),replace=False))
    def variant(w,v):
        if v==0:return w
        w=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w).astype(np.float32)
        return w if v==1 else lfilter([1,-.65],[1],w).astype(np.float32)
    xs=[];ys=[];weights=[];families=[]
    def add(w,y,mass,family):
        if len(w)<64000:return
        ends=np.unique(np.linspace(64000,len(w),min(12,1+(len(w)-64000)//7680)).astype(int))
        for v in range(3):
            z=rhythm_features(variant(w,v),ends);xs.append(z);ys.extend([y]*len(z));weights.extend([mass/(3*len(z))]*len(z));families.extend([family]*len(z))
    for count,c in enumerate(chosen):
        r=rows[int(c)];mass=3 if any(s in r['labels'].lower() for s in ('truck','engine','mechanism','skateboard','bus','fan')) else 1
        add(read_wav16(Path(DATA)/r['path'])[:16000*30],0,mass,'legacy:'+str(c))
        if count%100==0:print('temporal negatives',count,len(chosen),flush=True)
    labelpaths=[Path('experiments/rail-corpus-v2/human-interval-labels.json'),Path('experiments/us-freight-candidates/commons/human-interval-labels.json')]
    approved=[]
    for path in labelpaths:
        for r in json.loads(path.read_text())['labels']:
            if r['rhythmic_clatter']!='present':continue
            assert r['review_status']=='HUMAN_REVIEWED'
            sid=str(r['source_id']);assert sid!='commons-bnsf-snow-2016' and sid!='535384'
            base=path.parent/'sources'/sid
            w=read_wav16(base/'audio.wav')[round(r['start_s']*16000):round(r['end_s']*16000)]
            add(w,1,1,sid);approved.append(sid)
    folder=Path('/lab/work/audio-diagnostics/paired-training-01');q=json.loads((folder/'quality-and-alignment.json').read_text());assert q['alignment_accepted']
    capture=read_wav16(folder/'capture.wav')
    for seg in q['segments']:
        sid=str(seg['source_id']);label=int(seg['expected_class']=='train')
        w=capture[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)]
        add(w,label,1 if label else .25,sid)
    x=np.concatenate(xs);y=np.array(ys);weight=np.array(weights)
    # Equal total class mass, equal positive family mass across source/capture.
    fam=np.array(families)
    for sid in np.unique(fam[y==1]):
        m=(fam==sid)&(y==1);weight[m]/=weight[m].sum()
    for label in (0,1):weight[y==label]*=(len(y)/2)/weight[y==label].sum()
    classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=1000,random_state=20261106))
    classifier.fit(x,y,logisticregression__sample_weight=weight)
    pickle.dump(classifier,open(root/'rhythm.pkl','wb'))
    # Cached rhythm rows were generated from the same dense native/bandpass
    # waveforms. Only legacy validation rows are read for scoring here.
    rd=dict(np.load('experiments/temporal-rail/evaluation-rhythm.npz'))
    nontrain=d['split']!=0
    assert np.array_equal(rd['clip'],d['clip'][nontrain]) and np.array_equal(rd['version'],d['version'][nontrain])
    m=(d['split'][nontrain]==1)&~np.isin(rd['clip'],cat['future_evaluation_excluded_legacy_rows'])
    dc=rd['clip'][m];dv=rd['version'][m];level=d['level'][nontrain][m];rx=rd['x'][m]
    f=import_module('06_foundation')
    from tf_keras import models
    base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    vx=d['x'][nontrain][m];bp=np.concatenate([base(vx[i:i+2048],training=False).numpy().ravel() for i in range(0,len(vx),2048)]);bp[level < -65]=0
    logits=classifier.decision_function(rx);groups=[]
    for c in np.unique(dc):
        for v in (0,1):
            ids=np.flatnonzero((dc==c)&(dv==v));groups.append((int(c),v,int(rows[int(c)]['label']),ids))
            # Endpoint = .975 + .48*n; require a full four-second context.
            logits[ids[np.arange(len(ids))*.48+.975 < 4]]=-100
    logits[level < -65]=-100
    oldfalse={(c,v) for c,v,label,ids in groups if label==0 and f.event_score(bp[ids])>=.869}
    def records(bias):
        cp=sigmoid(logits+bias);p=np.maximum(bp,cp)
        return [dict(clip=c,version=v,label=label,score=f.event_score(p[ids]),base_score=f.event_score(bp[ids]),temporal_score=f.event_score(cp[ids])) for c,v,label,ids in groups]
    lo,hi=-40.,20.
    for _ in range(24):
        mid=(lo+hi)/2;rr=records(mid)
        false={(r['clip'],r['version']) for r in rr if r['label']==0 and r['score']>=.869}
        if false<=oldfalse:lo=mid
        else:hi=mid
    bias=lo-.5;rr=records(bias);yy=np.array([bool(r['label']) for r in rr]);sc=np.array([r['score'] for r in rr])
    report={'context_seconds':4,'training_negative_clips':[int(c) for c in chosen],'reviewed_positive_sources':approved,'fit_capture':'paired-training-01','comparison_captures_fitted':False,'label_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in labelpaths},'bias':bias,'threshold':.869,'validation':f.metrics(yy,sc,.869),'validation_baseline':f.metrics(yy,np.array([r['base_score'] for r in rr]),.869),'validation_records':rr,'development_comparison':[],'status':'not deployed','limits':['only four reviewed clatter source families','mixed squeal in two positive sources','cold start shorter than four seconds blocks new branch','no new squeal bonus; inherited baseline squeal behavior unconstrained','source-shared comparisons are not independent acceptance','offline speech suppression omitted']}
    for cid in ('paired-training-01','paired-comparison-02','printer-paused-comparison-03'):
        folder=Path('/lab/work/audio-diagnostics')/cid;q=json.loads((folder/'quality-and-alignment.json').read_text());wave=read_wav16(folder/'capture.wav')
        for seg in q['segments']:
            w=wave[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)];ends=np.arange(15600,len(w)+1,7680)
            cp=sigmoid(classifier.decision_function(rhythm_features(w,ends))+bias)
            windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680];level=20*np.log10(np.sqrt(np.mean(windows.astype(np.float64)**2,axis=1)+1e-12))
            cp[(ends<64000)|(level < -65)]=0
            report['development_comparison'].append(dict(capture=cid,source=str(seg['source_id']),expected_class=seg['expected_class'],temporal_branch_score=f.event_score(cp),branch_trigger=bool(f.event_score(cp)>=.869)))
    (root/'results.json').write_text(json.dumps(report,indent=2));print('RESULT',report['validation'],report['validation_baseline'],bias,report['development_comparison'],flush=True)

if __name__=='__main__':main()
