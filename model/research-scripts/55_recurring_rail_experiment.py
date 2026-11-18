"""Fixed recurrence rule; validation calibrates bias, no model fitting.

Two confident observations at least 1.92 seconds apart within 5.76 seconds.
Original base/volume/event rules retained. Synthetic validation prefixes expose
cold-start shortcuts; all calibration is completed before microphone scoring.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,pickle
from pathlib import Path
from importlib import import_module
import numpy as np
from common import MANIFEST
from rail_fusion import sigmoid
from rail_recurrence import recurrence

def main():
    from tf_keras import models
    root=Path('experiments/recurring-rail');root.mkdir(exist_ok=True)
    source=Path('experiments/wheel-rail-flatspots-gates');config=json.loads((source/'results.json').read_text());expert=pickle.load(open(source/'experts.pkl','rb'))['rhythm']
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')));d=dict(np.load('experiments/channel-robust/features.npz'));f=import_module('06_foundation');base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    def arrays(split):
        m=(d['split']==split)&~np.isin(d['clip'],config['excluded_evaluation_rows']);x=d['x'][m];c=d['clip'][m];v=d['version'][m];level=d['level'][m]
        groups=[(int(ci),vi,int(rows[int(ci)]['label']),np.flatnonzero((c==ci)&(v==vi))) for ci in np.unique(c) for vi in (0,1)]
        b=np.concatenate([base(x[j:j+2048],training=False).numpy().ravel() for j in range(0,len(x),2048)]);b[level < -65]=0
        raw=expert.decision_function(x);raw[level < -65]=-np.inf
        recurrent=np.full(len(raw),-np.inf);cold=np.full(len(raw),-np.inf);neg=[(ci,ids) for ci,vi,label,ids in groups if not label]
        for count,(ci,vi,label,ids) in enumerate(groups):
            pool=[other for other,ii in neg if other!=ci];other=pool[count%len(pool)];prefix=next(ii for c0,ii in neg if c0==other)[-11:]
            joined=np.r_[raw[prefix],raw[ids]];recurrent[ids]=recurrence(joined)[len(prefix):];cold[ids]=recurrence(raw[ids])
        return b,raw,recurrent,cold,level,groups
    b,raw,r,cold,level,groups=arrays(1)
    oldfalse={(ci,vi) for ci,vi,label,ids in groups if not label and f.event_score(b[ids])>=.869}
    def records(b,cl,level,gg,bias):
        contribution=sigmoid(cl+bias);contribution[level < -65]=0;p=np.maximum(b,contribution)
        return [dict(clip=ci,version=vi,label=label,score=f.event_score(p[ids]),base_score=f.event_score(b[ids])) for ci,vi,label,ids in gg]
    lo,hi=-30.,15.
    for _ in range(24):
        mid=(lo+hi)/2;rr=records(b,r,level,groups,mid);false={(x['clip'],x['version']) for x in rr if not x['label'] and x['score']>=.869}
        if false<=oldfalse:lo=mid
        else:hi=mid
    bias=lo-.5;rr=records(b,r,level,groups,bias)
    def metrics(rr,key='score'):return f.metrics(np.array([bool(x['label']) for x in rr]),np.array([x[key] for x in rr]),.869)
    report=dict(policy='max of min of two expert logits, separated by >=4 hops, in last 12 windows; raw/current volume gate -65 retained',hop_s=.48,min_lag=4,history=12,bias=bias,threshold=.869,validation=metrics(rr),validation_baseline=metrics(rr,'base_score'),validation_cold_start=metrics(records(b,cold,level,groups,bias)),validation_records=rr,limits=['historically reused development data','no new raw waveform classifier fitting','synthetic feature prefixes are not real continuous streams','sound scores do not prove physical wheel defects','live speech suppression omitted'],status='not deployed')
    eb,eraw,er,ecold,el,eg=arrays(2);err=records(eb,er,el,eg,bias);report.update(evaluation=metrics(err),evaluation_baseline=metrics(err,'base_score'),evaluation_records=err,new_evaluation_false=[x for x in err if not x['label'] and x['score']>=.869 and x['base_score']<.869])
    report['microphone_references']=[]
    for name in ('colorado','engine'):
        data=dict(np.load(Path('experiments/colorado-window-diagnostic')/f'{name}-features.npz'));x=data['x'];level=data['level'];bp=base(x.astype(np.float32),training=False).numpy().ravel();bp[level < -65]=0
        logits=expert.decision_function(x);logits[level < -65]=-np.inf;cl=recurrence(logits);p=np.maximum(bp,sigmoid(cl+bias));p[level < -65]=0
        report['microphone_references'].append(dict(name=name,score=f.event_score(p),trigger=bool(f.event_score(p)>=.869)))
    # Separate clips from captures 02/03: source-shared development only.
    (root/'results.json').write_text(json.dumps(report,indent=2));print('RESULT',report['validation'],report['evaluation'],'new false',len(report['new_evaluation_false']),bias,report['microphone_references'],flush=True)

if __name__=='__main__':main()
