"""Fixed-setting stream-boundary stress test; never fit or recalibrate."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,pickle
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt
from common import DATA,MANIFEST,read_wav16
from rail_rhythm import rhythm_features
from rail_fusion import sigmoid

def main():
    root=Path('experiments/reviewed-temporal');previous=json.loads((root/'results.json').read_text())
    classifier=pickle.load(open(root/'rhythm.pkl','rb'));bias=previous['bias'];f=import_module('06_foundation')
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    cat=json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text())
    d=dict(np.load('experiments/channel-robust/features.npz'))
    mask=(d['split']==1)&~np.isin(d['clip'],cat['future_evaluation_excluded_legacy_rows'])
    clip=d['clip'][mask];version=d['version'][mask];level=d['level'][mask]
    negatives=np.unique(clip[d['label'][mask]==0]);selected=np.unique(clip)
    cache=root/'continuous-validation-rhythm.npz'
    if cache.exists():
        rd=dict(np.load(cache));assert np.array_equal(rd['clip'],clip) and np.array_equal(rd['version'],version);features=rd['x']
    else:
        features=np.zeros((len(clip),4*(2*len(import_module('rail_rhythm').LAGS)+8)),np.float32)
        for count,c in enumerate(selected):
            w=read_wav16(Path(DATA)/rows[int(c)]['path'])[:16000*30]
            if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
            other=int(negatives[count%len(negatives)])
            if other==c:other=int(negatives[(count+1)%len(negatives)])
            bed=read_wav16(Path(DATA)/rows[other]['path'])[-64000:]
            if len(bed)<64000:bed=np.pad(bed,(64000-len(bed),0))
            ends=np.arange(15600,len(w)+1,7680)+64000;joined=np.r_[bed,w]
            for v in (0,1):
                raw=joined if v==0 else sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),joined).astype(np.float32)*10**(-24/20)
                ids=np.flatnonzero((clip==c)&(version==v));assert len(ids)==len(ends)
                features[ids]=rhythm_features(raw,ends)
            if count%100==0:print('continuous validation',count,len(selected),flush=True)
        np.savez_compressed(cache,x=features,clip=clip,version=version)
    from tf_keras import models
    base=models.load_model('experiments/channel-robust/head.h5',compile=False);vx=d['x'][mask]
    bp=np.concatenate([base(vx[i:i+2048],training=False).numpy().ravel() for i in range(0,len(vx),2048)]);bp[level < -65]=0
    cp=sigmoid(classifier.decision_function(features)+bias);cp[level < -65]=0;p=np.maximum(bp,cp);records=[]
    for c in selected:
        for v in (0,1):
            ids=np.flatnonzero((clip==c)&(version==v));records.append(dict(clip=int(c),version=v,label=int(rows[int(c)]['label']),score=f.event_score(p[ids]),base_score=f.event_score(bp[ids]),temporal_score=f.event_score(cp[ids])))
    y=np.array([bool(r['label']) for r in records]);s=np.array([r['score'] for r in records]);b=np.array([r['base_score'] for r in records])
    result={'validation':f.metrics(y,s,.869),'validation_baseline':f.metrics(y,b,.869),'fixed_bias':bias,'recalibrated':False,
        'new_false_combinations':[r for r in records if not r['label'] and r['score']>=.869 and r['base_score']<.869],
        'policy':'four-second prefix from different non-train recording in same validation split; abrupt join stress test; full temporal context available at first endpoint',
        'limits':['synthetic recording boundaries are not real field streams','historically reused validation','head-only scoring omits speech suppression'], 'records':records}
    (root/'continuous-context-results.json').write_text(json.dumps(result,indent=2));print('RESULT',result['validation'],result['validation_baseline'],'new false',len(result['new_false_combinations']),flush=True)

if __name__=='__main__':main()
