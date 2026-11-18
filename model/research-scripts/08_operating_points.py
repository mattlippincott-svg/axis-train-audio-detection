"""Compare fixed validation false-positive budgets, including a two-head ensemble."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv
import json
import collections
from pathlib import Path
from importlib import import_module
import numpy as np
from common import MANIFEST, OUT


def main():
    from tf_keras.models import load_model
    old=import_module('03_train_head'); f=import_module('06_foundation')
    root=Path(OUT).parent/'experiments'/'foundation'
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r['source']=='esc50': groups[r['id'].split('-')[1]].append(i)
        elif r['source']=='fsd50k': groups[r['id'].split('/')[-1].replace('.wav','')].append(i)
    excluded={i for ids in groups.values() if len({rows[j]['source'] for j in ids})>1 for i in ids}
    d=dict(np.load(Path(OUT)/'embeddings.npz'))
    keep=(d['aug']==0)&(d['split']!=0)&~np.isin(d['clip'],list(excluded))
    d={k:v[keep] for k,v in d.items()}
    w,b=old.yamnet_classifier();x=old.features(d['emb'],w,b)
    pred={}
    for name,path in [('baseline',root/'baseline'/'head.h5'),('candidate',root/'candidate-head.h5')]:
        m=load_model(path,compile=False)
        pred[name]=np.concatenate([m(x[i:i+2048],training=False).numpy().ravel() for i in range(0,len(x),2048)])
    pred['ensemble']=(pred['baseline']+pred['candidate'])/2
    result={}
    for name,p in pred.items():
        _,y,s=f.summarize(p,d,rows,1,.509)
        # Budget declared before examining these comparisons: <=0.5% negative
        # validation clips. Never select an operating point from test results.
        grid=np.linspace(.01,.99,197)
        eligible=[t for t in grid if f.metrics(y,s,t)['negative_clip_false_positive_rate']<=.005]
        t=max(eligible,key=lambda t:(f.metrics(y,s,t)['recall'],-float(t)))
        result[name]={'threshold':float(t),'validation':f.summarize(p,d,rows,1,t)[0],
                      'test':f.summarize(p,d,rows,2,t)[0]}
    np.savez_compressed(root/'evaluation-predictions.npz',**d,**pred)
    (root/'operating-points.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{'threshold':v['threshold'],'validation':v['validation']['all'],'test':v['test']['all']} for k,v in result.items()},indent=2))


if __name__=='__main__': main()
