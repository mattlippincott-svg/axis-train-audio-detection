"""Train a dedicated 1.935-second rhythmic rail classifier; frozen backbone.

Only approved exact rhythmic intervals and capture 01 fit the positives.
Validation calibrates once; original baseline and -65 dBFS/event rules retained.
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
from common import DATA,MANIFEST,read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

LENGTH=30960

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/long-field-classifier');root.mkdir(exist_ok=True);f=import_module('06_foundation');ch=import_module('13_channel_robust')
    reference=json.loads(Path('experiments/wheel-rail-flatspots-gates/results.json').read_text());rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')));d=dict(np.load('experiments/channel-robust/features.npz'))
    pin=layers.Input((192,64));s,e=yamnet.yamnet(pin,params.Params(patch_window_seconds=1.92));core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'));mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,LENGTH],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001);s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    def encode(w,ends,prefix=None):
        if prefix is None:prefix=np.zeros(LENGTH,np.float32)
        prefix=np.asarray(prefix,np.float32)[-LENGTH:];joined=np.r_[prefix,w]
        windows=np.stack([np.pad(joined[max(0,len(prefix)+int(end)-LENGTH):len(prefix)+int(end)],(max(0,LENGTH-len(prefix)-int(end)),0)) for end in ends])
        normal,level=ch.condition(windows);x=np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)])
        return x.astype(np.float16),level
    def variant(w,v):return w if v==0 else sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w).astype(np.float32)*10**(-24/20)
    labelpaths=[Path('experiments/rail-corpus-v2/human-interval-labels.json'),Path('experiments/us-freight-candidates/commons/human-interval-labels.json'),Path('experiments/us-freight-candidates/review-round-2/human-interval-labels.json'),Path('experiments/us-freight-candidates/review-round-2/human-11-12-labels.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in labelpaths}
    cache=root/'training-features.npz'
    if not cache.exists():
        arrays={k:[] for k in ('x','level','label','family','kind')}
        def add(w,ends,label,family,kind):
            x,level=encode(w,ends)
            for k,v in dict(x=x,level=level,label=np.full(len(x),label),family=np.full(len(x),family),kind=np.full(len(x),kind)).items():arrays[k].append(v)
        negclips=np.unique(d['clip'][(d['split']==0)&(d['label']==0)&~np.isin(d['clip'],reference['excluded_training_rows'])])
        for count,c in enumerate(negclips):
            w=read_wav16(Path(DATA)/rows[int(c)]['path'])[:16000*30]
            if len(w)<LENGTH:w=np.pad(w,(0,LENGTH-len(w)))
            ends=np.unique(np.linspace(LENGTH,len(w),min(8,1+(len(w)-LENGTH)//7680)).astype(int))
            for v in (0,1):add(variant(w,v),ends,0,'legacy:'+str(c),'legacy')
            if count%100==0:print('long-field negatives',count,len(negclips),flush=True)
        approved=[]
        for path in labelpaths:
            for r in json.loads(path.read_text(encoding='utf-8-sig'))['labels']:
                if r.get('review_status')=='HUMAN_REVIEWED' and r.get('training_approved',True) and r.get('rhythmic_clatter')=='present':approved.append((r,path.parent/'sources'/str(r['source_id'])/'audio.wav'))
        assert len(approved)==7
        for r,path in approved:
            w=read_wav16(path)[round(r['start_s']*16000):round(r['end_s']*16000)];ends=np.arange(15600,len(w)+1,7680);ends=ends[ends>=LENGTH]
            for v in range(4):
                raw=w.copy()
                if v:
                    raw=variant(w,1)
                    if v>1:raw=lfilter([1,-.65],[1],raw).astype(np.float32)*10**(.4)
                add(raw,ends,1,str(r['source_id']),'reviewed')
        folder=Path('/lab/work/audio-diagnostics/paired-training-01');q=json.loads((folder/'quality-and-alignment.json').read_text());assert q['alignment_accepted'];wave=read_wav16(folder/'capture.wav')
        for seg in q['segments']:
            w=wave[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)];ends=np.arange(15600,len(w)+1,7680);ends=ends[ends>=LENGTH]
            for gain in (0,6,12):add(w*10**(gain/20),ends,int(seg['expected_class']=='train'),str(seg['source_id']),'microphone')
        np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in arrays.items()});(root/'training-provenance.json').write_text(json.dumps(dict(label_sha256=hashes,negative_clips=[int(c) for c in negclips],positive_sources=[str(r['source_id']) for r,_ in approved],acoustic_frames=192),indent=2))
    assert json.loads((root/'training-provenance.json').read_text())['label_sha256']==hashes
    train=dict(np.load(cache));weight=np.zeros(len(train['x']))
    legacy=(train['kind']=='legacy')&(train['level']>=-70)
    for family in np.unique(train['family'][legacy]):
        m=legacy&(train['family']==family);c=int(family.split(':')[1]);mass=3 if any(t in rows[c]['labels'].lower() for t in ('truck','engine','bus','skateboard','mechanism','fan','siren','bicycle')) else 1;weight[m]=mass/max(1,m.sum())
    weight[legacy]*=.9/max(1e-20,weight[legacy].sum())
    mn=(train['kind']=='microphone')&(train['label']==0)
    for family in np.unique(train['family'][mn]):
        m=mn&(train['family']==family);weight[m]=.1/len(np.unique(train['family'][mn]))/m.sum()
    families=np.unique(train['family'][train['label']==1])
    for family in families:
        source=(train['family']==family)&(train['kind']=='reviewed')&(train['level']>=-70);mic=(train['family']==family)&(train['kind']=='microphone')&(train['label']==1)
        weight[source]=(0.5 if mic.any() else 1)/len(families)/max(1,source.sum())
        if mic.any():weight[mic]=.5/len(families)/mic.sum()
    weight*=len(weight)/2;expert=LogisticRegression(C=.03,max_iter=750,random_state=20261110);expert.fit(train['x'].astype(np.float32),train['label'],sample_weight=weight);pickle.dump(expert,open(root/'expert.pkl','wb'));del train
    base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    def split_features(split):
        cache=root/f'split-{split}-features.npz';m=(d['split']==split)&~np.isin(d['clip'],reference['excluded_evaluation_rows']);dc=d['clip'][m];dv=d['version'][m]
        if not cache.exists():
            x=np.zeros((m.sum(),1545),np.float16);neg=np.unique(dc[d['label'][m]==0])
            for count,c in enumerate(np.unique(dc)):
                w=read_wav16(Path(DATA)/rows[int(c)]['path'])[:16000*30]
                if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
                other=int(neg[count%len(neg)])
                if other==c:other=int(neg[(count+1)%len(neg)])
                prefix=read_wav16(Path(DATA)/rows[other]['path'])[-LENGTH:];ends=np.arange(15600,len(w)+1,7680)
                for v in (0,1):
                    # Filter the synthetic concatenation before splitting so
                    # long-field acoustic context does not reset at the join.
                    joined=variant(np.r_[prefix,w],v);xx,_=encode(joined[len(prefix):],ends,joined[:len(prefix)])
                    ids=np.flatnonzero((dc==c)&(dv==v));assert len(ids)==len(xx);x[ids]=xx
                if count%100==0:print('long-field split',split,count,flush=True)
            np.savez_compressed(cache,x=x,clip=dc,version=dv)
        data=dict(np.load(cache));assert np.array_equal(data['clip'],dc) and np.array_equal(data['version'],dv)
        b=np.concatenate([base(d['x'][m][j:j+2048],training=False).numpy().ravel() for j in range(0,m.sum(),2048)]);level=d['level'][m];b[level < -65]=0;logits=expert.decision_function(data['x'])
        groups=[(int(c),v,int(rows[int(c)]['label']),np.flatnonzero((dc==c)&(dv==v))) for c in np.unique(dc) for v in (0,1)]
        return b,logits,level,groups
    b,logits,level,groups=split_features(1);oldfalse={(c,v) for c,v,label,ids in groups if not label and f.event_score(b[ids])>=.869}
    def records(b,logits,level,groups,bias):
        cp=sigmoid(logits+bias);cp[level < -65]=0;p=np.maximum(b,cp)
        return [dict(clip=c,version=v,label=label,score=f.event_score(p[ids]),base_score=f.event_score(b[ids])) for c,v,label,ids in groups]
    lo,hi=-30.,15.
    for _ in range(24):
        mid=(lo+hi)/2;rr=records(b,logits,level,groups,mid);false={(r['clip'],r['version']) for r in rr if not r['label'] and r['score']>=.869}
        if false<=oldfalse:lo=mid
        else:hi=mid
    bias=lo-.5;rr=records(b,logits,level,groups,bias)
    def metrics(rr,key='score'):return f.metrics(np.array([bool(r['label']) for r in rr]),np.array([r[key] for r in rr]),.869)
    report=dict(acoustic_seconds=LENGTH/16000,bias=bias,validation=metrics(rr),validation_baseline=metrics(rr,'base_score'),validation_records=rr,microphone_references=[],status='not deployed',limits=['frozen backbone pretrained on shorter fields','few approved rhythmic source families','historically reused validation','synthetic acoustic negative prefixes at validation boundaries','live speech suppression omitted'])
    for name in ('colorado','engine'):
        data=dict(np.load(Path('experiments/long-acoustic-probe')/f'{name}-192-features.npz'));original=dict(np.load(Path('experiments/colorado-window-diagnostic')/f'{name}-features.npz'));x=data['x'];level=original['level'];bp=base(original['x'].astype(np.float32),training=False).numpy().ravel();bp[level < -65]=0;cp=sigmoid(expert.decision_function(x)+bias);cp[level < -65]=0;p=np.maximum(bp,cp)
        report['microphone_references'].append(dict(name=name,score=f.event_score(p),trigger=bool(f.event_score(p)>=.869)))
    # A failed validation/probe candidate does not justify more evaluation reads.
    if report['validation']['tp']>=report['validation_baseline']['tp'] and report['microphone_references'][0]['trigger']:
        eb,ee,el,eg=split_features(2);err=records(eb,ee,el,eg,bias);report.update(evaluation=metrics(err),evaluation_baseline=metrics(err,'base_score'),evaluation_records=err)
    (root/'results.json').write_text(json.dumps(report,indent=2));print('RESULT',report['validation'],bias,report['microphone_references'],report.get('evaluation'),flush=True)

if __name__=='__main__':main()
