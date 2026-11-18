"""Channel-robust retraining experiment; inherited/source probes never train inputs.

Uses source-family exclusions, fixed split membership, simulated capture channels,
non-horn positive weighting, and negative-validation calibration. Does not deploy.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt
from common import DATA,OUT,MANIFEST,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path(OUT).parent/'experiments'/'channel-robust'
FOUNDATION=Path(OUT).parent/'experiments'/'foundation'
PATCH,HOP=15600,7680


def condition(w):
    rms=np.sqrt(np.mean(w.astype(np.float64)**2,axis=1)+1e-12)
    gain=np.maximum(1.,np.minimum(10**1.8,10**(-26/20)/rms))
    gain=np.minimum(gain,.98/np.maximum(np.max(np.abs(w),axis=1),1e-9))
    return (w*gain[:,None]).astype(np.float32),20*np.log10(rms)


def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models,optimizers,callbacks,utils,regularizers
    f=import_module('06_foundation')
    parser=argparse.ArgumentParser()
    parser.add_argument('--initialize',choices=['baseline','fresh'],default='baseline')
    args=parser.parse_args()
    rng=np.random.default_rng(20261104);tf.random.set_seed(20261104)
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    families=defaultdict(list)
    for i,r in enumerate(rows):
        if r['source']=='esc50':families[r['id'].split('-')[1]].append(i)
        elif r['source']=='fsd50k':families[r['id'].split('/')[-1].replace('.wav','')].append(i)
    excluded={i for ids in families.values() if len({rows[j]['source'] for j in ids})>1 for i in ids}
    probe_ids={'234677','535384','554502','819374'}
    chosen=[]
    for split in ('train','val','test'):
        for label,limit in [('1',800),('0',1800)]:
            pool=[i for i,r in enumerate(rows) if r['split']==split and r['label']==label and i not in excluded
                  and r['id'].split('/')[-1].replace('.wav','') not in probe_ids]
            chosen.extend(rng.choice(pool,min(limit,len(pool)),replace=False).tolist() if split=='train' else pool)
    (ROOT/'selection.json').write_text(json.dumps({'seed':20261104,'clips':chosen,'excluded_shared_rows':len(excluded),
        'reserved_probes':sorted(probe_ids),'limits_train':{'positive':800,'negative':1800},
        'evaluation':'first 30 s per recording, original and fixed simulated channel',
        'limitations':['weak clip labels','previously reused held-out data','simulated channels are not measured microphone responses']},indent=2))
    cache=ROOT/'features.npz'
    if not cache.exists():
        pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e])
        core.load_weights(os.path.join(DATA,'yamnet.h5'))
        mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
        @tf.function(input_signature=[tf.TensorSpec([None,PATCH],tf.float32)])
        def embed(w):
            lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
            s,e=core(lm,training=False)
            return tf.concat([e,s],axis=1)
        beds=[]
        for c in chosen:
            r=rows[c]
            if r['split']=='train' and r['label']=='0' and any(t in r['labels'] for t in ('Wind','Traffic','Speech','Rain')):
                beds.append(read_wav16(os.path.join(DATA,r['path']))[:16000*8])
                if len(beds)>=40:break
        arrays=defaultdict(list)
        for count,c in enumerate(chosen):
            r=rows[c];wave=read_wav16(os.path.join(DATA,r['path']))
            if r['split']!='train':wave=wave[:16000*30]
            if len(wave)<PATCH:wave=np.pad(wave,(0,PATCH-len(wave)))
            n=1+(len(wave)-PATCH)//HOP
            indices=np.arange(n)
            if r['split']=='train' and n>12:indices=np.sort(rng.choice(n,12,replace=False))
            windows=np.stack([wave[i*HOP:i*HOP+PATCH] for i in indices])
            versions=range(3) if r['split']=='train' else range(2)
            for version in versions:
                raw=windows.copy()
                if version:
                    # Two channel families: modest band limiting and stronger low-end loss.
                    lo,hi=(150.,5000.) if version==1 else (450.,3200.)
                    raw=sosfilt(butter(2,[lo,hi],btype='bandpass',fs=16000,output='sos'),raw,axis=1).astype(np.float32)
                    # Short delayed reflections approximate a room, without claiming
                    # to reproduce this user's room or microphone response.
                    if r['split']=='train':
                        delay=int(rng.integers(320,1600));raw[:,delay:]+=.2*raw[:,:-delay]
                        if beds and rng.random()<.5:
                            bed=beds[int(rng.integers(len(beds)))];bed=np.resize(bed,PATCH)
                            scale=np.sqrt(np.mean(raw**2,axis=1,keepdims=True)/(np.mean(bed**2)+1e-10))*10**(-rng.uniform(5,20)/20)
                            raw+=bed[None,:]*scale
                        raw*=10**(-rng.uniform(8,28)/20)
                    else:raw*=10**(-24/20)
                w,levels=condition(raw)
                x=np.concatenate([embed(w[i:i+32]).numpy() for i in range(0,len(w),32)])
                arrays['x'].append(x.astype(np.float16))
                for key,value,dtype in [('clip',c,np.int32),('label',int(r['label']),np.int8),
                    ('split',{'train':0,'val':1,'test':2}[r['split']],np.int8),('version',version,np.int8)]:
                    arrays[key].append(np.full(len(x),value,dtype=dtype))
                arrays['level'].append(levels.astype(np.float16))
            if count%100==0:print('features',count+1,'/',len(chosen),flush=True)
        np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in arrays.items()})
        del core,arrays
    d=dict(np.load(cache));x=d['x'];y=d['label'].astype(np.float32);tr=d['split']==0
    # Training only: emphasize train-positive windows without a horn or bell.
    s=x[:,1024:];horn=s[:,[324,325]].max(1);bell=s[:,[173,195,196,197,198,199,349]].max(1)
    rail=s[:,[322,323,326,327,328]].max(1);speech=s[:,:6].max(1)
    weight=np.ones(len(x),np.float32)
    weight[(y==1)&(horn<.1)&(bell<.1)]*=2
    weight[(y==0)&(s[:,337]>.1)]*=2
    weight[(y==0)&(bell>.1)]*=2
    weight[(y==1)&(speech>=.5)&(rail<.1)&(horn<.1)]=0
    weight[d['level'] < -70]=0
    # Equalize contribution by recording/augmentation before sound emphasis.
    for c in np.unique(d['clip'][tr]):
        for v in (0,1,2):
            m=tr&(d['clip']==c)&(d['version']==v)
            if m.any():weight[m]/=m.sum()
    weight[tr]/=weight[tr].mean()
    # Reset fit randomness so reusing the feature cache does not change the run.
    tf.random.set_seed(20261104)
    train_rng=np.random.default_rng(20261105)
    inp=layers.Input((1545,));z=layers.Dropout(.25)(inp)
    z=layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-4))(z)
    z=layers.Dropout(.3)(z);out=layers.Dense(1,activation='sigmoid',name='head_out')(z)
    candidate=(models.load_model(FOUNDATION/'baseline'/'head.h5',compile=False)
               if args.initialize=='baseline' else Model(inp,out))
    candidate.compile(optimizer=optimizers.Adam(3e-5 if args.initialize=='baseline' else 3e-4),loss='binary_crossentropy')
    class Batches(utils.Sequence):
        def __init__(self,mask,weighted):self.ids=np.flatnonzero(mask);self.weighted=weighted
        def __len__(self):return int(np.ceil(len(self.ids)/512))
        def __getitem__(self,i):
            ids=self.ids[i*512:(i+1)*512]
            return (x[ids],y[ids],weight[ids]) if self.weighted else (x[ids],y[ids])
        def on_epoch_end(self):
            if self.weighted:train_rng.shuffle(self.ids)
    train=Batches(tr,True);train.on_epoch_end()
    candidate.fit(train,validation_data=Batches(d['split']==1,False),epochs=40,verbose=2,workers=1,max_queue_size=2,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True)])
    candidate.save(ROOT/'head.h5')
    old=models.load_model(FOUNDATION/'baseline'/'head.h5',compile=False)
    def predict(m):return np.concatenate([m(x[i:i+2048],training=False).numpy().ravel() for i in range(0,len(x),2048)])
    predictions={'baseline_conditioned':predict(old),'channel_candidate':predict(candidate)}
    report={'initialization':args.initialize,'notes':['head-only causal smoothing, seven-window trigger, raw -65 dBFS gate',
        'validation selects a common threshold across original and simulated channels','held-out data is exploratory, not pristine']}
    for name,p in predictions.items():
        records=[]
        for split in (1,2):
            for c in np.unique(d['clip'][d['split']==split]):
                for v in (0,1):
                    m=(d['split']==split)&(d['clip']==c)&(d['version']==v);scores=p[m].copy();scores[d['level'][m]<-65]=0
                    records.append({'clip':int(c),'split':split,'version':v,'label':int(rows[c]['label']),'score':f.event_score(scores)})
        va=[r for r in records if r['split']==1];labels=np.array([r['label']==1 for r in va]);scores=np.array([r['score'] for r in va])
        # Treat any false trigger across the two versions as a false-positive recording.
        negatives=sorted({r['clip'] for r in va if not r['label']})
        ns=np.array([max(r['score'] for r in va if r['clip']==c) for c in negatives])
        grid=np.r_[np.linspace(.01,.99,197),1.00001]
        eligible=[t for t in grid if np.mean(ns>=t)<=.005]
        threshold=max(eligible,key=lambda t:(f.metrics(labels,scores,t)['recall'],-float(t)))
        result={'threshold':float(threshold),'validation':{},'test':{}}
        for split,label in [(1,'validation'),(2,'test')]:
            for v in (0,1):
                rr=[r for r in records if r['split']==split and r['version']==v]
                result[label][str(v)]=f.metrics(np.array([r['label']==1 for r in rr]),np.array([r['score'] for r in rr]),threshold)
        report[name]=result
        (ROOT/(name+'-records.json')).write_text(json.dumps(records))
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
