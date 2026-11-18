"""Adapt upper acoustic layers on an independent fork; retain original sound tags.

The generic YAMNet tag branch stays frozen. The train branch gets independently
cloned upper layers, so fine-tuning train recognition cannot relabel its sound UI.
All known played probes remain excluded from training.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from common import DATA,OUT,MANIFEST,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path(OUT).parent/'experiments'/'upper-finetune'
CHANNEL=Path(OUT).parent/'experiments'/'channel-robust'
CUT='layer12/pointwise_conv/relu'


def components():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    p=layers.Input((96,64));s,e=yamnet.yamnet(p,params.Params());core=Model(p,[s,e]);core.load_weights(os.path.join(DATA,'yamnet.h5'))
    core.trainable=False
    cut=core.get_layer(CUT).output
    template=Model(cut,e,name='upper_template')
    tail=models.clone_model(template);tail.set_weights(template.get_weights());tail._name='adapted_upper'
    for layer in tail.layers:
        layer.trainable=not isinstance(layer,layers.BatchNormalization)
    extractor=Model(p,[cut,s],name='frozen_acoustic_features')
    head=models.load_model(CHANNEL/'head.h5',compile=False)
    ci=layers.Input(tuple(cut.shape[1:]),name='lower_features');si=layers.Input((521,),name='original_tags')
    emb=tail(ci,training=False)
    prob=head(layers.Concatenate()([emb,si]))
    trainer=Model([ci,si],prob)
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def extract(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        return extractor(lm,training=False)
    return core,tail,head,trainer,extract


def transform(w,seed):
    rng=np.random.default_rng(seed);raw=w.copy()
    coef=float(rng.uniform(.45,.85));raw=lfilter([1.,-coef],[1.],raw,axis=1).astype(np.float32)
    raw=sosfilt(butter(2,[80.,6500.],btype='bandpass',fs=16000,output='sos'),raw,axis=1).astype(np.float32)
    delay=int(rng.integers(160,1200));raw[:,delay:]+=.15*raw[:,:-delay]
    return raw*10**(-rng.uniform(8,24)/20)


def main():
    import tensorflow as tf
    from tf_keras import callbacks,optimizers,utils
    channel=import_module('13_channel_robust');f=import_module('06_foundation')
    ROOT.mkdir(exist_ok=True)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    chosen=json.loads((CHANNEL/'selection.json').read_text())['clips']
    core,tail,head,model,extract=components()
    print('cut shape',model.inputs[0].shape,'trainable parameters',sum(int(np.prod(v.shape)) for v in model.trainable_weights),flush=True)
    cache=ROOT/'fit-features.npz'
    if not cache.exists():
        arrays={k:[] for k in ['lower','tags','y','split','weight']}
        selected=[c for c in chosen if rows[c]['split'] in ('train','val')]
        for index,c in enumerate(selected):
            r=rows[c];wave=read_wav16(os.path.join(DATA,r['path']))
            if len(wave)<15600:wave=np.pad(wave,(0,15600-len(wave)))
            n=1+(len(wave)-15600)//7680
            rng=np.random.default_rng(20261107+c)
            ids=np.sort(rng.choice(n,min(4,n),replace=False))
            w=np.stack([wave[i*7680:i*7680+15600] for i in ids])
            for version,raw in enumerate([w,transform(w,20261107+c)]):
                norm,levels=channel.condition(raw);lower,tags=extract(norm);tags=tags.numpy();lower=lower.numpy()
                weights=np.full(len(w),1/len(w),np.float32)
                if r['label']=='1':
                    horn=tags[:,[324,325]].max(1);bell=tags[:,[173,195,196,197,198,199,349]].max(1)
                    rail=tags[:,[322,323,326,327,328]].max(1);speech=tags[:,:6].max(1)
                    weights[(horn<.1)&(bell<.1)]*=2
                    weights[(speech>.5)&(rail<.1)&(horn<.1)]=0
                else:
                    weights[tags[:,337]>.1]*=2
                    weights[tags[:,[195,196,197]].max(1)>.1]*=2
                weights[levels < -70]=0
                arrays['lower'].append(lower.astype(np.float16));arrays['tags'].append(tags.astype(np.float16))
                arrays['y'].append(np.full(len(w),int(r['label']),np.float32));arrays['split'].append(np.full(len(w),r['split']=='val',np.int8));arrays['weight'].append(weights)
            if index%100==0:print('upper cache',index+1,'/',len(selected),flush=True)
        np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in arrays.items()})
        del arrays
    d=dict(np.load(cache));tr=d['split']==0;d['weight'][tr]/=max(1e-9,d['weight'][tr].mean())
    # Reference tag weights must not change as the independent train fork adapts.
    reference=[v.numpy().copy() for v in core.weights]
    tf.random.set_seed(20261107)
    rng=np.random.default_rng(20261108)
    class Batches(utils.Sequence):
        def __init__(self,mask,weighted):self.ids=np.flatnonzero(mask);self.weighted=weighted
        def __len__(self):return int(np.ceil(len(self.ids)/64))
        def __getitem__(self,i):
            ids=self.ids[i*64:(i+1)*64];inputs=[d['lower'][ids].astype(np.float32),d['tags'][ids].astype(np.float32)]
            return (inputs,d['y'][ids],d['weight'][ids]) if self.weighted else (inputs,d['y'][ids])
        def on_epoch_end(self):
            if self.weighted:rng.shuffle(self.ids)
    model.compile(optimizer=optimizers.Adam(1e-5),loss='binary_crossentropy')
    train=Batches(tr,True);train.on_epoch_end()
    model.fit(train,validation_data=Batches(~tr,False),epochs=8,verbose=2,workers=1,max_queue_size=2,
              callbacks=[callbacks.EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)])
    assert all(np.array_equal(before,after.numpy()) for before,after in zip(reference,core.weights)), 'Frozen tag backbone changed'
    tail.save(ROOT/'adapted-upper.h5');head.save(ROOT/'head.h5')
    (ROOT/'training-info.json').write_text(json.dumps({'initial_head':'channel-robust 0.6.1','cut_layer':CUT,
        'frozen_tag_weights_unchanged':True,'fit_windows':len(d['y']),'training_windows':int(tr.sum()),
        'limitations':['weak clip labels','reused held-out data','synthetic channels','reference probes excluded from training']},indent=2))
    del d
    records=[]
    for index,c in enumerate([c for c in chosen if rows[c]['split']!='train']):
        r=rows[c];wave=read_wav16(os.path.join(DATA,r['path']))[:16000*30]
        if len(wave)<15600:wave=np.pad(wave,(0,15600-len(wave)))
        w=np.lib.stride_tricks.sliding_window_view(wave,15600)[::7680].copy()
        for version in (0,1):
            raw=w if version==0 else sosfilt(butter(2,[150.,5000.],btype='bandpass',fs=16000,output='sos'),w,axis=1).astype(np.float32)*10**(-24/20)
            norm,levels=channel.condition(raw);pred=[]
            for start in range(0,len(w),32):
                lower,tags=extract(norm[start:start+32]);pred.extend(model([lower,tags],training=False).numpy().ravel())
            p=np.array(pred);p[levels < -65]=0
            records.append({'clip':c,'split':r['split'],'version':version,'label':int(r['label']),'score':f.event_score(p)})
        if index%100==0:print('upper evaluation',index+1,flush=True)
    (ROOT/'records.json').write_text(json.dumps(records))
    va=[r for r in records if r['split']=='val'];neg=sorted({r['clip'] for r in va if r['label']==0})
    ns=np.array([max(r['score'] for r in va if r['clip']==c) for c in neg]);grid=np.r_[np.linspace(.01,.99,197),1.00001]
    y=np.array([r['label']==1 for r in va]);s=np.array([r['score'] for r in va])
    eligible=[t for t in grid if np.mean(ns>=t)<=.005]
    threshold=max(eligible,key=lambda t:(f.metrics(y,s,t)['recall'],-float(t)))
    report={'threshold':float(threshold)}
    for split in ('val','test'):
        report[split]={}
        for version in (0,1):
            rr=[r for r in records if r['split']==split and r['version']==version]
            report[split][str(version)]=f.metrics(np.array([r['label']==1 for r in rr]),np.array([r['score'] for r in rr]),threshold)
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
