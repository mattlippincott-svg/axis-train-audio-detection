"""Add two explicit real locomotive-engine source families to head training.

Training anchors: Freesound 135700 (EMD load test) and 417553 (train idle).
146325 and 773336 remain unseen-source diagnostics. The user's aligned NS
engine capture is a DEVELOPMENT validation reference, never a training input.
This is deliberately a small-data experiment, not a generalization claim.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,sys,argparse
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import lfilter,butter,sosfilt
from common import DATA,OUT,MANIFEST,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path(OUT).parent/'experiments'/'engine-anchors'
CHANNEL=Path(OUT).parent/'experiments'/'channel-robust'


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--expanded',action='store_true');args=parser.parse_args()
    anchors=(135700,417553,17854,17855,324109) if args.expanded else (135700,417553)
    families={135700:'EMD_load',417553:'Retiro_idle',17854:'Austin_patchen',17855:'Austin_patchen',324109:'CSX_room'}
    if args.expanded:ROOT=Path(OUT).parent/'experiments'/'engine-diverse'
    import tensorflow as tf
    from tf_keras import Model,layers,models,utils,callbacks,optimizers
    f=import_module('06_foundation');channel=import_module('13_channel_robust')
    ROOT.mkdir(exist_ok=True)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    d=dict(np.load(CHANNEL/'features.npz'));x=d['x'];y=d['label'].astype(np.float32);tr=d['split']==0
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(os.path.join(DATA,'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    rng=np.random.default_rng(20261109)
    added=[];source_ids=[]
    panel=Path(OUT).parent/'experiments'/'upper-finetune'/'engine-panel'
    for sid in anchors:
        wave=read_wav16(panel/(str(sid)+'.wav'))[:16000*60]
        n=1+(len(wave)-15600)//7680;ids=np.sort(rng.choice(n,min(64,n),replace=False))
        w=np.stack([wave[i*7680:i*7680+15600] for i in ids])
        for version in range(8):
            raw=w.copy()
            if version:
                raw=lfilter([1.,-rng.uniform(.4,.85)],[1.],raw,axis=1).astype(np.float32)
                raw=sosfilt(butter(2,[100.,rng.uniform(3500,6500)],btype='bandpass',fs=16000,output='sos'),raw,axis=1).astype(np.float32)
                raw*=10**(-rng.uniform(0,28)/20)
            norm,level=channel.condition(raw);feat=np.concatenate([embed(norm[i:i+32]).numpy() for i in range(0,len(w),32)])
            keep=level>=-70
            added.append(feat[keep].astype(np.float16));source_ids.extend([sid]*int(keep.sum()))
    extra=np.concatenate(added);extra_sources=np.array(source_ids)
    # User's failed interval is validation only; acoustic alignment is verified.
    lab=Path('/lab/work/audio-diagnostics/ns-061-armed');alignment=json.loads((lab/'alignment.json').read_text())
    if not alignment['accepted']:raise RuntimeError('Development reference lacks alignment')
    lo,hi=alignment['source_first20_capture_interval_s'];ref=read_wav16(lab/'capture.wav')[round(lo*16000):round(hi*16000)]
    rw=np.lib.stride_tricks.sliding_window_view(ref,15600)[::7680].copy();rn,ref_levels=channel.condition(rw);ref_x=embed(rn).numpy()
    # Keep inherited negative/positive windows; equal recording weighting.
    weight=np.ones(len(x),np.float32);tags=x[:,1024:]
    horn=tags[:,[324,325]].max(1);bell=tags[:,[195,196,197]].max(1);speech=tags[:,:6].max(1);rail=tags[:,[322,323,326,327,328]].max(1)
    weight[(y==1)&(horn<.1)&(bell<.1)]*=2;weight[(y==0)&(tags[:,337]>.1)]*=2;weight[(y==0)&(bell>.1)]*=2
    weight[(y==1)&(speech>.5)&(rail<.1)&(horn<.1)]=0;weight[d['level'] < -70]=0
    for c in np.unique(d['clip'][tr]):
        for v in (0,1,2):
            m=tr&(d['clip']==c)&(d['version']==v)
            if m.any():weight[m]/=m.sum()
    weight[tr]/=weight[tr].mean()
    ew=np.zeros(len(extra),np.float32)
    for family in sorted({families[sid] for sid in anchors}):
        m=np.isin(extra_sources,[sid for sid in anchors if families[sid]==family]);ew[m]=float(weight[tr].sum()*.05/max(1,m.sum()))
    train_x=np.concatenate([x[tr],extra]);train_y=np.r_[y[tr],np.ones(len(extra),np.float32)];train_w=np.r_[weight[tr],ew]
    candidate=models.load_model(CHANNEL/'head.h5',compile=False)
    candidate.compile(optimizer=optimizers.Adam(3e-5),loss='binary_crossentropy')
    tf.random.set_seed(20261109)
    class Batches(utils.Sequence):
        def __init__(self):self.ids=np.arange(len(train_x))
        def __len__(self):return int(np.ceil(len(self.ids)/512))
        def __getitem__(self,i):
            ids=self.ids[i*512:(i+1)*512];return train_x[ids],train_y[ids],train_w[ids]
        def on_epoch_end(self):rng.shuffle(self.ids)
    vm=d['split']==1;vx=x[vm];vc=d['clip'][vm];vv=d['version'][vm];vl=d['level'][vm]
    groups=[]
    for c in np.unique(vc):
        for v in (0,1):
            mask=(vc==c)&(vv==v);groups.append((int(c),v,np.flatnonzero(mask)))
    history=[]
    def evaluate():
        pred=np.concatenate([candidate(vx[i:i+2048],training=False).numpy().ravel() for i in range(0,len(vx),2048)])
        pred[vl < -65]=0
        records=[(c,v,int(rows[c]['label']),f.event_score(pred[ids])) for c,v,ids in groups]
        negatives=sorted({c for c,v,l,s in records if l==0})
        ns=np.array([max(s for cc,v,l,s in records if cc==c) for c in negatives])
        grid=np.r_[np.linspace(.01,.99,197),1.00001];eligible=[t for t in grid if np.mean(ns>=t)<=.005]
        labs=np.array([l==1 for c,v,l,s in records]);scores=np.array([s for c,v,l,s in records])
        threshold=max(eligible,key=lambda t:(f.metrics(labs,scores,t)['recall'],-float(t)))
        rp=candidate(ref_x,training=False).numpy().ravel();rp[ref_levels < -65]=0;ref_score=f.event_score(rp)
        metrics=f.metrics(labs,scores,threshold)
        # The known failure is development validation, explicitly not a pristine test.
        # Require preserving at least 170/324 generic positive-version detections.
        key=(int(ref_score>=threshold and metrics['tp']>=170),metrics['tp'],float(ref_score-threshold))
        return key,dict(threshold=float(threshold),reference_score=ref_score,reference_pass=bool(ref_score>=threshold),validation=metrics)
    class Select(callbacks.Callback):
        def __init__(self):super().__init__();self.best=None
        def on_epoch_end(self,epoch,logs=None):
            key,report=evaluate();report['epoch']=epoch+1;history.append(report);print('engine validation',report,flush=True)
            if self.best is None or key>self.best:self.best=key;candidate.save(ROOT/'head.h5');(ROOT/'selected.json').write_text(json.dumps(report,indent=2))
    train=Batches();train.on_epoch_end()
    candidate.fit(train,epochs=20,verbose=2,workers=1,max_queue_size=2,callbacks=[Select()])
    (ROOT/'history.json').write_text(json.dumps(history,indent=2))
    (ROOT/'provenance.json').write_text(json.dumps({'training_anchor_ids':list(anchors),'held_out_engine_ids':[146325,773336,546944],
        'anchor_family_count':len({families[sid] for sid in anchors}),'extra_augmented_windows':len(extra),'anchor_weight_fraction_relative_to_existing_train':.05*len({families[sid] for sid in anchors}),
        'development_reference':'aligned NS microphone engine interval; validation only',
        'limitations':['Only two new positive source families','engine diagnostics already observed on previous models','weak labels in inherited data','not a deployment validation']},indent=2))


if __name__=='__main__':main()
