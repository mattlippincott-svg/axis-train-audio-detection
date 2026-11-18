"""Supplement frozen 0.6.1 with a calibrated low-band locomotive expert.

No source-reference waveform is used in gradient training. The user's known
engine failure is development validation; separate engine authors remain held out.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import csv,json,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from common import DATA,OUT,MANIFEST,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path(OUT).parent/'experiments'/'low-band'
CHANNEL=Path(OUT).parent/'experiments'/'channel-robust'
ANCHORS={135700:'EMD_load',417553:'Retiro',17854:'Austin',17855:'Austin',324109:'CSX_room'}


def low_band_tf(w):
    import tensorflow as tf
    # 64-tap Hann low-pass, decimate to 1 kHz; retain the same fixed-window ABI.
    taps=np.hanning(64).astype(np.float32);taps/=taps.sum()
    low=tf.nn.conv1d(w[:,:,None],tf.constant(taps[:,None,None]),stride=16,padding='SAME')[:,:,0]
    win=np.hanning(975).astype(np.float32)
    frequencies=np.arange(20,251,dtype=np.float32)
    phase=2*np.pi*np.arange(975,dtype=np.float32)[:,None]*frequencies[None,:]/1000
    cosine=(np.cos(phase)*win[:,None]*2/win.sum()).astype(np.float32)
    sine=(np.sin(phase)*win[:,None]*2/win.sum()).astype(np.float32)
    real=tf.matmul(low,tf.constant(cosine));imag=tf.matmul(low,tf.constant(sine))
    power=(real*real+imag*imag)/(tf.reduce_mean(w*w,axis=1,keepdims=True)+1e-12)
    edges=[20,30,40,50,65,80,100,125,150,180,220,251]
    bands=np.zeros((231,len(edges)-1),np.float32)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        m=(frequencies>=lo)&(frequencies<hi);bands[m,i]=1/m.sum()
    return tf.clip_by_value(tf.math.log(tf.matmul(power,tf.constant(bands))+1e-8),-16.,0.)/8


def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models,optimizers,losses,callbacks,utils,regularizers
    f=import_module('06_foundation');channel=import_module('13_channel_robust');ROOT.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261110);tf.random.set_seed(20261110)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')));d=dict(np.load(CHANNEL/'features.npz'))
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(os.path.join(DATA,'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def extract(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s,low_band_tf(w)],axis=1)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def low(w):return low_band_tf(w)
    cache=ROOT/'expert-data.npz'
    if not cache.exists():
        selection=json.loads((CHANNEL/'selection.json').read_text())['clips']
        negative=[c for c in selection if rows[c]['split']=='train' and rows[c]['label']=='0']
        hard=[c for c in negative if any(t in rows[c]['labels'].split(',') for t in ('Engine','Truck','Bus','Bell','church_bells','Idling'))]
        extra=[c for c in negative if c not in set(hard)];negative=hard+rng.choice(extra,min(400,len(extra)),replace=False).tolist()
        xx=[];yy=[];weights=[]
        for count,c in enumerate(negative):
            wave=read_wav16(os.path.join(DATA,rows[c]['path']))
            if len(wave)<15600:wave=np.pad(wave,(0,15600-len(wave)))
            n=1+(len(wave)-15600)//7680;ids=np.sort(rng.choice(n,min(6,n),replace=False));w=np.stack([wave[i*7680:i*7680+15600] for i in ids])
            for raw in (w,sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w,axis=1).astype(np.float32)*.0630957):
                norm,level=channel.condition(raw);feat=extract(norm).numpy();keep=level>=-75
                xx.append(feat[keep]);yy.extend([0]*int(keep.sum()));weights.extend([1/max(1,int(keep.sum()))]*int(keep.sum()))
            if count%100==0:print('expert negatives',count+1,'/',len(negative),flush=True)
        negative_mass=sum(weights)
        panel=Path(OUT).parent/'experiments'/'upper-finetune'/'engine-panel'
        for sid,family in ANCHORS.items():
            wave=read_wav16(panel/(str(sid)+'.wav'))[:16000*60];n=1+(len(wave)-15600)//7680
            ids=np.sort(rng.choice(n,min(64,n),replace=False));w=np.stack([wave[i*7680:i*7680+15600] for i in ids]);features=[]
            for version in range(8):
                raw=w.copy()
                if version:
                    raw=lfilter([1.,-rng.uniform(.4,.85)],[1.],raw,axis=1).astype(np.float32)*10**(-rng.uniform(0,28)/20)
                norm,level=channel.condition(raw);feat=extract(norm).numpy();features.append(feat[level>=-75])
            feat=np.concatenate(features);xx.append(feat);yy.extend([1]*len(feat))
            siblings=sum(v==family for v in ANCHORS.values());weights.extend([negative_mass/4/siblings/max(1,len(feat))]*len(feat))
        np.savez_compressed(cache,x=np.concatenate(xx).astype(np.float16),y=np.array(yy,np.float32),weight=np.array(weights,np.float32))
    training=dict(np.load(cache));training['weight']/=training['weight'].mean()
    valcache=ROOT/'validation-data.npz'
    if not valcache.exists():
        vm=d['split']==1;vx=d['x'][vm];vc=d['clip'][vm];vv=d['version'][vm];vl=d['level'][vm];lf=np.zeros((len(vx),11),np.float32)
        for index,c in enumerate(np.unique(vc)):
            wave=read_wav16(os.path.join(DATA,rows[c]['path']))[:16000*30]
            if len(wave)<15600:wave=np.pad(wave,(0,15600-len(wave)))
            w=np.lib.stride_tricks.sliding_window_view(wave,15600)[::7680].copy()
            for version in (0,1):
                mask=(vc==c)&(vv==version);raw=w if not version else sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w,axis=1).astype(np.float32)*.063095734
                norm,_=channel.condition(raw);features=np.concatenate([low(norm[i:i+64]).numpy() for i in range(0,len(norm),64)])
                if len(features)!=int(mask.sum()):raise RuntimeError('Validation alignment mismatch')
                lf[mask]=features
            if index%100==0:print('expert validation',index+1,flush=True)
        np.savez_compressed(valcache,x=np.concatenate([vx,lf],axis=1).astype(np.float16),clip=vc,version=vv,level=vl)
    val=dict(np.load(valcache));groups=[]
    for c in np.unique(val['clip']):
        for v in (0,1):groups.append((int(c),v,np.flatnonzero((val['clip']==c)&(val['version']==v))))
    negatives=sorted({c for c,v,ids in groups if rows[c]['label']=='0'})
    baseline=models.load_model(CHANNEL/'head.h5',compile=False)
    base=np.concatenate([baseline(val['x'][i:i+2048,:1545],training=False).numpy().ravel() for i in range(0,len(val['x']),2048)])
    lab=Path('/lab/work/audio-diagnostics/ns-061-armed');a=json.loads((lab/'alignment.json').read_text());lo,hi=a['source_first20_capture_interval_s']
    ref=read_wav16(lab/'capture.wav')[round(lo*16000):round(hi*16000)];rw=np.lib.stride_tricks.sliding_window_view(ref,15600)[::7680].copy();rn,rl=channel.condition(rw);rx=extract(rn).numpy();rb=baseline(rx[:,:1545],training=False).numpy().ravel()
    inp=layers.Input((1556,));z=layers.Dense(32,activation='relu',kernel_regularizer=regularizers.l2(.001))(inp);z=layers.Dropout(.2)(z);out=layers.Dense(1,name='engine_logit')(z);expert=Model(inp,out)
    expert.compile(optimizer=optimizers.Adam(3e-4),loss=losses.BinaryCrossentropy(from_logits=True))
    def sigmoid(z):return 1/(1+np.exp(-np.clip(z,-60,60)))
    def summarize(z,bias):
        p=np.maximum(base,sigmoid(z+bias));p[val['level'] < -65]=0
        records=[(c,v,int(rows[c]['label']),f.event_score(p[ids])) for c,v,ids in groups]
        false={c for c,v,label,s in records if label==0 and s>=.869}
        return records,len(false)
    history=[]
    class Select(callbacks.Callback):
        def __init__(self):super().__init__();self.best=None
        def on_epoch_end(self,epoch,logs=None):
            z=np.concatenate([expert(val['x'][i:i+2048],training=False).numpy().ravel() for i in range(0,len(base),2048)])
            lo,hi=-20.,20.
            for _ in range(24):
                mid=(lo+hi)/2
                if summarize(z,mid)[1]/len(negatives)<=.005:lo=mid
                else:hi=mid
            bias=lo-.01;records,nfp=summarize(z,bias);tp=sum(label==1 and s>=.869 for c,v,label,s in records)
            rp=np.maximum(rb,sigmoid(expert(rx,training=False).numpy().ravel()+bias));rp[rl < -65]=0;score=f.event_score(rp)
            report={'epoch':epoch+1,'bias':bias,'validation_positive_versions_detected':tp,'validation_false_recordings':nfp,'reference_score':score,'reference_pass':bool(score>=.869)}
            history.append(report);key=(int(score>=.869),tp,float(score))
            print('low-band validation',report,flush=True)
            if self.best is None or key>self.best:self.best=key;expert.save(ROOT/'expert.h5');(ROOT/'selected.json').write_text(json.dumps(report,indent=2))
    class Batches(utils.Sequence):
        def __init__(self):self.ids=np.arange(len(training['y']))
        def __len__(self):return int(np.ceil(len(self.ids)/256))
        def __getitem__(self,i):
            ids=self.ids[i*256:(i+1)*256];return training['x'][ids],training['y'][ids],training['weight'][ids]
        def on_epoch_end(self):rng.shuffle(self.ids)
    batches=Batches();batches.on_epoch_end();expert.fit(batches,epochs=15,verbose=2,workers=1,max_queue_size=2,callbacks=[Select()])
    (ROOT/'history.json').write_text(json.dumps(history,indent=2))
    (ROOT/'provenance.json').write_text(json.dumps({'anchors':ANCHORS,'threshold':.869,'low_band_hz':[20,250],
        'reference_role':'development validation only','held_out_sources':[146325,773336,546944],
        'limits':['four positive source families','not pristine diagnostics','requires independent negative and engine validation before release']},indent=2))


if __name__=='__main__':main()
