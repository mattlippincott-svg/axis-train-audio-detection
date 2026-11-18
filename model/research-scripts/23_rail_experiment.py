"""Offline clatter expert and supporting-only squeal expert. Never deploys.

New anchor labels come from uploader descriptions, with acoustic exclusions.
Evaluation sources never enter the training arrays. Existing backbone/head are
frozen; validation has historical reuse and is explicitly development evidence.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,sys,pickle,argparse
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from sklearn.linear_model import LogisticRegression
from common import DATA,OUT,MANIFEST,read_wav16
from rail_fusion import rail_scores,sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path(OUT).parent/'experiments'/'rail-sounds'
CHANNEL=ROOT.parent/'channel-robust'
EXCLUDED={'535384':'2020-08-03 freight train: user identified overlapping car engine'}
TRAIN={769890:'clatter',199536:'clatter',700035:'squeal'}
TEST={181868:'squeal'}

def grouped(d,mask):
    clips=d['clip'][mask];versions=d['version'][mask]
    return [(int(c),int(v),np.flatnonzero((clips==c)&(versions==v)))
            for c in np.unique(clips) for v in np.unique(versions[clips==c])]

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--rolling-context',action='store_true');parser.add_argument('--hard-negatives',action='store_true');args=parser.parse_args()
    cache_root=ROOT
    if args.rolling_context:ROOT=ROOT.parent/'rail-context';ROOT.mkdir(exist_ok=True)
    if args.hard_negatives:ROOT=cache_root.parent/'rail-hard-negatives';ROOT.mkdir(exist_ok=True)
    import tensorflow as tf
    from tf_keras import Model,layers,models
    f=import_module('06_foundation');channel=import_module('13_channel_robust')
    rng=np.random.default_rng(20261103)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    metadata={}
    for part in ('dev','eval'):metadata.update(json.load(open(Path(DATA)/'meta'/f'{part}_clips_info_FSD50K.json')))
    source_id=lambda r:r['id'].split('/')[-1].replace('.wav','')
    # Keep whole uploader families away from negative training when using one of
    # their recordings as an anchor or evaluation example.
    reserved_authors={'WeenyBeany','mlekoman','JotrainG','cognito perceptu','TheFlyFishingFilmmaker','eksaa','tom_woysky','BWSmithATL','tim.kahn'}
    excluded_rows={i for i,r in enumerate(rows) if source_id(r) in EXCLUDED or source_id(r) in {str(s) for s in (*TRAIN,*TEST)}
                   or metadata.get(source_id(r),{}).get('uploader') in reserved_authors}
    d=dict(np.load(CHANNEL/'features.npz'))
    assert not any(source_id(rows[int(c)]) in EXCLUDED for c in np.unique(d['clip']))
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(os.path.join(DATA,'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    external=cache_root/'external-features.npz'
    if not external.exists():
        arrays={k:[] for k in ('x','level','source','version','time')}
        for sid in (*TRAIN,*TEST):
            wave=read_wav16(cache_root/f'{sid}.wav')[:16000*120]
            times=np.arange(1+(len(wave)-15600)//7680)
            w=np.stack([wave[i*7680:i*7680+15600] for i in times])
            for v in range(8 if sid in TRAIN else 2):
                raw=w.copy()
                if v:
                    raw=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),raw,axis=1).astype(np.float32)
                    if v>1:raw=lfilter([1,-rng.uniform(.3,.8)],[1],raw,axis=1).astype(np.float32)
                    raw*=10**(-(24 if v==1 else rng.uniform(8,28))/20)
                norm,level=channel.condition(raw)
                x=np.concatenate([embed(norm[i:i+32]).numpy() for i in range(0,len(norm),32)])
                for k,value in dict(x=x.astype(np.float16),level=level,source=np.full(len(x),sid),version=np.full(len(x),v),time=times*.48).items():arrays[k].append(value)
        np.savez_compressed(external,**{k:np.concatenate(v) for k,v in arrays.items()})
    ext=dict(np.load(external));x=d['x'];tags=x[:,1024:]
    neg=(d['split']==0)&(d['label']==0)&~np.isin(d['clip'],list(excluded_rows))&(d['level']>=-70)
    # Equal recording weights, with extra weight for likely confusers.
    nw=np.ones(neg.sum(),np.float64);nc=d['clip'][neg]
    for c in np.unique(nc):nw[nc==c]/=(nc==c).sum()
    hard=tags[neg][:,[337,307,308,195,196,197,389,392]].max(1)>.1
    nw[hard]*=2;nw/=nw.sum()
    extra_neg=None;extra_weight=None
    if args.hard_negatives:
        # TRAIN-ONLY confuser emphasis; no test waveform enters fitting.
        confusers=('Skateboard','Engine','Truck','Bus','Wind','Waves','Water','Mechanism','Siren','Machinery','Bicycle')
        for c in np.unique(nc):
            if any(t.lower() in rows[int(c)]['labels'].lower() for t in confusers):nw[nc==c]*=4
        nw/=nw.sum()
        # Existing engine anchors become explicit non-clatter/non-squeal examples.
        # These are the earlier training sources, not the held-out engine panel.
        engine=dict(np.load(cache_root.parent/'low-band'/'expert-data.npz'))
        em=engine['y']==1;extra_neg=engine['x'][em,:1545]
        extra_weight=engine['weight'][em].astype(np.float64);extra_weight/=extra_weight.sum()
    experts={};training_report={}
    et=ext['x'][:,1024:];clean=(et[:,[324,325,195,196,197]].max(1)<.1)&(et[:,:6].max(1)<.2)&(ext['level']>=-65)
    for target in ('clatter','squeal'):
        ids=[sid for sid,t in TRAIN.items() if t==target]
        pos=np.isin(ext['source'],ids)&clean
        if target=='clatter':pos &= et[:,327]<.2
        pw=np.zeros(pos.sum());pc=ext['source'][pos]
        for sid in ids:
            if not (pc==sid).any():raise RuntimeError(f'No usable windows for {sid}')
            pw[pc==sid]=1/(len(ids)*(pc==sid).sum())
        # Weights sum to number of rows so C is comparable across fits.
        nx=x[neg];negative_weight=nw
        if extra_neg is not None:
            nx=np.concatenate([nx,extra_neg]);negative_weight=np.r_[nw*.75,extra_weight*.25]
        tx=np.concatenate([nx,ext['x'][pos]]).astype(np.float32)
        ty=np.r_[np.zeros(len(nx)),np.ones(pos.sum())]
        weights=np.r_[negative_weight,pw]*len(ty)/2
        expert=LogisticRegression(C=.03,max_iter=500,solver='lbfgs',random_state=20261103)
        expert.fit(tx,ty,sample_weight=weights);experts[target]=expert
        training_report[target]={'positive_sources':ids,'positive_windows':int(pos.sum()),'negative_recordings':len(np.unique(nc)),
                                 'engine_negative_windows':0 if extra_neg is None else len(extra_neg),
                                 'training_positive_probability_median':float(np.median(expert.predict_proba(tx[len(nx):])[:,1]))}
    with open(ROOT/'experts.pkl','wb') as h:pickle.dump(experts,h)
    base=models.load_model(CHANNEL/'head.h5',compile=False)
    def base_predict(z):return np.concatenate([base(z[i:i+2048],training=False).numpy().ravel() for i in range(0,len(z),2048)])
    def outputs(z):return base_predict(z),experts['clatter'].decision_function(z),experts['squeal'].decision_function(z)
    def combine(z,values,bias):
        return rail_scores(*values,z[:,1024:],bias,args.rolling_context)[0]
    vm=d['split']==1;vg=grouped(d,vm);vx=x[vm];vl=d['level'][vm];vv=outputs(vx)
    baseline_val=vv[0].copy();baseline_val[vl < -65]=0
    baseline_false_versions={(c,v) for c,v,ids in vg if rows[c]['label']=='0' and f.event_score(baseline_val[ids])>=.869}
    negatives={c for c,v,ids in vg if rows[c]['label']=='0'}
    def records(mask,values,bias):
        z=x[mask];p=combine(z,values,bias);p[d['level'][mask]<-65]=0
        return [dict(clip=c,version=v,label=int(rows[c]['label']),score=f.event_score(p[ids])) for c,v,ids in grouped(d,mask)]
    lo,hi=-30.,30.
    for _ in range(25):
        mid=(lo+hi)/2;rr=records(vm,vv,mid)
        fp={r['clip'] for r in rr if r['label']==0 and r['score']>=.869}
        false_versions={(r['clip'],r['version']) for r in rr if r['label']==0 and r['score']>=.869}
        allowed=(false_versions<=baseline_false_versions) if args.hard_negatives else len(fp)/len(negatives)<=.005
        if allowed:lo=mid
        else:hi=mid
    bias=lo-(1. if args.hard_negatives else .01)
    report={'training':training_report,'clatter_logit_bias':bias,'threshold':.869,'rolling_context_required':args.rolling_context,'hard_negatives':args.hard_negatives,
            'calibration_rule':'no new false validation versions, minus one logit margin' if args.hard_negatives else 'joint false recording rate <= 0.5%',
            'exclusions':EXCLUDED,'reserved_authors':sorted(reserved_authors),'policy':'clatter independent; squeal additive only with clatter, bell or horn',
            'limits':['uploader-level weak positive labels','not human verified intervals','two clatter training recordings, one squeal recording',
                      'frozen pretrained backbone may have seen related source material','historically reused validation/test','not a release candidate']}
    for name,mask in [('validation',vm),('test',d['split']==2)]:
        val=vv if name=='validation' else outputs(x[mask]);rr=records(mask,val,bias)
        (ROOT/f'{name}-records.json').write_text(json.dumps(rr,indent=2))
        report[name]=f.metrics(np.array([r['label'] for r in rr]),np.array([r['score'] for r in rr]),.869)
        old=val[0].copy();old[d['level'][mask]<-65]=0
        baseline_records=[dict(clip=c,version=v,label=int(rows[c]['label']),score=f.event_score(old[ids])) for c,v,ids in grouped(d,mask)]
        (ROOT/f'{name}-baseline-records.json').write_text(json.dumps(baseline_records,indent=2))
        report[name+'_baseline']=f.metrics(np.array([r['label'] for r in baseline_records]),np.array([r['score'] for r in baseline_records]),.869)
        report[name+'_targeted_records']=[dict(r,source=source_id(rows[r['clip']])) for r in rr
            if source_id(rows[r['clip']]) in {'316616','232178','382586','420655','87376','95067'}]
    ep=outputs(ext['x']);pred=combine(ext['x'],ep,bias);pred[ext['level']<-65]=0
    probes=[]
    for sid in (*TRAIN,*TEST):
        for v in (0,1):
            m=(ext['source']==sid)&(ext['version']==v)
            cp=sigmoid(ep[1][m]+bias);cp[ext['level'][m]<-65]=0
            if args.rolling_context:cp*=np.clip(ext['x'][m,1024+326]/.2,0,1)
            probes.append(dict(source=sid,version=v,role='training anchor' if sid in TRAIN else 'held-out diagnostic',
                base_score=f.event_score(np.where(ext['level'][m]>=-65,ep[0][m],0)),candidate_score=f.event_score(pred[m]),
                clatter_only_score=f.event_score(cp),squeal_probability_peak=float(sigmoid(ep[2][m]+bias).max())))
    report['probes']=probes
    (ROOT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
