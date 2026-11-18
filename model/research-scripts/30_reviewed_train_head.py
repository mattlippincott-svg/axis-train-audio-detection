"""Fine-tune train presence with five human-reviewed TRAIN intervals.

Includes rolling-only positives without labeling them as click-clacks. Selects
checkpoint and supported app sensitivity only on legacy validation. Never
scores reserved corpus recordings or deploys. No squeal shortcut is added.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,pickle,sys,argparse,hashlib
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,OUT,MANIFEST,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path(OUT).parent/'experiments'/'reviewed-train-head'
CHANNEL=ROOT.parent/'channel-robust';CORPUS=ROOT.parent/'rail-corpus-v2'

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--confusers',action='store_true');parser.add_argument('--us-reviewed',action='store_true');parser.add_argument('--paired-capture',action='store_true');parser.add_argument('--measured-channel',action='store_true');args=parser.parse_args()
    if args.confusers and args.us_reviewed:parser.error('Run independent experiments separately')
    if args.paired_capture and not args.us_reviewed:parser.error('Paired experiment requires --us-reviewed')
    if args.measured_channel and not args.paired_capture:parser.error('Measured-channel experiment requires --paired-capture')
    if args.confusers:ROOT=ROOT.parent/'reviewed-train-confusers'
    if args.us_reviewed:ROOT=ROOT.parent/'reviewed-train-us'
    if args.paired_capture:ROOT=ROOT.parent/'reviewed-train-paired'
    if args.measured_channel:ROOT=ROOT.parent/'reviewed-train-measured'
    import tensorflow as tf
    from tf_keras import Model,layers,models,optimizers,callbacks,utils
    f=import_module('06_foundation');channel=import_module('13_channel_robust')
    ROOT.mkdir(exist_ok=True);rng=np.random.default_rng(20261107);tf.random.set_seed(20261107)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    catalog=json.loads((CORPUS/'catalog-audited.json').read_text());labels=json.loads((CORPUS/'human-interval-labels.json').read_text())['labels']
    reviewed=dict(np.load(ROOT.parent/'reviewed-clatter'/'reviewed-features.npz'))
    labels_hash=hashlib.sha256((CORPUS/'human-interval-labels.json').read_bytes()).hexdigest()
    assert labels_hash==json.loads((ROOT.parent/'reviewed-clatter'/'cache-info.json').read_text())['label_sha256']
    assert set(reviewed['source'])=={r['source_id'] for r in labels}
    assert not set(reviewed['source']).intersection(r['id'] for r in catalog['sources'] if r['role']=='reserved_review')
    if args.us_reviewed:
        us=dict(np.load(ROOT.parent/'reviewed-us-data'/'features.npz'))
        us_labels=json.loads((ROOT.parent/'us-freight-candidates'/'commons'/'human-interval-labels.json').read_text())['labels']
        us_approved=[r for r in us_labels if r['source_id']=='commons-natchez-2011']
        assert set(us['source'])=={'commons-natchez-2011'} and len(us_approved)==1
        us_hash=hashlib.sha256((ROOT.parent/'us-freight-candidates'/'commons'/'human-interval-labels.json').read_bytes()).hexdigest()
        assert json.loads((ROOT.parent/'reviewed-us-data'/'provenance.json').read_text())['label_sha256']==us_hash
        for k in reviewed:
            reviewed[k]=np.concatenate([reviewed[k].astype(str) if k=='source' else reviewed[k],us[k]])
        labels=labels+us_approved
    training_exclusions=set(catalog['future_training_excluded_legacy_rows'])
    if args.measured_channel:
        training_exclusions.update(json.loads((ROOT.parent/'us-freight-candidates'/'acquired'/'catalog.json').read_text())['future_training_excluded_legacy_rows'])
    d=dict(np.load(CHANNEL/'features.npz'));x=d['x'];tr=(d['split']==0)&~np.isin(d['clip'],list(training_exclusions))
    weight=np.ones(tr.sum(),np.float32);clips=d['clip'][tr];versions=d['version'][tr];y=d['label'][tr].astype(np.float32)
    tags=x[tr,1024:];horn=tags[:,[324,325]].max(1);bell=tags[:,[195,196,197]].max(1)
    weight[(y==1)&(horn<.1)&(bell<.1)]*=2
    weight[(y==0)&(tags[:,[195,196,197,337]].max(1)>.1)]*=2
    if args.confusers:
        for c in np.unique(clips[y==0]):
            if any(t.lower() in rows[int(c)]['labels'].lower() for t in ('Siren','Skateboard','Mechanical_fan','Mechanisms','Truck','Bus','Traffic','Wind','Water')):
                weight[clips==c]*=3
    weight[d['level'][tr]<-70]=0
    for c in np.unique(clips):
        for v in (0,1,2):
            m=(clips==c)&(versions==v)
            if m.any():weight[m]/=m.sum()
    weight/=weight.mean()
    rx=reviewed['x'];rw=np.zeros(len(rx),np.float32)
    for sid in np.unique(reviewed['source']):
        m=(reviewed['source']==sid)&(reviewed['level']>=-70)
        rw[m]=weight.sum()*(.2/len(np.unique(reviewed['source'])))/max(1,m.sum())
    supplemental_x=rx;supplemental_y=np.ones(len(rx),np.float32);supplemental_w=rw
    paired=None
    if args.paired_capture:
        paired=dict(np.load(ROOT.parent/'paired-training-01'/'features.npz'));pw=np.zeros(len(paired['x']),np.float32)
        positive_families=np.unique(reviewed['source'])
        for sid in np.unique(paired['source']):
            m=paired['source']==sid
            if np.all(paired['label'][m]==1):
                assert sid in positive_families
                rw[reviewed['source']==sid]*=.5
                pw[m]=weight.sum()*(.2/len(positive_families))*.5/m.sum()
            else:
                assert np.all(paired['label'][m]==0)
                pw[m]=weight.sum()*.025/m.sum()
        supplemental_x=np.concatenate([rx,paired['x']]);supplemental_y=np.r_[np.ones(len(rx),np.float32),paired['label']]
        supplemental_w=np.r_[rw,pw]
    tx=np.concatenate([x[tr],supplemental_x]);ty=np.r_[y,supplemental_y];tw=np.r_[weight,supplemental_w]
    if args.measured_channel:
        measured=dict(np.load(ROOT.parent/'measured-capture-path'/'features.npz'));mw=np.zeros(len(measured['x']),np.float32)
        assert all(rows[int(c)]['split']=='train' for c in np.unique(measured['clip']))
        for label in (0,1):
            clips_m=np.unique(measured['clip'][measured['label']==label])
            for c in clips_m:
                m=(measured['clip']==c)&(measured['label']==label)
                mw[m]=weight.sum()*.05/max(1,len(clips_m))/m.sum()
        tx=np.concatenate([tx,measured['x']]);ty=np.r_[ty,measured['label']];tw=np.r_[tw,mw]
    candidate=models.load_model(CHANNEL/'head.h5',compile=False)
    candidate.compile(optimizer=optimizers.Adam(3e-5),loss='binary_crossentropy')
    base=models.load_model(CHANNEL/'head.h5',compile=False)
    def predict(model,z):return np.concatenate([model(z[i:i+2048],training=False).numpy().ravel() for i in range(0,len(z),2048)])
    vm=(d['split']==1)&~np.isin(d['clip'],catalog['future_evaluation_excluded_legacy_rows']);vx=x[vm];vl=d['level'][vm]
    def groups(mask):
        cs=d['clip'][mask];vs=d['version'][mask]
        return [(int(c),v,int(rows[int(c)]['label']),np.flatnonzero((cs==c)&(vs==v))) for c in np.unique(cs) for v in (0,1)]
    vg=groups(vm)
    def records(pred,level,gg,gate=-65):
        pred=pred.copy();pred[level < gate]=0
        return [dict(clip=c,version=v,label=l,score=f.event_score(pred[ids])) for c,v,l,ids in gg]
    baseline_val=records(predict(base,vx),vl,vg)
    fp_budget=sum(r['label']==0 and r['score']>=.869 for r in baseline_val)
    grid=[(s,float(.95-.009*(s-1))) for s in range(1,101)]
    gates=(-65,-70,-75) if args.paired_capture else (-65,)
    def operating_point(pred):
        choices=[]
        for gate in gates:
            rr=records(pred,vl,vg,gate);labs=np.array([bool(r['label']) for r in rr]);sc=np.array([r['score'] for r in rr])
            choices.extend((s,t,f.metrics(labs,sc,t),gate) for s,t in grid if np.sum((~labs)&(sc>=t))<=fp_budget)
        return max(choices,key=lambda v:(v[2]['tp'],-v[2]['fp'],v[3],v[1])) if choices else None
    baseline_recalibrated=operating_point(predict(base,vx))
    history=[]
    class Select(callbacks.Callback):
        def __init__(self):super().__init__();self.best=None
        def on_epoch_end(self,epoch,logs=None):
            choice=operating_point(predict(candidate,vx))
            if choice is None:history.append({'epoch':epoch+1,'eligible':False});return
            s,t,metrics,gate=choice
            report=dict(epoch=epoch+1,sensitivity=s,threshold=t,min_level_dbfs=gate,validation=metrics)
            history.append(report);print('reviewed-head validation',report,flush=True)
            key=(metrics['tp'],-metrics['fp'],gate,t)
            if self.best is None or key>self.best:
                self.best=key;candidate.save(ROOT/'head.h5');(ROOT/'selected.json').write_text(json.dumps(report,indent=2))
    class Batches(utils.Sequence):
        def __init__(self):self.ids=np.arange(len(tx))
        def __len__(self):return int(np.ceil(len(tx)/512))
        def __getitem__(self,i):
            ids=self.ids[i*512:(i+1)*512];return tx[ids],ty[ids],tw[ids]
        def on_epoch_end(self):rng.shuffle(self.ids)
    batches=Batches();batches.on_epoch_end();candidate.fit(batches,epochs=12,verbose=2,workers=1,max_queue_size=2,callbacks=[Select()])
    (ROOT/'history.json').write_text(json.dumps(history,indent=2))
    selected=json.loads((ROOT/'selected.json').read_text());threshold=selected['threshold'];gate=selected['min_level_dbfs'];candidate=models.load_model(ROOT/'head.h5',compile=False)
    report={'selected':selected,'confuser_emphasis':args.confusers,'us_reviewed_added':args.us_reviewed,'label_sha256':labels_hash,
        'reviewed_train_positive_sources':[int(s) if str(s).isdigit() else str(s) for s in np.unique(reviewed['source'])],
        'additional_us_label_sha256':us_hash if args.us_reviewed else None,
        'gate_dominant_source_excluded':'commons-bnsf-snow-2016' if args.us_reviewed else None,
        'reviewed_weight_relative_to_existing_training':.2,'validation_false_version_budget':fp_budget,
        'paired_training_capture':args.paired_capture,'paired_negative_weight_relative_to_existing':.05 if args.paired_capture else 0,
        'measured_channel_augmentation':args.measured_channel,'measured_channel_weight_relative_to_existing':.1 if args.measured_channel else 0,
        'legacy_training_exclusions':sorted(training_exclusions),
        'validation_gate_choices':list(gates),
        'reserved_source_ids_unscored':[r['id'] for r in catalog['sources'] if r['role']=='reserved_review'],
        'limits':[f'{len(labels)} ten-second reviewed train intervals','new positives are development/training, not acceptance',
                  'historically reused evaluation','frozen baseline/backbone related material exposure unknown',
                  'head-only scoring omits live speech suppression','general train head is not a clatter-specific explanation',
                  'squeal-alone behavior of the inherited classifier is not constrained by this change']}
    for name,mask in [('validation',vm),('evaluation',(d['split']==2)&~np.isin(d['clip'],catalog['future_evaluation_excluded_legacy_rows']))]:
        gg=groups(mask);rr=records(predict(candidate,x[mask]),d['level'][mask],gg,gate);old=records(predict(base,x[mask]),d['level'][mask],gg)
        yy=np.array([bool(r['label']) for r in rr]);report[name]=f.metrics(yy,np.array([r['score'] for r in rr]),threshold)
        report[name+'_baseline']=f.metrics(yy,np.array([r['score'] for r in old]),.869)
        if args.paired_capture:
            bs,bt,bm,bg=baseline_recalibrated
            rec=records(predict(base,x[mask]),d['level'][mask],gg,bg)
            report[name+'_baseline_recalibrated']=dict(sensitivity=bs,threshold=bt,min_level_dbfs=bg,metrics=f.metrics(yy,np.array([r['score'] for r in rec]),bt))
        (ROOT/f'{name}-records.json').write_text(json.dumps(rr,indent=2))
    report['reviewed_development_results']=[];rp=predict(candidate,rx);rb=predict(base,rx)
    rp[reviewed['level']<gate]=0;rb[reviewed['level']<-65]=0
    for sid in np.unique(reviewed['source']):
        for v in (0,1):
            m=(reviewed['source']==sid)&(reviewed['version']==v)
            report['reviewed_development_results'].append(dict(source=int(sid) if str(sid).isdigit() else str(sid),version=v,baseline_score=f.event_score(rb[m]),candidate_score=f.event_score(rp[m])))
    if paired is not None:
        pp=predict(candidate,paired['x']);pb=predict(base,paired['x']);pp[paired['level']<gate]=0;pb[paired['level']<-65]=0
        report['paired_training_diagnostics']=[]
        for sid in np.unique(paired['source']):
            m=(paired['source']==sid)&(paired['version']==0)
            report['paired_training_diagnostics'].append(dict(source=sid,label=int(paired['label'][m][0]),baseline_score=f.event_score(pb[m]),candidate_score=f.event_score(pp[m]),candidate_trigger=bool(f.event_score(pp[m])>=threshold)))
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    lab=Path('/lab/work/audio-diagnostics');alignment=json.loads((lab/'ns-061-armed'/'alignment.json').read_text());assert alignment['accepted']
    lo,hi=alignment['source_first20_capture_interval_s'];engine=read_wav16(lab/'ns-061-armed'/'capture.wav')[round(lo*16000):round(hi*16000)]
    report['microphone']=[]
    for name,w in [('Colorado mixed freight',read_wav16(lab/'colorado-replay'/'captured.wav')),('NS engine-only',engine)]:
        windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,lv=channel.condition(windows)
        mx=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
        cp=predict(candidate,mx);bp=predict(base,mx);cp[lv < gate]=0;bp[lv < -65]=0
        report['microphone'].append(dict(name=name,baseline_score=f.event_score(bp),candidate_score=f.event_score(cp),candidate_trigger=bool(f.event_score(cp)>=threshold)))
        if args.paired_capture:
            bs,bt,bm,bg=baseline_recalibrated;bc=predict(base,mx);bc[lv < bg]=0
            report['microphone'][-1]['baseline_recalibrated']=dict(score=f.event_score(bc),threshold=bt,min_level_dbfs=bg,trigger=bool(f.event_score(bc)>=bt))
    (ROOT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
