"""Adapt independent upper acoustic branch; comparison captures never fit.

Fixed two-epoch exploratory experiment. Full legacy validation determines whether
it merits further work; no deployment or evaluation-set threshold selection.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import argparse,csv,json
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt
from common import DATA,MANIFEST,read_wav16

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--anchored',action='store_true');args=parser.parse_args()
    import tensorflow as tf
    from tf_keras import models,optimizers
    root=Path('experiments/paired-acoustic-anchored' if args.anchored else 'experiments/paired-acoustic');root.mkdir(exist_ok=True)
    ch=import_module('13_channel_robust');upper=import_module('17_upper_finetune');f=import_module('06_foundation')
    tf.random.set_seed(20261106);rng=np.random.default_rng(20261106)
    core,tail,head,model,extract=upper.components()
    head.set_weights(models.load_model('experiments/reviewed-train-measured/head.h5',compile=False).get_weights())
    # Freeze decision head: this experiment changes acoustic representation only.
    head.trainable=False
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    cat=json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text())
    excluded=set(cat['future_training_excluded_legacy_rows'])
    excluded.update(json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())['future_training_excluded_legacy_rows'])
    chosen=json.loads(Path('experiments/channel-robust/selection.json').read_text())['clips']
    selected=[]
    for label in ('0','1'):
        pool=[c for c in chosen if rows[c]['split']=='train' and rows[c]['label']==label and c not in excluded]
        selected.extend(rng.choice(pool,min(300,len(pool)),replace=False).tolist())
    arrays={k:[] for k in ('lower','tags','y','weight','teacher','anchor')}
    def add(raw,label,mass,anchor=False):
        normal,levels=ch.condition(raw);lo,ta=extract(normal)
        teacher=head(tf.concat([tail(lo,training=False),ta],axis=1),training=False).numpy().ravel()
        weights=np.full(len(raw),mass/max(1,len(raw)),np.float32);weights[levels < -80]=0
        for k,v in dict(lower=lo.numpy().astype(np.float16),tags=ta.numpy().astype(np.float16),y=np.full(len(raw),label,np.float32),weight=weights,teacher=teacher,anchor=np.full(len(raw),anchor,np.float32)).items():arrays[k].append(v)
    def windows(w):
        if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
        return np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy()
    for ix,c in enumerate(selected):
        r=rows[c];w=windows(read_wav16(Path(DATA)/r['path'])[:16000*30]);ids=np.sort(rng.choice(len(w),min(4,len(w)),replace=False));w=w[ids]
        add(w,int(r['label']),1,anchor=True);add(upper.transform(w,20261106+c),int(r['label']),1,anchor=True)
        if ix%100==0:print('legacy training features',ix,len(selected),flush=True)
    lab=Path('/lab');playlist=json.loads((lab/'outputs/train-microphone-pairing-playback.json').read_text(encoding='utf-8-sig'))
    source=read_wav16(lab/'outputs/train-microphone-pairing-playback.wav')
    folder=lab/'work/audio-diagnostics/paired-training-01';quality=json.loads((folder/'quality-and-alignment.json').read_text());assert quality['alignment_accepted']
    capture=read_wav16(folder/'capture.wav')
    # Total paired family mass 20% of legacy mass, split across five families,
    # source/capture and three gain versions. All comparisons remain unfitted.
    mass=len(selected)*2*.2/5/2/3
    for seg,q in zip(playlist['source_segments'],quality['segments']):
        assert str(seg['source_id'])==str(q['source_id'])
        for wave,start,end in [(source,seg['playback_start_s'],seg['playback_end_s']),(capture,q['capture_start_s'],q['capture_end_s'])]:
            w=windows(wave[round(start*16000):round(end*16000)])
            for gain in (0,6,12):add(w*10**(gain/20),int(seg['expected_class']=='train'),mass)
    d={k:np.concatenate(v) for k,v in arrays.items()};del arrays
    d['weight']/=d['weight'].mean();reference=[v.numpy().copy() for v in core.weights]
    def anchored_loss(target,pred):
        label=target[:,0];teacher=tf.clip_by_value(target[:,1],1e-5,1-1e-5);anchor=target[:,2]
        p=tf.clip_by_value(tf.reshape(pred,[-1]),1e-5,1-1e-5)
        bce=-(label*tf.math.log(p)+(1-label)*tf.math.log(1-p))
        soft=-(teacher*tf.math.log(p)+(1-teacher)*tf.math.log(1-p))
        upward=tf.square(tf.nn.relu(p-teacher-.02))
        return bce+anchor*(.5*soft+10*(1-label)*upward)
    model.compile(optimizer=optimizers.Adam(3e-6),loss=anchored_loss if args.anchored else 'binary_crossentropy')
    target=np.stack([d['y'],d['teacher'],d['anchor']],axis=1) if args.anchored else d['y']
    model.fit([d['lower'].astype(np.float32),d['tags'].astype(np.float32)],target,sample_weight=d['weight'],batch_size=64,epochs=2,verbose=2,shuffle=True)
    assert all(np.array_equal(a,b.numpy()) for a,b in zip(reference,core.weights))
    tail.save(root/'adapted-upper.h5');head.save(root/'head.h5');del d
    def score(w):
        normal,levels=ch.condition(w);pred=[]
        for i in range(0,len(w),32):
            lo,ta=extract(normal[i:i+32]);pred.extend(model([lo,ta],training=False).numpy().ravel())
        p=np.array(pred);p[levels < -65]=0
        return f.event_score(p)
    records=[]
    for ix,c in enumerate([c for c in chosen if rows[c]['split']=='val' and c not in cat['future_evaluation_excluded_legacy_rows']]):
        r=rows[c];w=windows(read_wav16(Path(DATA)/r['path'])[:16000*30])
        for v in (0,1):
            raw=w if v==0 else sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w,axis=1).astype(np.float32)*10**(-24/20)
            records.append(dict(clip=c,version=v,label=int(r['label']),score=score(raw)))
        if ix%100==0:print('validation',ix,flush=True)
    baseline=json.loads(Path('experiments/reviewed-train-measured/validation-records.json').read_text())
    assert {(r['clip'],r['version']) for r in records}=={(r['clip'],r['version']) for r in baseline}
    y=np.array([bool(r['label']) for r in records]);s=np.array([r['score'] for r in records])
    options=[(sens,.95-.009*(sens-1)) for sens in range(1,101)]
    eligible=[(sens,t,f.metrics(y,s,t)) for sens,t in options if np.sum((~y)&(s>=t))<=4]
    choice=max(eligible,key=lambda v:(v[2]['tp'],-v[2]['fp'],v[1])) if eligible else None
    report={'selection':choice,'anchored_training':args.anchored,'anchor_policy':'legacy train soft-target BCE weight 0.5 plus negative upward squared penalty weight 10 above teacher +0.02' if args.anchored else None,'fit_capture':'paired-training-01','comparison_captures_fitted':False,'legacy_training_clips':selected,'frozen_sound_tags_unchanged':True,'epochs':2,'learning_rate':3e-6,'head_frozen':True,'validation_records':records,'development_comparison':[],'status':'not deployed','limits':['same-content microphone comparison is not unseen-source acceptance','general head does not enforce squeal supporting-only policy','offline scoring omits speech suppression']}
    for cid in ('paired-training-01','paired-comparison-02','printer-paused-comparison-03'):
        folder=lab/'work/audio-diagnostics'/cid;q=json.loads((folder/'quality-and-alignment.json').read_text());wave=read_wav16(folder/'capture.wav')
        for seg in q['segments']:
            value=score(windows(wave[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)]))
            report['development_comparison'].append(dict(capture=cid,source=str(seg['source_id']),expected_class=seg['expected_class'],score=value,trigger=bool(value>=choice[1]) if choice else None))
    (root/'results.json').write_text(json.dumps(report,indent=2));print('RESULT',choice,report['development_comparison'],flush=True)

if __name__=='__main__':main()
