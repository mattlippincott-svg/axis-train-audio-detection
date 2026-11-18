"""Reviewed rhythmic rail expert plus supporting-only wheel-on-rail evidence.

Physical force is not a label. The exact 769890 excerpt is a wheel-on-rail
support positive, not promoted to a confirmed isolated-impact/clatter label.
Validation only; no deployment or reserved/test scoring.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import argparse,csv,json,pickle,sys,hashlib
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from sklearn.linear_model import LogisticRegression
from common import DATA,MANIFEST,read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate-sweep',action='store_true');parser.add_argument('--flatspots',action='store_true');parser.add_argument('--intermodal',action='store_true');args=parser.parse_args()
    if args.intermodal and not args.flatspots:parser.error('--intermodal requires --flatspots')
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/wheel-rail-support-gates' if args.gate_sweep else 'experiments/wheel-rail-support');root.mkdir(exist_ok=True)
    if args.flatspots:root=Path('experiments/wheel-rail-flatspots-gates' if args.gate_sweep else 'experiments/wheel-rail-flatspots');root.mkdir(exist_ok=True)
    if args.intermodal:root=Path('experiments/wheel-rail-intermodal-gates' if args.gate_sweep else 'experiments/wheel-rail-intermodal');root.mkdir(exist_ok=True)
    f=import_module('06_foundation');ch=import_module('13_channel_robust');rng=np.random.default_rng(20261110)
    paths=[Path('experiments/rail-corpus-v2/human-interval-labels.json'),Path('experiments/us-freight-candidates/commons/human-interval-labels.json'),Path('experiments/us-freight-candidates/review-round-2/human-interval-labels.json')]
    if args.flatspots:paths.append(Path('experiments/us-freight-candidates/review-round-2/human-11-12-labels.json'))
    approved=[]
    for path in paths:
        for r in json.loads(path.read_text(encoding='utf-8-sig'))['labels']:
            if r.get('review_status')=='HUMAN_REVIEWED' and r.get('training_approved',True) and r.get('rhythmic_clatter')=='present':approved.append((r,path.parent/'sources'/str(r['source_id'])/'audio.wav'))
    intermodalpath=Path('experiments/intermodal-reference/reference.json')
    if args.intermodal:
        r=json.loads(intermodalpath.read_text(encoding='utf-8-sig'))
        assert r['start_s']==35 and r['end_s']==85 and r['local_audio_origin_source_s']==35
        wavepath=intermodalpath.parent/r['local_audio_file'];assert hashlib.sha256(wavepath.read_bytes()).hexdigest()==r['audio_sha256']
        # File contains only the 50-second excerpt. Local 0 means source 35.
        approved.append((dict(source_id=r['source_id'],creator=r['creator'],family=r['family'],start_s=0,end_s=50,original_start_s=35,original_end_s=85),wavepath))
    expected={'216642','76078','636899','commons-natchez-2011','157864','256259'}
    if args.flatspots:expected.add('580528')
    if args.intermodal:expected.add('youtube-u5p9VbIsduc')
    assert {str(r['source_id']) for r,_ in approved}==expected
    targetpath=Path('experiments/wheel-on-rail-target.json');target=json.loads(targetpath.read_text(encoding='utf-8-sig'))
    assert target['review_status']=='HUMAN_REVIEWED' and target['wheel_on_rail']=='clearly_recognizable'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[targetpath]+([intermodalpath] if args.intermodal else [])}
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    def encode(w):
        windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,level=ch.condition(windows)
        return np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)]),level
    cache=Path('experiments/wheel-rail-support/reviewed-features.npz');cache_info=cache.parent/'cache-provenance.json'
    if args.flatspots:cache=root/'reviewed-features.npz';cache_info=root/'cache-provenance.json'
    if not cache.exists():
        arrays={k:[] for k in ('x','level','source','clatter','version')}
        examples=[(r,path,1) for r,path in approved]+[(target,Path('experiments/rail-sounds/769890.wav'),0)]
        for r,path,isclatter in examples:
            w=read_wav16(path)[round(r['start_s']*16000):round(r['end_s']*16000)]
            for v in range(4):
                raw=w.copy()
                if v:
                    raw=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),raw).astype(np.float32)
                    if v>1:raw=lfilter([1,-.65],[1],raw).astype(np.float32)
                    raw*=10**(-(24 if v==1 else float(rng.uniform(8,24)))/20)
                x,level=encode(raw)
                for k,value in dict(x=x.astype(np.float16),level=level,source=np.full(len(x),str(r['source_id'])),clatter=np.full(len(x),isclatter),version=np.full(len(x),v)).items():arrays[k].append(value)
            print('encoded approved interval',r['source_id'],r['start_s'],r['end_s'],flush=True)
        np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in arrays.items()})
        cache_info.write_text(json.dumps(dict(label_sha256=hashes),indent=2))
    assert json.loads(cache_info.read_text())['label_sha256']==hashes
    reviewed=dict(np.load(cache));paired=dict(np.load('experiments/paired-training-01/features.npz'))
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')));meta={}
    for part in ('dev','eval'):meta.update(json.loads((Path(DATA)/'meta'/f'{part}_clips_info_FSD50K.json').read_text()))
    sid=lambda r:r['id'].split('/')[-1].replace('.wav','')
    authors={r['creator'].casefold() for r,_ in approved}|{target['creator'].casefold()}
    cat=json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text())
    trainexcluded=set(cat['future_training_excluded_legacy_rows'])
    trainexcluded.update(json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())['future_training_excluded_legacy_rows'])
    related={i for i,r in enumerate(rows) if meta.get(sid(r),{}).get('uploader','').casefold() in authors}
    trainexcluded.update(related)
    evalexcluded=set(cat['future_evaluation_excluded_legacy_rows'])|related
    d=dict(np.load('experiments/channel-robust/features.npz'))
    neg=(d['split']==0)&(d['label']==0)&(d['level']>=-70)&~np.isin(d['clip'],list(trainexcluded))
    nx=d['x'][neg];nc=d['clip'][neg];nw=np.zeros(len(nx))
    for c in np.unique(nc):
        m=nc==c;mass=3 if any(t in rows[int(c)]['labels'].lower() for t in ('truck','engine','bus','skateboard','mechanism','fan','siren','bicycle')) else 1
        nw[m]=mass/m.sum()
    nw/=nw.sum()
    # Same-path training negatives are small supplementary confusers, not a
    # false-alarm test. No old Colorado/NS or comparison capture enters fitting.
    pn=paired['label']==0;pnw=np.zeros(pn.sum());pns=paired['source'][pn]
    for source in np.unique(pns):pnw[pns==source]=1/max(1,(pns==source).sum())
    pnw/=pnw.sum();nx=np.concatenate([nx,paired['x'][pn]]);nw=np.r_[nw*.9,pnw*.1]
    experts={};fit={}
    for name in ('rhythm','wheel'):
        m=(reviewed['level']>=-70)&((reviewed['clatter']==1) if name=='rhythm' else True)
        # The wheel-only reference remains outside the rhythm target entirely,
        # rather than being invented as either a rhythm positive or negative.
        px=reviewed['x'][m];sources=reviewed['source'][m];pw=np.zeros(len(px));families=np.unique(sources)
        for source in families:pw[sources==source]=1/max(1,(sources==source).sum())
        pm=(paired['label']==1)&np.isin(paired['source'],families);ps=paired['source'][pm];pmw=np.zeros(pm.sum())
        for source in np.unique(ps):
            pw[sources==source]*=.5;pmw[ps==source]=.5/max(1,(ps==source).sum())
        px=np.concatenate([px,paired['x'][pm]]);pw=np.r_[pw,pmw];pw/=pw.sum()
        x=np.concatenate([nx,px]).astype(np.float32);y=np.r_[np.zeros(len(nx)),np.ones(len(px))];weight=np.r_[nw,pw]*len(y)/2
        expert=LogisticRegression(C=.03,max_iter=750,random_state=20261110);expert.fit(x,y,sample_weight=weight);experts[name]=expert
        fit[name]=dict(source_ids=list(families),source_windows=int(m.sum()),microphone_training_windows=int(pm.sum()));print('fit',name,fit[name],flush=True)
    pickle.dump(experts,open(root/'experts.pkl','wb'))
    base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    def basepredict(x):return np.concatenate([base(x[j:j+2048],training=False).numpy().ravel() for j in range(0,len(x),2048)])
    vm=(d['split']==1)&~np.isin(d['clip'],list(evalexcluded));vx=d['x'][vm];vl=d['level'][vm];vc=d['clip'][vm];vv=d['version'][vm]
    groups=[(int(c),v,int(rows[int(c)]['label']),np.flatnonzero((vc==c)&(vv==v))) for c in np.unique(vc) for v in (0,1)]
    bp=basepredict(vx);bp[vl < -65]=0;cl=experts['rhythm'].decision_function(vx);wheel=experts['wheel'].predict_proba(vx)[:,1]
    oldfalse={(c,v) for c,v,label,ids in groups if label==0 and f.event_score(bp[ids])>=.869}
    def combine(b,cl,wp,levels,bias,support,gate=-65):
        cp=sigmoid(cl+bias)
        # Wheel evidence is supporting-only: cannot create a score without
        # rhythmic rail evidence. No added horn, bell, squeal or crossing path.
        original=b.copy();original[levels < -65]=0
        contribution=np.minimum(1,cp+support*wp*cp*(cp>=.5));contribution[levels < gate]=0
        p=np.maximum(original,contribution)
        return p,cp
    def records(bias,support,gate):
        p,_=combine(bp,cl,wheel,vl,bias,support,gate)
        return [dict(clip=c,version=v,label=label,score=f.event_score(p[ids]),base_score=f.event_score(bp[ids])) for c,v,label,ids in groups]
    variants=[]
    for gate,support in [(gate,support) for gate in ((-65,-70,-75) if args.gate_sweep else (-65,)) for support in (0.,.1)]:
        lo,hi=-30.,15.
        for _ in range(24):
            mid=(lo+hi)/2;rr=records(mid,support,gate);false={(r['clip'],r['version']) for r in rr if r['label']==0 and r['score']>=.869}
            if false<=oldfalse:lo=mid
            else:hi=mid
        bias=lo-.5;rr=records(bias,support,gate);y=np.array([bool(r['label']) for r in rr]);metrics=f.metrics(y,np.array([r['score'] for r in rr]),.869)
        variants.append(dict(min_level_dbfs=gate,support=support,bias=bias,validation=metrics,records=rr));print('variant',gate,support,bias,metrics,flush=True)
    selected=max(variants,key=lambda r:(r['validation']['tp'],-r['validation']['fp'],r['min_level_dbfs'],-r['support']))
    report=dict(target='audible rhythmic rail and wheel-on-rail recognition; no physical force requirement',training=fit,negative_training_clips=len(np.unique(nc)),excluded_training_rows=sorted(trainexcluded),excluded_evaluation_rows=sorted(evalexcluded),label_sha256=hashes,variants=variants,selected={k:v for k,v in selected.items() if k!='records'},threshold=.869,development_comparison=[],status='not deployed',limits=['few exact reviewed source families','sound heads are correlated, not independent proof of separate physical sources','new wheel support requires rhythm evidence but inherited general baseline behavior is unchanged','source-shared comparison captures are not independent acceptance','live speech suppression omitted','validation historically reused'])
    yy=np.array([bool(label) for c,v,label,ids in groups]);report['validation_baseline']=f.metrics(yy,np.array([f.event_score(bp[ids]) for c,v,label,ids in groups]),.869)
    for cid in ('paired-training-01','paired-comparison-02','printer-paused-comparison-03'):
        folder=Path('/lab/work/audio-diagnostics')/cid;q=json.loads((folder/'quality-and-alignment.json').read_text());wave=read_wav16(folder/'capture.wav')
        for seg in q['segments']:
            x,level=encode(wave[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)]);b=basepredict(x)
            cl=experts['rhythm'].decision_function(x);wp=experts['wheel'].predict_proba(x)[:,1]
            p,cp=combine(b,cl,wp,level,selected['bias'],selected['support'],selected['min_level_dbfs'])
            report['development_comparison'].append(dict(capture=cid,source=str(seg['source_id']),expected_class=seg['expected_class'],score=f.event_score(p),rhythm_ungated_score=f.event_score(cp),trigger=bool(f.event_score(p)>=.869)))
    (root/'results.json').write_text(json.dumps(report,indent=2));print('RESULT',report['selected'],report['development_comparison'],flush=True)

if __name__=='__main__':main()
