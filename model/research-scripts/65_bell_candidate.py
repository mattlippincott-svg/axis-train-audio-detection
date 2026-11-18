"""Single-family exploratory bell expert; negative-only validation calibration.

No held-out test scoring or deployment. Not sufficient for a production bell model.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import argparse,csv,json,pickle,sys,importlib,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfilt
from sklearn.linear_model import LogisticRegression
from common import read_wav16
from rail_fusion import sigmoid
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round-two',action='store_true');args=parser.parse_args()
    import tensorflow as tf
    from tf_keras import Model,layers,models
    f=importlib.import_module('06_foundation');ch=importlib.import_module('13_channel_robust')
    root=Path('experiments/bell-candidate-round-two' if args.round_two else 'experiments/bell-candidate');root.mkdir(exist_ok=True)
    labels=Path('experiments/bell-audit/human-labels.json');review=json.loads(labels.read_text())
    assert review['labels'][0]['split']=='train' and review['labels'][1]['split']=='val'
    rows=list(csv.DictReader(open('data/manifest.csv',encoding='utf-8')))
    meta=json.loads(Path('data/meta/dev_clips_info_FSD50K.json').read_text())
    related={c for c,r in enumerate(rows) if meta.get(r['id'].split('/')[-1],{}).get('uploader')=='Corsica_S'}
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights('data/yamnet.h5')
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    def encode(w):
        raw=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy()
        normal,level=ch.condition(raw)
        return np.concatenate([embed(normal[j:j+32]).numpy() for j in range(0,len(normal),32)]),level
    sources_fit=[read_wav16('data/clips/fsd50k/dev/28777.wav')]
    contrast=None;newlabels=None
    if args.round_two:
        newlabels=Path('experiments/bell-external-check/human-labels.json');newreview=json.loads(newlabels.read_text())
        assert [(r['source_id'],r['start_s'],r['end_s']) for r in newreview['labels']]==[('854498',9,12),('752578',0,30)]
        sources_fit.append(read_wav16('experiments/bell-external-check/sources/854498/audio.wav')[9*16000:12*16000])
        contrast=read_wav16('experiments/bell-external-check/sources/752578/audio.wav')[:30*16000]
    def augment(source):
        parts=[]
        for band in (False,True):
            w=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),source).astype(np.float32) if band else source
            for gain in (0,-12,-24):parts.append(encode(w*10**(gain/20))[0])
        return np.concatenate(parts)
    positive=[augment(source) for source in sources_fit];px=np.concatenate(positive)
    pw=np.concatenate([np.full(len(a),1/(len(positive)*len(a))) for a in positive])
    archive=np.load('experiments/channel-robust/features.npz')
    clip=archive['clip'];split=archive['split'];version=archive['version'];level=archive['level'];x=archive['x'].astype(np.float32)
    existing=json.loads(Path('experiments/wheel-rail-intermodal-gates/results.json').read_text())
    exclude=set(existing['excluded_training_rows'])|related
    negatives=(split==0)&np.array([rows[int(c)]['label']=='0' for c in clip])&~np.isin(clip,list(exclude))&(level>=-70)
    nx=x[negatives];nc=clip[negatives];nw=np.zeros(len(nx))
    for c in np.unique(nc):
        m=nc==c;mass=4 if 'bell' in rows[int(c)]['labels'].lower() else 1;nw[m]=mass/m.sum()
    nw/=nw.sum()
    if contrast is not None:
        cx=augment(contrast);nx=np.concatenate([nx,cx]);nw=np.r_[nw*.8,np.full(len(cx),.2/len(cx))]
    fitx=np.concatenate([nx,px]);y=np.r_[np.zeros(len(nx)),np.ones(len(px))]
    weight=np.r_[nw,pw]*len(y)/2
    expert=LogisticRegression(C=.03,max_iter=750,random_state=20261115)
    expert.fit(fitx,y,sample_weight=weight);pickle.dump(expert,open(root/'expert.pkl','wb'))
    base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    rhythm=pickle.load(open('experiments/wheel-rail-intermodal-gates/experts.pkl','rb'))['rhythm'];rbias=existing['selected']['bias']
    def current(xx):
        bp=np.concatenate([base(xx[j:j+2048],training=False).numpy().ravel() for j in range(0,len(xx),2048)])
        return np.maximum(bp,sigmoid(rhythm.decision_function(xx)+rbias))
    ve=set(existing['excluded_evaluation_rows'])|related
    vm=(split==1)&~np.isin(clip,list(ve));vx=x[vm];vc=clip[vm];vv=version[vm];vl=level[vm]
    bp=current(vx);bp[vl < -90]=0;cl=expert.decision_function(vx)
    contrast_logits=expert.decision_function(cx) if contrast is not None else None
    groups=[(int(c),v,int(rows[int(c)]['label']),np.flatnonzero((vc==c)&(vv==v))) for c in np.unique(vc) for v in (0,1)]
    baseline=[dict(clip=c,version=v,label=label,score=f.event_score(bp[ix])) for c,v,label,ix in groups]
    oldfalse={(r['clip'],r['version']) for r in baseline if not r['label'] and r['score']>=.869}
    def records(bias):
        p=np.maximum(bp,sigmoid(cl+bias));p[vl < -90]=0
        return [dict(clip=c,version=v,label=label,score=f.event_score(p[ix])) for c,v,label,ix in groups]
    lo,hi=-30.,15.
    for _ in range(22):
        mid=(lo+hi)/2;rr=records(mid)
        false={(r['clip'],r['version']) for r in rr if not r['label'] and r['score']>=.869}
        if false<=oldfalse and (contrast_logits is None or np.max(contrast_logits+mid)<=0):lo=mid
        else:hi=mid
    bias=lo-.5;rr=records(bias)
    yy=np.array([bool(r['label']) for r in rr])
    short_bias=None;short_report=None
    def short_score(p):
        return float(np.lib.stride_tricks.sliding_window_view(p,5).min(1).max()) if len(p)>=5 else 0.
    if args.round_two:
        # Separate exploratory bell path: five raw consecutive scores (2-second
        # hold at .48-second hops). General head timing is left untouched.
        def short_records(offset):
            p=sigmoid(cl+offset);p[vl < -90]=0
            return [dict(clip=c,version=v,label=label,score=max(baseline[n]['score'],short_score(p[ix]))) for n,(c,v,label,ix) in enumerate(groups)]
        lo,hi=-30.,15.
        for _ in range(22):
            mid=(lo+hi)/2;sr=short_records(mid)
            false={(r['clip'],r['version']) for r in sr if not r['label'] and r['score']>=.869}
            if false<=oldfalse and np.max(contrast_logits+mid)<=0:lo=mid
            else:hi=mid
        short_bias=lo-.5;sr=short_records(short_bias)
        short_report=dict(expert_bias=short_bias,hold_windows=5,hop_s=.48,smoothing='none on independent expert path',threshold=.869,validation=f.metrics(yy,np.array([r['score'] for r in sr]),.869),scope='Offline independent bell-informed path; inherited train path unchanged; not implemented in device runtime.')
    diagnostics=[]
    sources=[(sid,'training fit' if sid=='28777' else 'held-out bell version; development validation',read_wav16(f'data/clips/fsd50k/dev/{sid}.wav')) for sid in ('28777','127451')]
    microphone=read_wav16('/lab/work/audio-diagnostics/ns-061-armed/capture.wav')
    sources.extend([('ns-marked-bell','development microphone; approximate bell-mark to fading interval',microphone[68*16000:76*16000]),
        ('ns-engine-first20','development microphone; aligned source first20s before marked bell',microphone[round(23.23025*16000):round(43.23025*16000)])])
    if args.round_two:
        sources.extend([('freight-9-12','training fit; exact clear bell; too short for isolated sustained-trigger evaluation',sources_fit[1]),
            ('steam-0-30','training bell-only contrast; remains train positive',contrast),
            ('electronic-crossing','development challenge; not used for fitting',read_wav16('experiments/bell-external-check/sources/737301/audio.wav'))])
    for sid,role,original in sources:
        for gain in ((0,-24,-48) if sid.isdecimal() else (0,)):
            w=original*10**(gain/20);xx,ll=encode(w);b=current(xx);bell=sigmoid(expert.decision_function(xx)+bias);new=np.maximum(b,bell)
            b[ll < -90]=0;new[ll < -90]=0;bell[ll < -90]=0
            diagnostics.append(dict(source_id=sid,role=role,digital_gain_db=gain,rms_dbfs=float(20*np.log10(max(1e-12,np.sqrt(np.mean(w*w))))),peak_dbfs=float(20*np.log10(max(1e-12,np.max(np.abs(w))))),current_sustained=f.event_score(b),candidate_sustained=f.event_score(new),new_expert_sustained=f.event_score(bell),new_expert_window_min=float(bell.min()),new_expert_window_median=float(np.median(bell)),new_expert_window_max=float(bell.max()),windows=len(bell)))
            if short_bias is not None:
                short=sigmoid(expert.decision_function(xx)+short_bias);short[ll < -90]=0
                diagnostics[-1]['short_path_score']=short_score(short)
    report=dict(status='exploratory; not exported or installed',label_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),positive_families=1,positive_augmented_windows=len(px),negative_clips=len(np.unique(nc)),bias=bias,threshold=.869,gate=-90,calibration='Largest bell bias preserving current validation negative false-trigger set, minus .5 logit safety margin; no positive validation score selects bias.',validation_current=f.metrics(yy,np.array([r['score'] for r in baseline]),.869),validation_candidate=f.metrics(yy,np.array([r['score'] for r in rr]),.869),diagnostics=diagnostics,related_creator_rows_excluded=sorted(related),limits=['Single positive recording family; cannot establish generalization.', 'No test partition scored. Validation historically reused.', 'Only raw model sustained scores, without live independent horn triggers or event state.', 'Boost-only input conditioning unchanged; gain variants do not simulate real distance or microphones.'])
    report['limits'].extend(['Positive clip includes unlabelled overlapping train sounds; new expert is not a verified bell subtype classifier.', 'Digital attenuation checks are diagnostic, not new independent recordings.'])
    if args.round_two:
        report['positive_families']=2
        report['short_path_experiment']=short_report
        report['limits'][0]='Two positive recording families; insufficient to establish generalization.'
        report['external_label_sha256']=hashlib.sha256(newlabels.read_bytes()).hexdigest()
        report['bell_only_contrast']='Source 752578, 0–30s only; not audible bell, but remains train-positive in base and rhythm heads.'
        report['calibration']+=' Round two also caps the expert on augmented training bell-absent steam windows at 0.5 before the .5-logit safety margin.'
        report['source_audio_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('data/clips/fsd50k/dev/28777.wav'),Path('experiments/bell-external-check/sources/854498/audio.wav'),Path('experiments/bell-external-check/sources/752578/audio.wav')]}
    (root/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
