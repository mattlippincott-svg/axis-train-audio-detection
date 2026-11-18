"""Learn causal temporal train presence from known TRAIN audio only.

Validation selects checkpoint/threshold under the existing four-FP budget.
No test/reserved scoring, no deployment, no implied clatter-specific label.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import argparse,csv,json,sys,hashlib
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt
from common import DATA,MANIFEST,read_wav16
from temporal_context import context_indices
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path('experiments/learned-temporal');CONTEXT=8

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--residual',action='store_true');parser.add_argument('--full-train',action='store_true');parser.add_argument('--round-two',action='store_true');args=parser.parse_args()
    if args.residual:ROOT=Path('experiments/learned-temporal-residual')
    if args.full_train:
        if not args.residual:parser.error('--full-train requires --residual')
        ROOT=Path('experiments/learned-temporal-full')
    if args.round_two:
        if not args.full_train:parser.error('--round-two requires --full-train and --residual')
        ROOT=Path('experiments/learned-temporal-round-two')
    import tensorflow as tf
    from tf_keras import Model,layers,models,optimizers,utils,callbacks
    ROOT.mkdir(exist_ok=True);rng=np.random.default_rng(20261110);tf.random.set_seed(20261110)
    ch=import_module('13_channel_robust');f=import_module('06_foundation')
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    cat=json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text())
    excluded=set(cat['future_training_excluded_legacy_rows'])
    excluded.update(json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())['future_training_excluded_legacy_rows'])
    chosen=json.loads(Path('experiments/paired-acoustic/results.json').read_text(encoding='utf-8-sig'))['legacy_training_clips']
    if args.full_train:chosen=[c for c in json.loads(Path('experiments/channel-robust/selection.json').read_text())['clips'] if rows[c]['split']=='train' and c not in excluded]
    assert all(rows[c]['split']=='train' and c not in excluded for c in chosen)
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    def encode(w):
        if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
        windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,level=ch.condition(windows)
        x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
        return x.astype(np.float16),level
    cache=Path('experiments/learned-temporal/training-sequences.npz');provenance=Path('experiments/learned-temporal/training-provenance.json')
    if args.full_train:cache=ROOT/'training-sequences.npz';provenance=ROOT/'training-provenance.json'
    labelpaths=[Path('experiments/rail-corpus-v2/human-interval-labels.json'),Path('experiments/us-freight-candidates/commons/human-interval-labels.json')]
    if args.round_two:labelpaths.append(Path('experiments/us-freight-candidates/review-round-2/human-interval-labels.json'))
    labelhash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in labelpaths}
    capturepath=Path('/lab/work/audio-diagnostics/paired-training-01/capture.wav')
    capturehash=hashlib.sha256(capturepath.read_bytes()).hexdigest()
    if not cache.exists():
        arrays={k:[] for k in ('x','level','y','family','kind','sequence')};sequence=0;intervals=[]
        if args.round_two:
            basepath=Path('experiments/learned-temporal-full');baseprov=json.loads((basepath/'training-provenance.json').read_text())
            assert baseprov['legacy_clips']==chosen and baseprov['capture_sha256']==capturehash
            assert baseprov['label_sha256']=={str(p):labelhash[str(p)] for p in labelpaths[:-1]}
            previous=dict(np.load(basepath/'training-sequences.npz'))
            arrays={k:[previous[k]] for k in arrays};sequence=int(previous['sequence'].max())+1;intervals=baseprov['reviewed_intervals'].copy();del previous
        def add(w,label,family,kind):
            nonlocal sequence
            x,level=encode(w)
            values=dict(x=x,level=level,y=np.full(len(x),label),family=np.full(len(x),family),kind=np.full(len(x),kind),sequence=np.full(len(x),sequence))
            for k,v in values.items():arrays[k].append(v)
            sequence+=1
        for index,c in enumerate([] if args.round_two else chosen):
            r=rows[c];w=read_wav16(Path(DATA)/r['path'])[:16000*30]
            # One deterministic contiguous twelve-second segment, not random
            # independent frames. Context never crosses an unknown gap.
            if len(w)>16000*12:
                start=int(rng.integers(0,len(w)-16000*12+1));w=w[start:start+16000*12]
            add(w,int(r['label']),'legacy:'+str(c),'legacy')
            filtered=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w).astype(np.float32)*10**(-24/20)
            add(filtered,int(r['label']),'legacy:'+str(c),'legacy')
            if index%100==0:print('dense training features',index,len(chosen),flush=True)
        for path in labelpaths[-1:] if args.round_two else labelpaths:
            for r in json.loads(path.read_text(encoding='utf-8-sig'))['labels']:
                if r.get('review_status')!='HUMAN_REVIEWED':continue
                if r.get('training_approved') is False:continue
                if r.get('source_id')=='commons-bnsf-snow-2016':continue
                sid=str(r['source_id']);assert sid!='535384'
                w=read_wav16(path.parent/'sources'/sid/'audio.wav')[round(r['start_s']*16000):round(r['end_s']*16000)]
                add(w,1,sid,'reviewed');add(sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),w).astype(np.float32)*10**(-24/20),1,sid,'reviewed')
                intervals.append(dict(source=sid,start=r['start_s'],end=r['end_s'],label='train presence; not automatically clatter'))
        folder=capturepath.parent;q=json.loads((folder/'quality-and-alignment.json').read_text());assert q['alignment_accepted'];wave=read_wav16(capturepath)
        for seg in [] if args.round_two else q['segments']:
            w=wave[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)]
            for gain in (0,6,12):add(w*10**(gain/20),int(seg['expected_class']=='train'),str(seg['source_id']),'microphone')
        np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in arrays.items()})
        provenance.write_text(json.dumps(dict(label_sha256=labelhash,capture_sha256=capturehash,legacy_clips=chosen,reviewed_intervals=intervals,excluded_legacy_rows=sorted(excluded)),indent=2))
    prov=json.loads(provenance.read_text());assert prov['label_sha256']==labelhash and prov['capture_sha256']==capturehash and prov['legacy_clips']==chosen
    train=dict(np.load(cache));tx=train['x'];n=len(tx);weight=np.zeros(n,np.float32)
    legacy=train['kind']=='legacy'
    for sid in np.unique(train['family'][legacy]):
        m=legacy&(train['family']==sid)&(train['level']>=-70);weight[m]=1/max(1,m.sum())
    mass=weight.sum()
    for kind,total in [('reviewed',.2),('microphone',.1)]:
        families=np.unique(train['family'][train['kind']==kind])
        for sid in families:
            m=(train['kind']==kind)&(train['family']==sid)&(train['level']>=-80);weight[m]=mass*total/max(1,len(families))/max(1,m.sum())
    weight/=max(1e-9,weight.mean())
    # Prefixes come from other TRAIN negatives. Real silence is not an implicit
    # label cue and the model is trained with context at the first endpoint.
    training_groups=[]
    for sid in np.unique(train['sequence']):
        ids=np.flatnonzero(train['sequence']==sid);training_groups.append((str(train['family'][ids[0]]),ids))
    negative_sequences=[g for g in training_groups if legacy[g[1][0]] and train['y'][g[1][0]]==0]
    indices=context_indices(n,training_groups,negative_sequences,CONTEXT)
    oldhead=models.load_model('experiments/reviewed-train-measured/head.h5',compile=False)
    teacher=np.concatenate([oldhead(tx[i:i+2048],training=False).numpy().ravel() for i in range(0,n,2048)])
    # Fixed feature scaling from TRAIN only, shared by every validation/capture.
    mean=tx.astype(np.float32).mean(0);std=np.maximum(tx.astype(np.float32).std(0),.05)
    inp=layers.Input((CONTEXT,1545));z=layers.Lambda(lambda v:(v-tf.constant(mean))/tf.constant(std))(inp)
    z=layers.Dense(32,activation='relu')(z)
    z=layers.Conv1D(24,3,padding='causal',activation='relu')(z)
    z=layers.Conv1D(24,3,dilation_rate=2,padding='causal',activation='relu')(z)
    z=layers.Lambda(lambda v:v[:,-1,:])(z);z=layers.Dropout(.2)(z)
    if args.residual:
        oldhead.trainable=False
        current=oldhead(layers.Lambda(lambda v:v[:,-1,:])(inp))
        correction=layers.Dense(1,activation='tanh',kernel_initializer='zeros',bias_initializer='zeros')(z)
        out=layers.Lambda(lambda values:tf.sigmoid(tf.math.log(tf.clip_by_value(values[0],1e-5,1-1e-5)/(1-tf.clip_by_value(values[0],1e-5,1-1e-5)))+2*values[1]))([current,correction])
    else:out=layers.Dense(1,activation='sigmoid',bias_initializer=tf.constant_initializer(-2))(z)
    net=Model(inp,out)
    def loss(target,pred):
        label=target[:,0];t=target[:,1];anchor=target[:,2];p=tf.clip_by_value(tf.reshape(pred,[-1]),1e-5,1-1e-5)
        return -(label*tf.math.log(p)+(1-label)*tf.math.log(1-p))+anchor*(1-label)*tf.square(tf.nn.relu(p-t-.02))
    net.compile(optimizer=optimizers.Adam(1e-4),loss=loss)
    targets=np.stack([train['y'],teacher,legacy.astype(float)],axis=1).astype(np.float32)
    class Batches(utils.Sequence):
        def __init__(self):self.ids=np.arange(n)
        def __len__(self):return int(np.ceil(n/256))
        def __getitem__(self,j):
            ids=self.ids[j*256:(j+1)*256];return tx[indices[ids]].astype(np.float32),targets[ids],weight[ids]
        def on_epoch_end(self):rng.shuffle(self.ids)
    d=dict(np.load('experiments/channel-robust/features.npz'))
    vm=(d['split']==1)&~np.isin(d['clip'],cat['future_evaluation_excluded_legacy_rows'])
    vx=d['x'][vm];vc=d['clip'][vm];vv=d['version'][vm];vl=d['level'][vm];groups=[]
    for c in np.unique(vc):
        for v in (0,1):
            ids=np.flatnonzero((vc==c)&(vv==v));groups.append((int(c),v,int(rows[int(c)]['label']),ids))
    def sequences(x,gg,continuous):
        return context_indices(len(x),[(str(c),ids) for c,v,label,ids in gg],[(str(c),ids) for c,v,label,ids in gg if label==0],CONTEXT,continuous)
    vi=sequences(vx,groups,True)
    def predict(x,ii):
        p=np.concatenate([net(x[ii[j:j+512]].astype(np.float32),training=False).numpy().ravel() for j in range(0,len(ii),512)])
        assert np.isfinite(p).all(),'Non-finite temporal prediction'
        return p
    def records(p,levels,gg,gate=-65):
        p=p.copy();p[levels < gate]=0
        return [dict(clip=c,version=v,label=label,score=f.event_score(p[ids])) for c,v,label,ids in gg]
    def operating(rr):
        y=np.array([bool(r['label']) for r in rr]);scores=np.array([r['score'] for r in rr]);choices=[]
        for sensitivity in range(1,101):
            threshold=.95-.009*(sensitivity-1);metrics=f.metrics(y,scores,threshold)
            if metrics['fp']<=4:choices.append(dict(sensitivity=sensitivity,threshold=threshold,metrics=metrics))
        return max(choices,key=lambda r:(r['metrics']['tp'],-r['metrics']['fp'],r['threshold'])) if choices else None
    history=[]
    class Select(callbacks.Callback):
        best=None
        def on_epoch_end(self,epoch,logs=None):
            rr=records(predict(vx,vi),vl,groups);choice=operating(rr);history.append(dict(epoch=epoch+1,selection=choice));print('validation',epoch+1,choice,flush=True)
            if choice:
                key=(choice['metrics']['tp'],-choice['metrics']['fp'],choice['threshold'])
                if self.best is None or key>self.best:
                    self.best=key;net.save_weights(ROOT/'selected-weights.h5');(ROOT/'selected.json').write_text(json.dumps(dict(epoch=epoch+1,**choice),indent=2))
    batches=Batches();batches.on_epoch_end();selector=Select()
    if args.residual:selector.on_epoch_end(-1)
    net.fit(batches,epochs=8,verbose=2,workers=1,max_queue_size=2,callbacks=[selector])
    report=dict(history=history,status='not deployed',residual=args.residual,architecture='per-frame Dense32, causal Conv24 width3, causal Conv24 width3 dilation2, last-frame output; optional frozen measured head plus bounded +/-2 logit correction',context_frames=CONTEXT,receptive_frames=7,fit_capture='paired-training-01',comparison_captures_fitted=False,validation_context='preceding features from another non-train validation recording family; synthetic join',validation_false_budget=4,parameters=net.count_params(),limits=['legacy clip labels are weak; not every frame necessarily contains train','only six short reviewed positive source families','source-shared microphone comparisons are not independent acceptance','general presence head does not enforce supporting-only squeal policy','head-only scoring omits speech suppression','feature prefixes do not reproduce waveform join transients'])
    if selector.best is None:
        (ROOT/'results.json').write_text(json.dumps(report,indent=2));print('NO ELIGIBLE MODEL',flush=True);return
    net.load_weights(ROOT/'selected-weights.h5');selected=json.loads((ROOT/'selected.json').read_text());report['selected']=selected
    report['validation_records']=records(predict(vx,vi),vl,groups)
    report['validation_cold_start_records']=records(predict(vx,sequences(vx,groups,False)),vl,groups)
    # Weights are selected before any comparison capture is scored.
    report['development_comparison']=[]
    for cid in ('paired-training-01','paired-comparison-02','printer-paused-comparison-03'):
        folder=Path('/lab/work/audio-diagnostics')/cid;q=json.loads((folder/'quality-and-alignment.json').read_text());wave=read_wav16(folder/'capture.wav')
        # Actual preceding capture audio is retained as context, rather than
        # assuming each reviewed excerpt starts after silence.
        x,level=encode(wave);ii=np.array([[max(0,j-CONTEXT+1+k) for k in range(CONTEXT)] for j in range(len(x))]);p=predict(x,ii);p[level < -65]=0
        starts=np.arange(len(x))*7680/16000
        for seg in q['segments']:
            m=(starts>=seg['capture_start_s'])&(starts+.975<=seg['capture_end_s']);score=f.event_score(p[m])
            report['development_comparison'].append(dict(capture=cid,source=str(seg['source_id']),expected_class=seg['expected_class'],score=score,trigger=bool(score>=selected['threshold'])))
    (ROOT/'results.json').write_text(json.dumps(report,indent=2));print('RESULT',selected,report['development_comparison'],flush=True)

if __name__=='__main__':main()
