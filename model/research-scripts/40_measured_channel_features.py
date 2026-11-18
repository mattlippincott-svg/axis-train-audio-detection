"""Apply the TRAINING-pair capture profile to diverse original TRAIN recordings.

The room profile never uses Colorado/NS comparisons or reserved audio. These
remain simulations, not additional independent microphone recordings.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,json,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import lfilter
from common import DATA,MANIFEST,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import yamnet,params

def main():
    import tensorflow as tf
    from tf_keras import Model,layers
    root=Path('experiments/measured-capture-path');cache=root/'features.npz'
    if cache.exists():print('existing measured-channel feature cache');return
    profile=json.loads((root/'profile.json').read_text());taps=np.load(root/'magnitude-fir.npy');bed=read_wav16(root/'room-background.wav')
    bg=20*np.log10(np.sqrt(np.mean(bed.astype(np.float64)**2)+1e-12))
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')));old=dict(np.load('experiments/channel-robust/features.npz'))
    excluded=set(json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text())['future_training_excluded_legacy_rows'])
    excluded.update(json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())['future_training_excluded_legacy_rows'])
    rng=np.random.default_rng(20261109);chosen=[]
    for label in (0,1):
        pool=[int(c) for c in np.unique(old['clip'][(old['split']==0)&(old['label']==label)]) if int(c) not in excluded]
        chosen.extend(rng.choice(pool,min(300,len(pool)),replace=False).tolist())
    assert all(rows[c]['split']=='train' for c in chosen)
    (root/'selection.json').write_text(json.dumps({'seed':20261109,'clips':chosen,'scope':'training only, up to 300 per class','excluded_rows':sorted(excluded)},indent=2))
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    channel=import_module('13_channel_robust');arrays={k:[] for k in ('x','clip','label','level')}
    for count,c in enumerate(chosen):
        r=rows[c];w=read_wav16(Path(DATA)/r['path'])[:16000*30]
        if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
        filtered=lfilter(taps,[1],w).astype(np.float32)*10**((profile['reference_gain_db']+rng.uniform(-6,6))/20)
        n=1+(len(w)-15600)//7680;ids=np.unique(np.round(np.linspace(0,n-1,min(12,n))).astype(int))
        raw=np.stack([filtered[i*7680:i*7680+15600] for i in ids])
        signal_level=20*np.log10(np.sqrt(np.mean(raw.astype(np.float64)**2,axis=1)+1e-12))
        keep=(signal_level>=bg+10) if r['label']=='1' else np.ones(len(raw),bool)
        raw=raw[keep]
        if len(raw):
            ambience=np.resize(bed,raw.size).reshape(raw.shape)*10**(rng.uniform(-6,6)/20)
            raw+=ambience;assert np.abs(raw).max()<.98
            normal,level=channel.condition(raw);x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
            for k,value in dict(x=x.astype(np.float16),clip=np.full(len(x),c),label=np.full(len(x),int(r['label'])),level=level).items():arrays[k].append(value)
        if count%50==0:print('measured-channel training features',count,len(chosen),flush=True)
    np.savez_compressed(cache,**{k:np.concatenate(v) for k,v in arrays.items()})
    print('feature rows',sum(len(v) for v in arrays['x']),flush=True)

if __name__=='__main__':main()
