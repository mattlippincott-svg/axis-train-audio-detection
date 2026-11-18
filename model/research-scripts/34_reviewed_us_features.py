"""Encode only the reviewed Natchez mixed-clatter interval, never gate-dominant clip 7."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import hashlib,json,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import butter,sosfilt,lfilter
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import yamnet,params

def main():
    import tensorflow as tf
    from tf_keras import Model,layers
    corpus=Path('experiments/us-freight-candidates/commons');root=Path('experiments/reviewed-us-data');root.mkdir(exist_ok=True)
    raw_labels=(corpus/'human-interval-labels.json').read_bytes();labels=json.loads(raw_labels)['labels']
    approved=[r for r in labels if r['source_id']=='commons-natchez-2011']
    assert len(approved)==1
    r=approved[0];assert r['review_status']=='HUMAN_REVIEWED' and r['rhythmic_clatter']=='present' and r['horn']=='absent' and r['bell']=='absent'
    assert (r['start_s'],r['end_s'])==(52,62)
    label_hash=hashlib.sha256(raw_labels).hexdigest()
    if (root/'features.npz').exists():
        assert json.loads((root/'provenance.json').read_text())['label_sha256']==label_hash
        print('reviewed US feature cache verified');return
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    w=read_wav16(corpus/'sources'/r['source_id']/'audio.wav')[round(r['start_s']*16000):round(r['end_s']*16000)]
    windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();channel=import_module('13_channel_robust');rng=np.random.default_rng(20261108)
    arrays={k:[] for k in ('x','source','version','level')}
    for v in range(8):
        raw=windows.copy()
        if v:
            raw=sosfilt(butter(2,[150,5000],btype='bandpass',fs=16000,output='sos'),raw,axis=1).astype(np.float32)
            if v>1:raw=lfilter([1,-rng.uniform(.3,.8)],[1],raw,axis=1).astype(np.float32)
            raw*=10**(-(24 if v==1 else rng.uniform(8,28))/20)
        normal,level=channel.condition(raw);x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
        for k,value in dict(x=x.astype(np.float16),source=np.full(len(x),r['source_id']),version=np.full(len(x),v),level=level).items():arrays[k].append(value)
    np.savez_compressed(root/'features.npz',**{k:np.concatenate(v) for k,v in arrays.items()})
    (root/'provenance.json').write_text(json.dumps({'label_sha256':label_hash,'included':[r],
        'excluded':['commons-bnsf-snow-2016'],'exclusion_reason':'crossing warning dominates; no reviewed independent clatter label',
        'limits':['ten seconds of mixed US freight clatter/background squeal','not clatter without squeal','reviewed source is training/development, not acceptance']},indent=2))
    print('encoded reviewed US interval',r['source_id'],len(windows)*8,flush=True)

if __name__=='__main__':main()
