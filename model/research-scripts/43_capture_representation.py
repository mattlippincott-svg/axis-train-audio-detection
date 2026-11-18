"""Read-only level/representation diagnostic; no fitting or device changes."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/capture-representation');root.mkdir(exist_ok=True)
    lab=Path('/lab');ch=import_module('13_channel_robust');f=import_module('06_foundation')
    p=layers.Input((96,64));s,e=yamnet.yamnet(p,params.Params());core=Model(p,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    nets={name:models.load_model(Path('experiments')/folder/'head.h5',compile=False) for name,folder in [('installed','channel-robust'),('candidate','reviewed-train-measured')]}
    manifest=json.loads((lab/'outputs/train-microphone-pairing-playback.json').read_text(encoding='utf-8-sig'))
    source=read_wav16(lab/'outputs/train-microphone-pairing-playback.wav')
    folder=lab/'work/audio-diagnostics/printer-paused-comparison-03'
    quality=json.loads((folder/'quality-and-alignment.json').read_text());assert quality['alignment_accepted']
    capture=read_wav16(folder/'capture.wav');records=[];vectors={}
    for seg,q in zip(manifest['source_segments'],quality['segments']):
        assert str(seg['source_id'])==str(q['source_id'])
        sid=str(seg['source_id'])
        for path,wave,start,end in [('source',source,seg['playback_start_s'],seg['playback_end_s']),('microphone',capture,q['capture_start_s'],q['capture_end_s'])]:
            w=wave[round(start*16000):round(end*16000)]
            windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy()
            normal,levels=ch.condition(windows)
            for target in ['installed',-38,-26,-14]:
                if target=='installed':a=normal
                else:
                    rms=np.sqrt(np.mean(windows**2,axis=1,keepdims=True)+1e-20)
                    gain=np.minimum(10**(target/20)/rms,.98/np.maximum(np.max(np.abs(windows),axis=1,keepdims=True),1e-12))
                    a=(windows*gain).astype(np.float32)
                x=embed(a).numpy();vectors[sid,path,str(target)]=x[:,:1024]
                scores={}
                for name,net in nets.items():
                    prob=net(x,training=False).numpy().ravel();g=prob.copy();g[levels < -65]=0
                    scores[name]={'ungated':f.event_score(prob),'original_level_gated':f.event_score(g)}
                records.append(dict(source=sid,path=path,target=target,expected_class=seg['expected_class'],scores=scores))
        for target in ['installed',-38,-26,-14]:
            a=vectors[sid,'source',str(target)];b=vectors[sid,'microphone',str(target)]
            cosine=np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
            next(r for r in reversed(records) if r['source']==sid and r['target']==target)['paired_embedding_cosine_median']=float(np.median(cosine))
    report={'records':records,'scope':'diagnostic only; capture 03 not fitted; no operating point selected',
        'limits':['shared source content is development data','target normalization may attenuate or exceed installed boost limit; not a deployment proposal','original raw volume gate retained separately','two negative intervals cannot establish false alarms','no device changes']}
    (root/'results.json').write_text(json.dumps(report,indent=2))
    for r in records:print(r,flush=True)

if __name__=='__main__':main()
