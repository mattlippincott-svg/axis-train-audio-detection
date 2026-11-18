"""Encode aligned actual-microphone training examples, including same-path negatives."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import hashlib,json,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import yamnet,params

def main():
    import tensorflow as tf
    from tf_keras import Model,layers
    folder=Path('/lab/work/audio-diagnostics/paired-training-01');root=Path('experiments/paired-training-01');root.mkdir(exist_ok=True)
    quality=json.loads((folder/'quality-and-alignment.json').read_text());assert quality['alignment_accepted'] and quality['corroborating_anchors']>=3
    wave=read_wav16(folder/'capture.wav');bg=quality['silence_median_rms_dbfs']
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    channel=import_module('13_channel_robust');arrays={k:[] for k in ('x','source','label','level','version')};summary=[]
    for seg in quality['segments']:
        assert seg['complete'];sid=str(seg['source_id']);assert sid!='commons-bnsf-snow-2016'
        lo=round(seg['capture_start_s']*16000);hi=round(seg['capture_end_s']*16000)
        windows=np.lib.stride_tricks.sliding_window_view(wave[lo:hi],15600)[::7680].copy()
        raw_level=20*np.log10(np.sqrt(np.mean(windows.astype(np.float64)**2,axis=1)+1e-12))
        keep=raw_level>=bg+10
        assert keep.any(),sid
        summary.append(dict(source_id=sid,label=seg['expected_class'],windows_total=len(windows),windows_kept=int(keep.sum())))
        for version,gain_db in enumerate((0,6,12)):
            raw=windows[keep]*10**(gain_db/20);assert np.abs(raw).max()<.98
            normal,level=channel.condition(raw);x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
            values=dict(x=x.astype(np.float16),source=np.full(len(x),sid),label=np.full(len(x),int(seg['expected_class']=='train')),
                        level=level,version=np.full(len(x),version))
            for k,v in values.items():arrays[k].append(v)
    np.savez_compressed(root/'features.npz',**{k:np.concatenate(v) for k,v in arrays.items()})
    report={'capture_sha256':hashlib.sha256((folder/'capture.wav').read_bytes()).hexdigest(),'quality':quality,
        'segments':summary,'digital_gain_augmentations_db':[0,6,12],
        'quality_floor':'raw window at least 10 dB above silence-median recording level; not the live event gate',
        'scope':'new paired TRAINING capture only; old Colorado and NS captures never enter fitting'}
    (root/'provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
