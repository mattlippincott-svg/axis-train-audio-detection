"""Read-only test of coarse TRAIN-pair magnitude correction on both captures.

Preserves each excerpt's overall RMS and uses ORIGINAL levels for event gates.
This is a lab-path diagnosis, not a proposed field microphone equalizer.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,sys
from pathlib import Path
from importlib import import_module
import numpy as np
from scipy.signal import firwin2,lfilter
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import yamnet,params

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/capture-level-comparison');profile=json.loads(Path('experiments/measured-capture-path/profile.json').read_text())
    edges=np.array(profile['band_edges_hz']);centers=np.sqrt(edges[:-1]*edges[1:]);db=-np.array(profile['relative_shape_db'])
    taps=firwin2(129,np.r_[0,centers,8000]/8000,10**(np.r_[db[0],db,db[-1]]/20))
    channel=import_module('13_channel_robust');f=import_module('06_foundation')
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    models_to_test={}
    for name,folder in [('installed','channel-robust'),('source_reviewed','reviewed-train-us'),('measured_path_trained','reviewed-train-measured')]:
        path=Path('experiments')/folder;cfg=dict(threshold=.869,min_level_dbfs=-65) if name=='installed' else json.loads((path/'selected.json').read_text())
        models_to_test[name]=(models.load_model(path/'head.h5',compile=False),cfg)
    results=[]
    for capture_id in ('paired-training-01','paired-comparison-02'):
        folder=Path('/lab/work/audio-diagnostics')/capture_id;quality=json.loads((folder/'quality-and-alignment.json').read_text());assert quality['alignment_accepted']
        wave=read_wav16(folder/'capture.wav')
        for seg in quality['segments']:
            w=wave[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)]
            corrected=lfilter(taps,[1],w).astype(np.float32);corrected*=np.sqrt((np.mean(w.astype(np.float64)**2)+1e-12)/(np.mean(corrected.astype(np.float64)**2)+1e-12))
            original=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy()
            original_levels=20*np.log10(np.sqrt(np.mean(original.astype(np.float64)**2,axis=1)+1e-12))
            for variant,audio in [('raw',w),('coarse_shape_corrected_rms_matched',corrected)]:
                windows=np.lib.stride_tricks.sliding_window_view(audio,15600)[::7680].copy();normal,_=channel.condition(windows)
                x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
                for name,(model,cfg) in models_to_test.items():
                    p=model(x,training=False).numpy().ravel();q=p.copy();q[original_levels<cfg.get('min_level_dbfs',-65)]=0
                    score=f.event_score(q)
                    results.append(dict(capture_id=capture_id,source=str(seg['source_id']),variant=variant,model=name,
                        score=score,ungated_score=f.event_score(p),threshold=cfg['threshold'],trigger=bool(score>=cfg['threshold'])))
    report={'results':results,'filter_derived_only_from':'paired-training-01',
        'limits':['coarse magnitude correction cannot undo all acoustic or phase effects','RMS matching does not guarantee matched perception',
                  'raw gate levels were preserved','not a calibrated mic correction or a field deployment recommendation','no fitting or device changes']}
    (root/'shape-diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
