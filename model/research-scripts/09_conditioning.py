"""Measure bounded RMS conditioning against the same unconditioned classifier.

Fixed conditioning: -26 dBFS target, <=36 dB boost, .98 peak ceiling.
Raw level gate -65 dBFS remains. Validation chooses the operating threshold.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv
import json
import sys
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA, OUT, MANIFEST, read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params
import yamnet


def main():
    import tensorflow as tf
    from tf_keras import Model, layers
    from tf_keras.models import load_model
    f=import_module('06_foundation');root=Path(OUT).parent/'experiments'/'foundation'
    par=params.Params()
    inp=layers.Input((96,64));scores,emb=yamnet.yamnet(inp,par)
    core=Model(inp,[scores,emb]);core.load_weights(os.path.join(DATA,'yamnet.h5'))
    head=load_model(root/'baseline'/'head.h5',compile=False)
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def infer(w):
        stft=tf.abs(tf.signal.stft(w,400,160,512))
        lm=tf.math.log(tf.matmul(stft,mel)+.001)
        s,e=core(lm,training=False)
        return head(tf.concat([e,s],axis=1),training=False)[:,0]
    def condition(w):
        rms=np.sqrt(np.mean(w.astype(np.float64)**2,axis=1)+1e-12)
        peak=np.max(np.abs(w),axis=1)
        gain=np.maximum(1.,np.minimum(10**(36/20),10**(-26/20)/rms))
        gain=np.minimum(gain,.98/np.maximum(peak,1e-9))
        return (w*gain[:,None]).astype(np.float32),20*np.log10(rms)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    d=np.load(root/'evaluation-predictions.npz')
    rng=np.random.default_rng(20261103)
    chosen=[]
    for split in (1,2):
        ids=np.unique(d['clip'][d['split']==split])
        for label,limit in [('0',250),('1',150)]:
            pool=np.array([c for c in ids if rows[c]['label']==label])
            chosen.extend(rng.choice(pool,min(limit,len(pool)),replace=False).tolist())
    records=[]
    for index,c in enumerate(chosen):
        row=rows[c];wave=read_wav16(os.path.join(DATA,row['path']))[:16000*30]
        if len(wave)<15600:wave=np.pad(wave,(0,15600-len(wave)))
        w=np.lib.stride_tricks.sliding_window_view(wave,15600)[::7680].copy()
        result={'clip':int(c),'source':row['source'],'label':int(row['label']),'split':row['split'], 'variants':{}}
        for attenuation in (0,-24):
            raw=(w*10**(attenuation/20)).astype(np.float32)
            norm,levels=condition(raw)
            for name,a in [('raw',raw),('conditioned',norm)]:
                p=np.concatenate([infer(a[i:i+32]).numpy() for i in range(0,len(a),32)])
                p[levels < -65]=0
                result['variants'][f'{name}_{attenuation}']=f.event_score(p)
        records.append(result)
        if index%25==0:print(f'conditioning {index+1}/{len(chosen)}',flush=True)
    (root/'conditioning-records.json').write_text(json.dumps(records,indent=2))
    report={'settings':{'target_dbfs':-26,'max_gain_db':36,'boost_only':True,'peak_limit':.98,'raw_gate_dbfs':-65},
            'notes':['Deterministic exploratory subset, maximum first 30 seconds per clip',
                     'Same baseline head; no retraining or horn shortcut',
                     'Synthetic attenuation is a level stress test, not microphone simulation',
                     'Shared Freesound families excluded; previous test usage still applies'], 'variants':{}}
    for name in records[0]['variants']:
        va=[r for r in records if r['split']=='val'];te=[r for r in records if r['split']=='test']
        y=np.array([r['label']==1 for r in va]);s=np.array([r['variants'][name] for r in va])
        # 250 negative validation recordings: at most one false-positive clip.
        grid=np.linspace(.01,.99,197)
        eligible=[t for t in grid if f.metrics(y,s,t)['negative_clip_false_positive_rate']<=.005]
        threshold=max(eligible,key=lambda t:(f.metrics(y,s,t)['recall'],-float(t)))
        report['variants'][name]={'threshold':float(threshold),'validation':f.metrics(y,s,threshold),
            'test':f.metrics(np.array([r['label']==1 for r in te]),np.array([r['variants'][name] for r in te]),threshold)}
    (root/'conditioning-comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
