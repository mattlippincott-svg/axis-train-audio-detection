"""Replace one TRAIN-only channel family with randomized spectral tilt.

Keeps validation/test features unchanged. No played clip is a training input.
This is a simulated channel family, informed by a diagnostic capture comparison,
not a measured inverse correction of that capture.
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
from scipy.signal import lfilter,butter,sosfilt
from common import OUT,DATA,MANIFEST,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet


def main():
    import tensorflow as tf
    from tf_keras import Model,layers
    f=import_module('13_channel_robust');root=f.ROOT;cache=root/'features.npz'
    marker=root/'tilt-cache-info.json'
    if marker.exists():
        print('Tilt cache already prepared');return
    d=dict(np.load(cache));rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    clips=np.unique(d['clip'][(d['split']==0)&(d['version']==2)])
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(os.path.join(DATA,'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    rng=np.random.default_rng(20261106)
    for index,c in enumerate(clips):
        mask=(d['split']==0)&(d['version']==2)&(d['clip']==c);count=int(mask.sum())
        wave=read_wav16(os.path.join(DATA,rows[c]['path']))
        if len(wave)<15600:wave=np.pad(wave,(0,15600-len(wave)))
        n=1+(len(wave)-15600)//7680;ids=np.sort(rng.choice(n,count,replace=False))
        raw=np.stack([wave[i*7680:i*7680+15600] for i in ids])
        coefficient=float(rng.uniform(.45,.85))
        raw=lfilter([1.,-coefficient],[1.],raw,axis=1).astype(np.float32)
        raw=sosfilt(butter(2,6500,btype='lowpass',fs=16000,output='sos'),raw,axis=1).astype(np.float32)
        delay=int(rng.integers(160,1200));raw[:,delay:]+=.15*raw[:,:-delay]
        raw*=10**(-rng.uniform(8,28)/20)
        w,level=f.condition(raw);d['x'][mask]=embed(w).numpy().astype(np.float16);d['level'][mask]=level.astype(np.float16)
        if index%100==0:print('tilt',index+1,'/',len(clips),flush=True)
    temp=root/'features-tilt.npz';np.savez_compressed(temp,**d)
    original=root/'features-initial.npz'
    if not original.exists():cache.rename(original)
    temp.replace(cache)
    marker.write_text(json.dumps({'seed':20261106,'training_recordings':len(clips),
        'updated':'training version 2 only','validation_test_changed':False,
        'coefficient_range':[.45,.85],'limitation':'Channel design informed by the saved diagnostic capture; diagnostics are not pristine evaluation'},indent=2))


if __name__=='__main__':main()
