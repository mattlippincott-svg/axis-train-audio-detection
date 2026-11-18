"""Export separate unchanged train head and calibrated short-bell expert."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
from common import read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet,features

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/bell-candidate-round-two');report=json.loads((root/'results.json').read_text())
    oldroot=Path('experiments/wheel-rail-intermodal-gates');oldreport=json.loads((oldroot/'results.json').read_text())
    rhythm=pickle.load(open(oldroot/'experts.pkl','rb'))['rhythm'];bell=pickle.load(open(root/'expert.pkl','rb'))
    inp=layers.Input(batch_shape=(1,15600),dtype=tf.float32,name='waveform')
    def condition(w):
        rms=tf.sqrt(tf.reduce_mean(w*w,axis=1,keepdims=True)+1e-12);peak=tf.reduce_max(tf.abs(w),axis=1,keepdims=True)
        gain=tf.maximum(1.,tf.minimum(tf.constant(10**1.8,tf.float32),tf.constant(10**(-26/20),tf.float32)/rms));gain=tf.minimum(gain,.98/tf.maximum(peak,1e-9));return w*gain
    w=layers.Lambda(condition)(inp);w=layers.Lambda(lambda v:tf.reshape(v,[15600]))(w)
    _,patches=features.waveform_to_log_mel_spectrogram_patches(w,params.Params(tflite_compatible=True))
    s,e=yamnet.yamnet(patches,params.Params(tflite_compatible=True));backbone=Model(inp,[s,e]);backbone.load_weights('data/yamnet.h5')
    x=layers.Concatenate()([e,s]);base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    def dense(expert,bias,name):
        layer=layers.Dense(1,activation='sigmoid',trainable=False,name=name);v=layer(x)
        layer.set_weights([expert.coef_.T.astype(np.float32),(expert.intercept_+bias).astype(np.float32)]);return v
    train=layers.Maximum()([base(x),dense(rhythm,oldreport['selected']['bias'],'reviewed_rhythmic_rail')])
    bellscore=dense(bell,report['short_path_experiment']['expert_bias'],'reviewed_train_bell')
    pair=layers.Concatenate(name='train_then_bell')([train,bellscore]);model=Model(inp,[pair,s])
    @tf.function(input_signature=[tf.TensorSpec([1,15600],tf.float32)])
    def serve(waveform):
        p,s=model(waveform,training=False);return {'a_train_and_bell':p,'b_yamnet_scores':s}
    converter=tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()],model)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    path=root/'bell-lab.tflite';path.write_bytes(converter.convert())
    def interpreter(path,n):
        i=tf.lite.Interpreter(model_path=str(path),num_threads=2);i.allocate_tensors()
        return i,i.get_input_details()[0]['index'],next(o['index'] for o in i.get_output_details() if o['shape'][-1]==n),next(o['index'] for o in i.get_output_details() if o['shape'][-1]==521)
    i,ii,pi,si=interpreter(path,2);old,oi,op,os=interpreter(oldroot/'intermodal-lab.tflite',1)
    rng=np.random.default_rng(69);vectors=[np.zeros(15600,np.float32)]+[(rng.normal(size=15600)*a).astype(np.float32) for a in (1e-5,.001,.01,.1)]
    definitions=[('freight-bell',Path('experiments/bell-external-check/sources/854498/audio.wav'),9,12,True),
        ('heldout-bell',Path('data/clips/fsd50k/dev/127451.wav'),0,4.393375,True),
        ('steam-no-bell',Path('experiments/bell-external-check/sources/752578/audio.wav'),0,30,False),
        ('electronic-crossing',Path('experiments/bell-external-check/sources/737301/audio.wav'),0,30.67075,False)]
    cases=[]
    for name,audio,start,end,expected in definitions:
        w=read_wav16(audio)[round(start*16000):round(end*16000)];records=[]
        for off in range(0,len(w)-15600+1,7680):
            v=w[off:off+15600];vectors.append(v)
            i.set_tensor(ii,v[None].astype(np.float32));i.invoke();p=i.get_tensor(pi).ravel()
            records.append(dict(head=float(p[0]),bell=float(p[1]),level=float(20*np.log10(max(1e-12,np.sqrt(np.mean(v*v)))))))
        cases.append(dict(name=name,expected_bell_event=expected,records=records))
    diff=tagdiff=traindiff=0
    for v in vectors:
        v=v[None].astype(np.float32);i.set_tensor(ii,v);i.invoke();old.set_tensor(oi,v);old.invoke();kp,ks=model(v,training=False)
        p=i.get_tensor(pi);s=i.get_tensor(si)
        assert np.isfinite(p).all() and np.isfinite(s).all()
        diff=max(diff,float(np.max(np.abs(p-kp.numpy()))),float(np.max(np.abs(s-ks.numpy()))))
        tagdiff=max(tagdiff,float(np.max(np.abs(s-old.get_tensor(os)))))
        traindiff=max(traindiff,float(abs(p.ravel()[0]-old.get_tensor(op).ravel()[0])))
    assert max(diff,tagdiff,traindiff)<.001,(diff,tagdiff,traindiff)
    (root/'runtime-cases.json').write_text(json.dumps(cases,indent=2))
    for case in cases:
        with (root/(case['name']+'.csv')).open('w') as f:
            for r in case['records']:f.write(f"{r['head']},{r['bell']},{r['level']}\n")
    result=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),vectors=len(vectors),keras_tflite_max_diff=diff,original_train_max_diff=traindiff,original_tags_max_diff=tagdiff,output_contract='two tensors: [unchanged train probability, dedicated train-bell expert] and original 521 tags',bell_threshold=.869,bell_hold_windows=5,status='laboratory candidate; not installed',source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'expert.pkl',root/'results.json']})
    (root/'export-info.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
