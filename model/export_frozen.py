"""Re-export shipped parameters; does not download data or retrain."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tf_keras import Model,layers,models
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'model/yamnet'))
import params,yamnet,features

def main():
    release=ROOT/'model/releases/0.6.3';manifest=json.loads((release/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert hashlib.sha256((release/name).read_bytes()).hexdigest()==h,name
    weights=np.load(release/'experts.npz',allow_pickle=False)
    inp=layers.Input(batch_shape=(1,15600),dtype=tf.float32,name='waveform')
    def condition(w):
        rms=tf.sqrt(tf.reduce_mean(w*w,axis=1,keepdims=True)+1e-12);peak=tf.reduce_max(tf.abs(w),axis=1,keepdims=True)
        gain=tf.maximum(1.,tf.minimum(tf.constant(10**1.8,tf.float32),tf.constant(10**(-26/20),tf.float32)/rms));gain=tf.minimum(gain,.98/tf.maximum(peak,1e-9));return w*gain
    w=layers.Lambda(condition)(inp);w=layers.Lambda(lambda v:tf.reshape(v,[15600]))(w)
    par=params.Params(tflite_compatible=True);_,patches=features.waveform_to_log_mel_spectrogram_patches(w,par)
    s,e=yamnet.yamnet(patches,par);backbone=Model(inp,[s,e]);backbone.load_weights(str(release/'yamnet.h5'))
    x=layers.Concatenate()([e,s]);base=models.load_model(str(release/'train-head.h5'),compile=False)
    def expert(name):
        layer=layers.Dense(1,activation='sigmoid',trainable=False,name=name);v=layer(x)
        layer.set_weights([weights[name+'_coef'],weights[name+'_bias']]);return v
    train=layers.Maximum()([base(x),expert('rhythm')]);bell=expert('bell')
    model=Model(inp,[layers.Concatenate()([train,bell]),s])
    @tf.function(input_signature=[tf.TensorSpec([1,15600],tf.float32)])
    def serve(waveform):
        p,s=model(waveform,training=False);return {'a_train_and_bell':p,'b_yamnet_scores':s}
    converter=tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()],model)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    dest=ROOT/'build/model';dest.mkdir(parents=True,exist_ok=True);path=dest/'traindet.tflite';path.write_bytes(converter.convert())
    def interpreter(p):
        i=tf.lite.Interpreter(model_path=str(p),num_threads=2);i.allocate_tensors()
        return i,i.get_input_details()[0]['index'],{int(o['shape'][-1]):o['index'] for o in i.get_output_details()}
    new,ni,no=interpreter(path);old,oi,oo=interpreter(release/'traindet.tflite')
    rng=np.random.default_rng(20261118);difference=0.
    vectors=[np.zeros((1,15600),np.float32)]+[(rng.normal(size=(1,15600))*a).astype(np.float32) for a in (1e-5,.001,.01,.1) for _ in range(4)]
    for v in vectors:
        new.set_tensor(ni,v);new.invoke();old.set_tensor(oi,v);old.invoke()
        for n in (2,521):
            actual=new.get_tensor(no[n]);assert np.isfinite(actual).all()
            difference=max(difference,float(np.max(np.abs(actual-old.get_tensor(oo[n])))))
    assert difference<.001,difference
    report=dict(vectors=len(vectors),maximum_difference=difference,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Frozen-parameter export parity, not model training or field validation')
    (dest/'parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
