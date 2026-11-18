"""Export the independent adapted acoustic fork and verify original tag parity."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,sys
from pathlib import Path
import numpy as np
from common import DATA,OUT,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet,features


def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path(OUT).parent/'experiments'/'upper-finetune';foundation=Path(OUT).parent/'experiments'/'foundation'
    par=params.Params(tflite_compatible=True)
    inp=layers.Input(batch_shape=(1,15600),dtype=tf.float32,name='waveform')
    def condition(w):
        rms=tf.sqrt(tf.reduce_mean(w*w,axis=1,keepdims=True)+1e-12)
        peak=tf.reduce_max(tf.abs(w),axis=1,keepdims=True)
        gain=tf.maximum(1.,tf.minimum(tf.constant(10**1.8,tf.float32),tf.constant(10**(-26/20),tf.float32)/rms))
        gain=tf.minimum(gain,.98/tf.maximum(peak,1e-9))
        return w*gain
    w=layers.Lambda(condition,name='bounded_conditioning')(inp)
    w=layers.Lambda(lambda v:tf.reshape(v,[15600]))(w)
    _,patches=features.waveform_to_log_mel_spectrogram_patches(w,par)
    s,e=yamnet.yamnet(patches,par)
    original=Model(inp,[s,e],name='original_sound_tags');original.load_weights(os.path.join(DATA,'yamnet.h5'))
    lower=original.get_layer('layer12/pointwise_conv/relu').output
    adapted=models.load_model(root/'adapted-upper.h5',compile=False)
    head=models.load_model(root/'head.h5',compile=False)
    p=head(layers.Concatenate()([adapted(lower),s]))
    model=Model(inp,[p,s])
    @tf.function(input_signature=[tf.TensorSpec([1,15600],tf.float32)])
    def serve(waveform):
        p,s=model(waveform,training=False);return {'a_train_prob':p,'b_yamnet_scores':s}
    converter=tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()],model)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    data=converter.convert();(root/'conditioned.tflite').write_bytes(data)
    def interpreter(path):
        i=tf.lite.Interpreter(model_path=str(path),num_threads=2);i.allocate_tensors();oo=i.get_output_details()
        return i,i.get_input_details()[0]['index'],next(o['index'] for o in oo if o['shape'][-1]==1),next(o['index'] for o in oo if o['shape'][-1]==521)
    i,ii,pi,si=interpreter(root/'conditioned.tflite')
    old,oi,_,oscores=interpreter(Path(OUT).parent/'experiments'/'channel-robust'/'conditioned.tflite')
    rng=np.random.default_rng(19)
    vectors=[np.zeros(15600,np.float32)]+[(rng.normal(size=15600)*a).astype(np.float32) for a in (1e-5,.001,.01,.1)]
    for probe in (foundation/'probes').glob('*.wav'):
        v=read_wav16(probe)[:15600];vectors += [v,v*.01]
    diff=tagdiff=0.
    for v in vectors:
        v=v.astype(np.float32).reshape(1,15600);i.set_tensor(ii,v);i.invoke();old.set_tensor(oi,v);old.invoke()
        pk,sk=model(v,training=False)
        if not np.isfinite(pk.numpy()).all() or not np.isfinite(sk.numpy()).all():raise RuntimeError('Non-finite output')
        diff=max(diff,float(np.max(np.abs(i.get_tensor(pi)-pk.numpy()))),float(np.max(np.abs(i.get_tensor(si)-sk.numpy()))))
        tagdiff=max(tagdiff,float(np.max(np.abs(i.get_tensor(si)-old.get_tensor(oscores)))))
    if diff>.001 or tagdiff>.001:raise RuntimeError(f'Parity failed: {diff}, original tags {tagdiff}')
    report={'bytes':len(data),'input':[1,15600],'keras_tflite_max_abs_diff':diff,
        'original_tag_max_abs_difference':tagdiff,'vectors':len(vectors),
        'ops':sorted({op['op_name'] for op in i._get_ops_details()}),
        'architecture':'shared frozen lower YAMNet; frozen original tags; independently fine-tuned upper train branch',
        'deployment_status':'not deployed; validation and on-device resource checks required'}
    (root/'model-info.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
