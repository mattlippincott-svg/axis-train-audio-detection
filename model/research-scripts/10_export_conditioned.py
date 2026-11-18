"""Export bounded level conditioning + frozen baseline, without touching model/out."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
from common import OUT, read_wav16


def main():
    import tensorflow as tf
    from tf_keras import Model,layers
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['foundation','channel-robust','engine-anchors','engine-diverse'],default='foundation')
    args=parser.parse_args()
    root=Path(OUT).parent/'experiments'/args.experiment
    export=import_module('04_export');export.OUT=str(root/'baseline' if args.experiment=='foundation' else root)
    baseline=export.build(tf)
    inp=layers.Input(batch_shape=(1,15600),dtype=tf.float32,name='waveform')
    def condition(w):
        rms=tf.sqrt(tf.reduce_mean(tf.square(w),axis=1,keepdims=True)+1e-12)
        peak=tf.reduce_max(tf.abs(w),axis=1,keepdims=True)
        gain=tf.minimum(tf.constant(10**(36/20),tf.float32),tf.constant(10**(-26/20),tf.float32)/rms)
        gain=tf.maximum(1.,gain)
        gain=tf.minimum(gain,.98/tf.maximum(peak,1e-9))
        return w*gain
    conditioned=layers.Lambda(condition,name='bounded_level_conditioning')(inp)
    p,s=baseline(conditioned)
    model=Model(inp,[p,s])
    @tf.function(input_signature=[tf.TensorSpec([1,15600],tf.float32)])
    def serve(waveform):
        p,s=model(waveform)
        return {'a_train_prob':p,'b_yamnet_scores':s}
    converter=tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()],model)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    raw=converter.convert();path=root/'conditioned.tflite';path.write_bytes(raw)
    interpreter=tf.lite.Interpreter(model_path=str(path));interpreter.allocate_tensors()
    ii=interpreter.get_input_details()[0]['index'];outs=interpreter.get_output_details()
    pi=next(o['index'] for o in outs if o['shape'][-1]==1)
    si=next(o['index'] for o in outs if o['shape'][-1]==521)
    vectors=[np.zeros(15600,np.float32)]
    rng=np.random.default_rng(17)
    vectors += [(rng.normal(size=15600)*a).astype(np.float32) for a in (.00001,.001,.01,.1)]
    for probe in (Path(OUT).parent/'experiments'/'foundation'/'probes').glob('*.wav'):
        wave=read_wav16(probe)
        vectors += [wave[:15600]*a for a in (1.,.01)]
    maxdiff=0.
    for v in vectors:
        v=np.asarray(v,np.float32).reshape(1,15600)
        interpreter.set_tensor(ii,v);interpreter.invoke()
        pk,sk=model(v,training=False)
        if not np.isfinite(pk.numpy()).all() or not np.isfinite(sk.numpy()).all():
            raise RuntimeError('Non-finite model outputs')
        maxdiff=max(maxdiff,float(np.max(np.abs(interpreter.get_tensor(pi)-pk.numpy()))),float(np.max(np.abs(interpreter.get_tensor(si)-sk.numpy()))))
    if maxdiff>.001:raise RuntimeError(f'Export parity failed: {maxdiff}')
    report={'bytes':len(raw),'input':[1,15600],'output_shapes':[o['shape'].tolist() for o in outs],
            'keras_tflite_max_abs_diff':maxdiff,'parity_vectors':len(vectors),
            'ops':sorted({op['op_name'] for op in interpreter._get_ops_details()}),
            'conditioning':{'target_dbfs':-26,'max_gain_db':36,'boost_only':True,'peak_ceiling':.98},
            'weights':('frozen inherited baseline; fresh classifier rejected at matched validation false-positive budget'
                       if args.experiment=='foundation' else 'new head trained with capture-channel augmentation and non-horn emphasis'),
            'status':'experimental; requires completed benchmark review and on-device verification'}
    (root/'conditioned-model-info.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
