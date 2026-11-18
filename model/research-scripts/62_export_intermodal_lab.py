"""Export experimental intermodal head, retaining original YAMNet sound tags."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,pickle,sys,hashlib
from pathlib import Path
import numpy as np
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet,features

def main():
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/wheel-rail-intermodal-gates');selection=json.loads((root/'results.json').read_text())['selected'];assert selection['support']==0
    expert=pickle.load(open(root/'experts.pkl','rb'))['rhythm']
    inp=layers.Input(batch_shape=(1,15600),dtype=tf.float32,name='waveform')
    def condition(w):
        rms=tf.sqrt(tf.reduce_mean(w*w,axis=1,keepdims=True)+1e-12);peak=tf.reduce_max(tf.abs(w),axis=1,keepdims=True)
        gain=tf.maximum(1.,tf.minimum(tf.constant(10**1.8,tf.float32),tf.constant(10**(-26/20),tf.float32)/rms));gain=tf.minimum(gain,.98/tf.maximum(peak,1e-9));return w*gain
    w=layers.Lambda(condition)(inp);w=layers.Lambda(lambda v:tf.reshape(v,[15600]))(w)
    par=params.Params(tflite_compatible=True);_,patches=features.waveform_to_log_mel_spectrogram_patches(w,par);s,e=yamnet.yamnet(patches,par)
    backbone=Model(inp,[s,e]);backbone.load_weights(str(Path(DATA)/'yamnet.h5'))
    x=layers.Concatenate()([e,s]);base=models.load_model('experiments/channel-robust/head.h5',compile=False)
    rhythm=layers.Dense(1,activation='sigmoid',trainable=False,name='reviewed_rhythmic_rail');c=rhythm(x);rhythm.set_weights([expert.coef_.T.astype(np.float32),(expert.intercept_+selection['bias']).astype(np.float32)])
    p=layers.Maximum()([base(x),c]);model=Model(inp,[p,s])
    @tf.function(input_signature=[tf.TensorSpec([1,15600],tf.float32)])
    def serve(waveform):
        p,s=model(waveform,training=False);return {'a_train_prob':p,'b_yamnet_scores':s}
    converter=tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()],model);converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    data=converter.convert();path=root/'intermodal-lab.tflite';path.write_bytes(data)
    def interpreter(path):
        i=tf.lite.Interpreter(model_path=str(path),num_threads=2);i.allocate_tensors();outs=i.get_output_details();return i,i.get_input_details()[0]['index'],next(o['index'] for o in outs if o['shape'][-1]==1),next(o['index'] for o in outs if o['shape'][-1]==521)
    i,ii,pi,si=interpreter(path);old,oi,op,os=interpreter(Path('experiments/channel-robust/conditioned.tflite'))
    rng=np.random.default_rng(62);vectors=[np.zeros(15600,np.float32)]+[(rng.normal(size=15600)*a).astype(np.float32) for a in (1e-5,.001,.01,.1)]
    for path in [Path('experiments/intermodal-reference/excerpt-35-85.wav'),Path('/lab/work/audio-diagnostics/fresh-wheel-rail-eval-02/capture.wav')]:
        wave=read_wav16(path)
        for start in np.linspace(0,len(wave)-15600,8).astype(int):vectors.append(wave[start:start+15600])
    diff=tagdiff=0.;expected=[]
    for v in vectors:
        v=v.reshape(1,15600).astype(np.float32);i.set_tensor(ii,v);i.invoke();old.set_tensor(oi,v);old.invoke();pk,sk=model(v,training=False)
        assert np.isfinite(pk.numpy()).all() and np.isfinite(sk.numpy()).all()
        diff=max(diff,float(np.max(np.abs(i.get_tensor(pi)-pk.numpy()))),float(np.max(np.abs(i.get_tensor(si)-sk.numpy()))));tagdiff=max(tagdiff,float(np.max(np.abs(i.get_tensor(si)-old.get_tensor(os)))))
        expected.append(dict(prob=float(i.get_tensor(pi).ravel()[0]),scores=i.get_tensor(si).ravel().tolist()))
    assert diff<.001 and tagdiff<.001,(diff,tagdiff)
    np.save(root/'lab-parity-vectors.npy',np.stack(vectors));(root/'lab-parity-expected.json').write_text(json.dumps(expected))
    report=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),keras_tflite_max_diff=diff,original_tags_max_diff=tagdiff,vectors=len(vectors),ops=sorted({op['op_name'] for op in i._get_ops_details()}),status='experimental laboratory package; user explicitly requested installation',conditioning='unchanged: boost-only target -26 dBFS, max36 dB, peak .98')
    (root/'lab-export-info.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
