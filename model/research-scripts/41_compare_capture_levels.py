"""Read-only comparison of quiet TRAIN capture and louder new-condition capture.

Capture 02 never enters fitting here. Same source material and room mean this
is a level-condition development comparison, not unseen-source acceptance.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,sys,argparse
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-printer-paused',action='store_true');args=parser.parse_args()
    import tensorflow as tf
    from tf_keras import Model,layers,models
    root=Path('experiments/printer-paused-comparison' if args.include_printer_paused else 'experiments/capture-level-comparison');root.mkdir(exist_ok=True)
    lab=Path('/lab/work/audio-diagnostics');channel=import_module('13_channel_robust');f=import_module('06_foundation')
    pin=layers.Input((96,64));s,e=yamnet.yamnet(pin,params.Params());core=Model(pin,[s,e]);core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def embed(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        s,e=core(lm,training=False);return tf.concat([e,s],axis=1)
    nets={}
    for name,folder in [('installed','channel-robust'),('source_reviewed','reviewed-train-us'),('paired_trained','reviewed-train-paired'),('measured_path_trained','reviewed-train-measured')]:
        path=Path('experiments')/folder;config=dict(threshold=.869,min_level_dbfs=-65) if name=='installed' else json.loads((path/'selected.json').read_text())
        nets[name]=(models.load_model(path/'head.h5',compile=False),config)
    records=[];quality={}
    capture_ids=['paired-training-01','paired-comparison-02']
    if args.include_printer_paused:capture_ids.append('printer-paused-comparison-03')
    for capture_id in capture_ids:
        folder=lab/capture_id;q=json.loads((folder/'quality-and-alignment.json').read_text());assert q['alignment_accepted']
        quality[capture_id]=q;wave=read_wav16(folder/'capture.wav')
        for seg in q['segments']:
            assert seg['complete'];w=wave[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)]
            windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680].copy();normal,level=channel.condition(windows)
            x=np.concatenate([embed(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
            results={}
            for name,(model,config) in nets.items():
                p=model(x,training=False).numpy().ravel();gated=p.copy();gated[level < config.get('min_level_dbfs',-65)]=0
                score=f.event_score(gated)
                results[name]=dict(score=score,ungated_score=f.event_score(p),threshold=config['threshold'],trigger=bool(score>=config['threshold']),
                                   gate_only_upper_bound=f.event_score((level>=config.get('min_level_dbfs',-65)).astype(float)))
            records.append(dict(capture_id=capture_id,source=str(seg['source_id']),expected_class=seg['expected_class'],rms_dbfs=seg['rms_dbfs'],
                median_window_dbfs=seg['median_window_dbfs'],above_gate_fraction=seg['above_volume_gate_fraction'],above_background_db=seg['above_silence_median_db'],models=results))
    deltas=[]
    for r in records:
        if r['capture_id']!='paired-comparison-02':continue
        old=next(o for o in records if o['capture_id']=='paired-training-01' and o['source']==r['source'])
        deltas.append(dict(source=r['source'],rms_increase_db=r['rms_dbfs']-old['rms_dbfs'],
            quiet_above_background_db=old['above_background_db'],louder_above_background_db=r['above_background_db']))
    report={'records':records,'level_changes':deltas,'quality_summaries':{k:{f:q[f] for f in ('capture_seconds','lag_s','corroborating_anchors','capture_peak_dbfs','clipped_sample_fraction','silence_median_rms_dbfs')} for k,q in quality.items()},
        'scope':'capture 02 remains outside training; source material is shared with training and this is not independent-source acceptance',
        'limits':['head-only evaluation omits live speech suppression','digital level is not calibrated dBA','background may change between takes',
                  'two negative excerpts cannot establish field false-alarm rates','no model or device configuration changed by this script']}
    if args.include_printer_paused:
        report['printer_paused_setup']={'printer':'paused per user','hvac':'off per user','volume_and_positions':'requested unchanged from take 02; not independently verified'}
        report['printer_pause_changes']=[]
        for r in records:
            if r['capture_id']!='printer-paused-comparison-03':continue
            old=next(o for o in records if o['capture_id']=='paired-comparison-02' and o['source']==r['source'])
            report['printer_pause_changes'].append(dict(source=r['source'],rms_change_db=r['rms_dbfs']-old['rms_dbfs'],
                background_separation_change_db=r['above_background_db']-old['above_background_db']))
    (root/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
