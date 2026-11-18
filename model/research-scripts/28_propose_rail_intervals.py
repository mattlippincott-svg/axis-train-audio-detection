"""Rank short review intervals on TRAINING CANDIDATES only.

These are model-assisted proposals, not labels. Reserved sources are never
screened here. Window scores are retained to expose the ranking's limitations.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json,sys,csv
from pathlib import Path
from importlib import import_module
import numpy as np
from common import DATA,read_wav16
sys.path.insert(0,str(Path(__file__).parent.parent/'yamnet'))
import params,yamnet

ROOT=Path('experiments/rail-corpus-v2')

def main():
    import tensorflow as tf
    from tf_keras import Model,layers
    channel=import_module('13_channel_robust')
    catalog=json.loads((ROOT/'catalog-audited.json').read_text())
    pin=layers.Input((96,64));scores,embedding=yamnet.yamnet(pin,params.Params());core=Model(pin,scores)
    core.load_weights(str(Path(DATA)/'yamnet.h5'))
    mel=tf.signal.linear_to_mel_weight_matrix(64,257,16000,125.,7500.)
    @tf.function(input_signature=[tf.TensorSpec([None,15600],tf.float32)])
    def tags(w):
        lm=tf.math.log(tf.matmul(tf.abs(tf.signal.stft(w,400,160,512)),mel)+.001)
        return core(lm,training=False)
    proposals=[];screened=[]
    for r in catalog['sources']:
        if not r['downloaded'] or r['role']!='train_review':continue
        sid=r['id'];folder=ROOT/'sources'/str(sid);w=read_wav16(folder/'audio.wav')
        starts=np.arange(0,len(w)-15600+1,7680)
        windows=np.stack([w[s:s+15600] for s in starts]);normal,level=channel.condition(windows)
        s=np.concatenate([tags(normal[i:i+32]).numpy() for i in range(0,len(normal),32)])
        horn=s[:,[324,325]].max(1);bell=s[:,[195,196,197]].max(1);speech=s[:,:6].max(1);music=s[:,132]
        wheel=s[:,326];squeal=s[:,327]
        candidates=[]
        for start in np.arange(2,max(2,len(w)/16000-10),5):
            m=(starts/16000>=start)&((starts+15600)/16000<=start+10)
            if not m.any():continue
            def mean(v):return float(np.mean(v[m]))
            info=dict(source_id=sid,creator=r['creator'],family=r['family'],role=r['role'],start_s=float(start),end_s=float(start+10),
                rail_wagon_mean=mean(wheel),squeal_mean=mean(squeal),horn_peak=float(horn[m].max()),bell_peak=float(bell[m].max()),
                speech_peak=float(speech[m].max()),music_peak=float(music[m].max()),mean_level_dbfs=mean(level),
                review_status='MODEL_PROPOSAL_NOT_GROUND_TRUTH')
            info['ranking_score']=mean(wheel)-2*mean(horn)-2*mean(bell)-mean(speech)-mean(music)
            candidates.append(info)
        chosen=[]
        for c in sorted(candidates,key=lambda c:c['ranking_score'],reverse=True):
            if all(c['end_s']<=b['start_s'] or c['start_s']>=b['end_s'] for b in chosen):chosen.append(c)
            if len(chosen)==2:break
        proposals.extend(chosen);screened.append(sid);print('proposed',sid,len(chosen),flush=True)
    reserved={r['id'] for r in catalog['sources'] if r['role']=='reserved_review'}
    assert not reserved.intersection(screened)
    report={'screened_source_ids':screened,'untouched_reserved_source_ids':sorted(reserved),'proposals':proposals,
        'limits':['model scores are not listening verification','wagon score is not a dedicated click-clack class',
                  'ranking can miss good audio because the inherited model has known errors','no positive label approved automatically']}
    (ROOT/'interval-proposals.json').write_text(json.dumps(report,indent=2))
    with open(ROOT/'interval-proposals.csv','w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(proposals[0]));writer.writeheader();writer.writerows(proposals)
    print('total proposed seconds',sum(c['end_s']-c['start_s'] for c in proposals),flush=True)

if __name__=='__main__':main()
