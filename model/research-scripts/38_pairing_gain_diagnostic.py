"""Compare existing models on paired capture gain variants; no fitting or deployment.

Gain is applied AFTER recording. This preserves recorded SNR and is not a
claim about what a louder speaker playback would capture acoustically.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import json
from pathlib import Path
from importlib import import_module
import numpy as np

def main():
    from tf_keras import models
    root=Path('experiments');d=dict(np.load(root/'paired-training-01'/'features.npz'));f=import_module('06_foundation')
    results=[]
    for name,folder in [('installed_baseline','channel-robust'),('reviewed_source_head','reviewed-train-us'),('paired_trained_head','reviewed-train-paired')]:
        model=models.load_model(root/folder/'head.h5',compile=False)
        config=dict(threshold=.869,min_level_dbfs=-65) if name=='installed_baseline' else json.loads((root/folder/'selected.json').read_text())
        gate=config.get('min_level_dbfs',-65);threshold=config['threshold']
        p=model(d['x'],training=False).numpy().ravel()
        for sid in np.unique(d['source']):
            for v,gain in enumerate((0,6,12)):
                m=(d['source']==sid)&(d['version']==v);q=p[m].copy();q[d['level'][m]<gate]=0
                score=f.event_score(q)
                results.append(dict(model=name,source=sid,label=int(d['label'][m][0]),digital_gain_db=gain,
                    sustained_score=score,threshold=threshold,min_level_dbfs=gate,trigger=bool(score>=threshold),
                    ungated_sustained_score=f.event_score(p[m])))
    report={'results':results,'limits':['post-recording gain is not louder speaker playback','paired model fitted these microphone intervals',
              'source-head models fitted the corresponding source recordings','two non-train training examples cannot establish false-alarm performance']}
    (root/'paired-training-01'/'gain-diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
