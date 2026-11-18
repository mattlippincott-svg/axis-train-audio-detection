"""Prepare deterministic conditioned-model vectors for non-injecting device parity."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import csv
import argparse
import json
from pathlib import Path
import numpy as np
from common import DATA,OUT,MANIFEST,read_wav16,write_wav16


def main():
    import tensorflow as tf
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=['foundation','channel-robust'],default='foundation');args=parser.parse_args()
    foundation=Path(OUT).parent/'experiments'/'foundation'
    root=Path(OUT).parent/'experiments'/args.experiment;dest=root/'device-vectors';dest.mkdir(exist_ok=True)
    i=tf.lite.Interpreter(model_path=str(root/'conditioned.tflite'),num_threads=2);i.allocate_tensors()
    ii=i.get_input_details()[0]['index'];pi=next(o['index'] for o in i.get_output_details() if o['shape'][-1]==1)
    vectors={}
    for p in (foundation/'probes').glob('*.wav'):
        wave=read_wav16(p)
        vectors[p.stem]=(wave[:16000*5]*.063095734).astype(np.float32)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    d=np.load(foundation/'evaluation-predictions.npz');allowed=set(d['clip'][d['split']==2].tolist())
    for tag in ('Truck','Speech','church_bells'):
        match=next((r for c,r in enumerate(rows) if c in allowed and r['label']=='0' and tag in r['labels'].split(',')),None)
        if match:vectors['negative_'+tag]=read_wav16(os.path.join(DATA,match['path']))[:16000*5]
    expected={}
    for name,w in vectors.items():
        p=dest/(name+'.wav');write_wav16(str(p),w);w=read_wav16(p)
        if len(w)<15600:w=np.pad(w,(0,15600-len(w)))
        scores=[]
        for off in range(0,len(w)-15600+1,7680):
            i.set_tensor(ii,w[off:off+15600].reshape(1,15600).astype(np.float32));i.invoke();scores.append(float(i.get_tensor(pi).ravel()[0]))
        expected[name]=scores
    (dest/'expected.json').write_text(json.dumps(expected,indent=2))
    print('Prepared',len(expected),'vectors')


if __name__=='__main__':main()
