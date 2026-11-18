"""Prepare digital parity audio only; analyze.cgi does not inject events."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import read_wav16,write_wav16

def main():
    dest=Path('/lab/work/bell-lab-063')
    parts=[np.zeros(16000,np.float32)]
    for path,start,end in [('data/clips/fsd50k/dev/127451.wav',0,4.393375),
        ('experiments/bell-external-check/sources/854498/audio.wav',9,12),
        ('experiments/bell-external-check/sources/737301/audio.wav',0,4),
        ('/lab/work/audio-diagnostics/ns-061-armed/capture.wav',68,76)]:
        parts.extend([read_wav16(path)[round(start*16000):round(end*16000)],np.zeros(16000,np.float32)])
    path=dest/'verification.wav';write_wav16(str(path),np.concatenate(parts));w=read_wav16(path)
    i=tf.lite.Interpreter(model_path='experiments/bell-candidate-round-two/bell-lab.tflite',num_threads=2);i.allocate_tensors()
    inp=i.get_input_details()[0]['index'];outputs=i.get_output_details();prob=next(o['index'] for o in outputs if o['shape'][-1]==2)
    expected=[]
    for start in range(0,len(w)-15600+1,7680):
        i.set_tensor(inp,w[start:start+15600][None].astype(np.float32));i.invoke();p=i.get_tensor(prob).ravel()
        expected.append(dict(t=start/16000,prob=float(p[0]),train_bell=float(p[1])))
    (dest/'verification-expected.json').write_text(json.dumps(expected,indent=2));print(len(expected),'windows')

if __name__=='__main__':main()
