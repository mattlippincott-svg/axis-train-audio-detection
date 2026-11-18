"""Estimate a bounded coarse capture-path magnitude profile from TRAINING pairs.

This is not a calibrated microphone response. It includes speakers, room,
microphone, device gain and AAC encoding. Never uses old comparison captures.
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import welch,firwin2
from common import read_wav16

def main():
    lab=Path('/lab');folder=lab/'work/audio-diagnostics/paired-training-01';root=Path('experiments/measured-capture-path');root.mkdir(exist_ok=True)
    quality=json.loads((folder/'quality-and-alignment.json').read_text());assert quality['alignment_accepted']
    source=read_wav16(lab/'outputs/train-microphone-pairing-playback.wav');capture=read_wav16(folder/'capture.wav')
    edges=np.array([80,125,250,500,1000,2000,4000,7500]);measurements=[]
    for seg in quality['segments']:
        x=source[round(seg['playback_start_s']*16000):round(seg['playback_end_s']*16000)]
        y=capture[round(seg['capture_start_s']*16000):round(seg['capture_end_s']*16000)]
        freq,px=welch(x,16000,nperseg=1024);_,py=welch(y,16000,nperseg=1024)
        bands=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=(freq>=lo)&(freq<hi);bands.append(float(10*np.log10((py[m].sum()+1e-20)/(px[m].sum()+1e-20))))
        measurements.append(dict(source=seg['source_id'],capture_to_source_power_db=bands))
    values=np.array([r['capture_to_source_power_db'] for r in measurements]);median=np.median(values,axis=0)
    reference=float(median[4]);relative=np.clip(median-reference,-18,18)
    centers=np.sqrt(edges[:-1]*edges[1:]);freqs=np.r_[0,centers,8000];gains=10**(np.r_[relative[0],relative,relative[-1]]/20)
    taps=firwin2(129,freqs/8000,gains).astype(np.float32)
    np.save(root/'magnitude-fir.npy',taps)
    # Known digital-silence portion of this training playback contains room sound.
    lo=round((quality['lag_s']+1)*16000);ambient=capture[lo:lo+3*16000]
    import soundfile as sf
    sf.write(root/'room-background.wav',ambient,16000,subtype='PCM_16')
    report={'band_edges_hz':edges.tolist(),'measurements':measurements,'median_capture_to_source_power_db':median.tolist(),
        'relative_shape_db':relative.tolist(),'reference_band_hz':[1000,2000],'reference_gain_db':reference,
        'fir_taps':len(taps),'relative_gain_bounds_db':[-18,18],
        'scope':'coarse magnitude augmentation for this TRAINING playback path only',
        'limits':['not a calibrated mic response','five short source excerpts in one room','room resonances and source spectra affect estimates',
                  'phase response and changing outdoor conditions are not reproduced','old Colorado and NS recordings unused']}
    (root/'profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
