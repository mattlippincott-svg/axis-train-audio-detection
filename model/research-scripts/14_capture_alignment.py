"""Attempt source/capture alignment; reject weak matches rather than assume timing."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfilt,correlate
from common import OUT,read_wav16


def main():
    root=Path(OUT).parent/'experiments'
    source=read_wav16(root/'foundation'/'probes'/'sustained-brakes.wav')
    captured=read_wav16('/lab/work/audio-diagnostics/tim-kahn-535384/capture.wav')
    filt=butter(3,[250,3500],btype='bandpass',fs=16000,output='sos')
    a=sosfilt(filt,source)
    b=sosfilt(filt,captured[:16000*25])
    size=16000*4
    # Select the loudest complete four-second capture block, independently of
    # correlation, to avoid a quiet-tail match being mistaken for alignment.
    energies=np.convolve(b*b,np.ones(1),mode='same')
    cs=np.r_[0.,np.cumsum(energies)]
    start=int(np.argmax(cs[size:]-cs[:-size]))
    template=b[start:start+size]
    numerator=correlate(a,template,mode='valid',method='fft')
    ca=np.r_[0.,np.cumsum(a*a)]
    denominator=np.sqrt(np.maximum(ca[size:]-ca[:-size],1e-20)*np.sum(template*template))
    ncc=numerator/denominator
    offset=int(np.argmax(np.abs(ncc)))
    match=float(ncc[offset])
    anchors=[]
    for st in (0,16000*4,16000*8,16000*12):
        t=b[st:st+size]
        if len(t)!=size:continue
        cor=correlate(a,t,mode='valid',method='fft')/np.sqrt(np.maximum(ca[size:]-ca[:-size],1e-20)*max(1e-20,np.sum(t*t)))
        j=int(np.argmax(np.abs(cor)))
        anchors.append({'capture_start_s':st/16000,'source_start_s':j/16000,'correlation':float(cor[j]),'lag_s':(j-st)/16000})
    lag=(offset-start)/16000
    corroborating=sum(abs(x['correlation'])>=.2 and abs(x['lag_s']-lag)<=.1 for x in anchors)
    result={'capture_block_start_seconds':start/16000,'source_block_start_seconds_relative_to_2m40s':offset/16000,
            'normalized_waveform_correlation':match,'anchors':anchors,'corroborating_anchors':corroborating,
            'alignment_accepted':abs(match)>=.3 and corroborating>=2,
            'limitations':'Acoustic paths, playback timing and clock drift can defeat waveform correlation. A rejected match provides no reliable alignment or frequency-response estimate.'}
    if result['alignment_accepted']:
        bands={}
        for lo,hi in [(30,125),(125,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,7500)]:
            f=butter(3,[lo,hi],btype='bandpass',fs=16000,output='sos')
            sr=sosfilt(f,source[offset:offset+size]);cr=sosfilt(f,captured[start:start+size])
            bands[f'{lo}-{hi}Hz']=float(10*np.log10((np.mean(cr*cr)+1e-20)/(np.mean(sr*sr)+1e-20)))
        result['capture_to_source_digital_power_difference_db']=bands
        result['band_note']='Includes unknown recording/playback gain; not a calibrated microphone transfer function.'
    (root/'channel-robust'/'capture-alignment.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
