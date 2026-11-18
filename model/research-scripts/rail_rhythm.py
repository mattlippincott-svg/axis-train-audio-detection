"""Causal four-second multi-band impact features for offline experiments.

No assumed train speed: correlations cover roughly 50 ms to 1.5 seconds.
No calibrated loudness is inferred from these amplitude-normalized features.
"""
import numpy as np

LAGS=np.unique(np.round(np.geomspace(5,150,20)).astype(int))
SIZE=64000

def rhythm_features(wave,ends):
    wave=np.asarray(wave,np.float32)
    # Each endpoint is exclusive: neither the STFT nor context reads its future.
    result=[]
    freq=np.fft.rfftfreq(512,1/16000)
    bands=[(50,250),(250,1000),(1000,3000),(3000,7500)]
    ends=np.asarray(ends,dtype=int)
    if np.any(ends<1) or np.any(ends>len(wave)):raise ValueError('Endpoint outside available waveform')
    # Compute overlapping spectra once per hop phase, rather than recomputing a
    # four-second STFT at every half-second endpoint. Selected slices are causal.
    padded=np.pad(wave,(SIZE,0));envelopes={}
    for phase in np.unique(ends%160):
        frames=np.lib.stride_tricks.sliding_window_view(padded[int(phase):],512)[::160]
        power=np.abs(np.fft.rfft(frames*np.hanning(512),axis=1))**2
        envelopes[int(phase)]=np.stack([np.sqrt(power[:,(freq>=lo)&(freq<hi)].sum(1)+1e-20) for lo,hi in bands])
    context_frames=1+(SIZE-512)//160
    for end in ends:
        end=int(end)
        features=[]
        start=end//160
        for band in range(len(bands)):
            env=envelopes[end%160][band,start:start+context_frames]
            env=env/(np.sqrt(np.mean(env**2))+1e-10)
            onset=np.maximum(0,np.diff(env,prepend=env[0]))
            for signal in (env,onset):
                z=signal-signal.mean();den=np.dot(z,z)+1e-9
                ac=np.array([np.dot(z[:-lag],z[lag:])/den for lag in LAGS])
                features.extend(ac)
            features.extend([np.std(env),np.percentile(env,95),np.max(env),
                np.mean(onset),np.std(onset),np.percentile(onset,95),
                np.mean(onset>.25),np.mean(onset>.75)])
        result.append(features)
    return np.nan_to_num(np.array(result,np.float32))
