"""Causal recurrence evidence from two observations separated in time."""
import numpy as np

def recurrence(values,min_lag=4,history=12):
    if min_lag<1 or history<=min_lag:raise ValueError('Invalid recurrence context')
    values=np.asarray(values,dtype=np.float64);out=np.full(len(values),-np.inf)
    for end in range(len(values)):
        start=max(0,end-history+1)
        for later in range(start+min_lag,end+1):
            earlier=np.max(values[start:later-min_lag+1])
            out[end]=max(out[end],min(earlier,values[later]))
    return out
