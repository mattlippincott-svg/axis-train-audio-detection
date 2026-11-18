"""Build causal feature contexts without borrowing a recording's own future."""
import numpy as np

def context_indices(size,groups,negative_pool,context=8,continuous=True):
    result=np.zeros((size,context),np.int32)
    for count,(family,ids) in enumerate(groups):
        ids=np.asarray(ids,dtype=np.int32)
        if not len(ids):raise ValueError('Empty target sequence')
        if continuous:
            options=[p for p in negative_pool if p[0]!=family and len(p[1])]
            if not options:raise ValueError('No negative prefix from a different recording family')
            prefix=np.asarray(options[count%len(options)][1],dtype=np.int32)[-(context-1):]
            prefix=np.pad(prefix,(max(0,context-1-len(prefix)),0),mode='edge')
        else:prefix=np.full(context-1,ids[0],np.int32)
        joined=np.r_[prefix,ids]
        for j,index in enumerate(ids):result[index]=joined[j:j+context]
    return result
