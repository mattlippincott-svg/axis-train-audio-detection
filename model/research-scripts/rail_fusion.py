"""Experimental score policy shared by the offline rail-sound experiments."""
import numpy as np

def sigmoid(v):
    return 1/(1+np.exp(-np.clip(v,-40,40)))

def rail_scores(base,clatter_logit,squeal_logit,tags,bias,rolling_context=False):
    clatter=sigmoid(clatter_logit+bias)
    squeal=sigmoid(squeal_logit+bias)
    if rolling_context:
        clatter=clatter*np.clip(tags[:,326]/.2,0,1)
    # Only distinct targets qualify: clatter, train horn/whistle or bells.
    # The squeal, engine and generic alarm categories never qualify themselves.
    other=np.maximum(clatter,tags[:,[324,325,195,196,197]].max(1))
    support=(other>=.35)*(.15*squeal*other)
    return np.clip(np.maximum(base,clatter)+support,0,1),clatter,support
