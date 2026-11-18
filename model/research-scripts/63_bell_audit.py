"""Audit generic bell evidence without fitting or scoring the test partition."""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np

def main():
    rows=list(csv.DictReader(open('data/manifest.csv',encoding='utf-8')))
    meta=json.loads(Path('data/meta/dev_clips_info_FSD50K.json').read_text())
    d=np.load('experiments/channel-robust/features.npz')
    # The cached feature layout is frozen YAMNet embedding followed by 521 tags.
    clip=d['clip'];version=d['version']
    selected=np.isin(clip,[c for c,r in enumerate(rows) if r['split'] in ('train','val') and 'bell' in r['labels'].lower()])&(version==0)
    x=d['x'];assert x.shape[1]==1545
    bell_scores=np.max(x[selected,1024+195:1024+198],axis=1)
    clip=clip[selected]
    records=[]
    for c,r in enumerate(rows):
        if r['split'] not in ('train','val') or 'bell' not in r['labels'].lower():continue
        sid=r['id'].split('/')[-1].removesuffix('.wav')
        m=meta.get(sid,{})
        ix=np.flatnonzero(clip==c)
        score=bell_scores[ix]
        records.append(dict(row=c,id=r['id'],split=r['split'],train_label=int(r['label']),labels=r['labels'],
            title=m.get('title'),description=m.get('description'),creator=m.get('uploader'),
            bell_peak=float(score.max()) if len(score) else None,
            bell_windows_at_least_half=int((score>=.5).sum()),windows=len(ix),
            qualification='clip tags and source metadata only; locomotive bell intervals not verified'))
    positive=[r for r in records if r['train_label']]
    report=dict(status='diagnostic only; no fitting, threshold selection, test scoring or deployment',
        generic_bell_classes=[195,196,197],
        counts=dict(Counter(f"{r['split']}:train_label={r['train_label']}" for r in records)),
        train_tagged_bell_sources=positive,
        negative_sources_with_bell_peak_at_least_half=sum(r['train_label']==0 and (r['bell_peak'] or 0)>=.5 for r in records),
        records=records,
        conclusion='Six of seven train-split bell-tagged positives are subway door alerts. Only source 28777 has metadata describing a passing train ringing a bell; source 127451 is validation and must remain outside fitting. Generic bell scores must not become an unconditional train trigger.')
    dest=Path('experiments/bell-audit');dest.mkdir(exist_ok=True)
    (dest/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
