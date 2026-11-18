"""Fixed, model-independent review excerpts with unchanged source gain."""
import hashlib,json
from pathlib import Path
from common import read_wav16,write_wav16

def main():
    root=Path('experiments/bell-external-check');out=Path('/lab/outputs')
    catalog=json.loads((root/'catalog.json').read_text());manifest=[]
    for sid,end,name in [(854498,24.5393125,'bell-review-3-freight'),(752578,30,'bell-review-4-steam')]:
        r=next(r for r in catalog['sources'] if r['id']==sid)
        path=root/'sources'/str(sid)/'audio.wav';assert hashlib.sha256(path.read_bytes()).hexdigest()==r['audio_sha256']
        w=read_wav16(path)[:round(end*16000)];dest=out/(name+'.wav');write_wav16(str(dest),w)
        manifest.append(dict(source_id=sid,creator=r['creator'],source_url=r['source_url'],licenses=r['license_urls'],original_start_s=0,original_end_s=len(w)/16000,file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),processing='Public HQ compressed preview decoded to mono 16k PCM16; fixed initial excerpt; no added gain or normalization',label_status='UNREVIEWED',selection='Fixed initial excerpt, not selected using model scores'))
    (out/'bell-review-round-two.json').write_text(json.dumps(dict(clips=manifest),indent=2))
    report='''# External bell development check

The candidate was frozen before these sources were acquired. No additional fitting or threshold changes occurred.

| Source | Current sustained head | Candidate | New expert alone |
|---|---:|---:|---:|
| Freight 854498 | 0.801 | 0.801 | 0.579 |
| Steam departure 752578 | 0.981 | 0.999 | 0.999 |
| Electronic crossing 737301 | 0.002 | 0.006 | 0.006 |

Threshold: 0.869. These are whole-recording model-score diagnostics without the separate horn shortcut. They do not prove bell-only recognition or live event behavior. Source descriptions are provisional labels; human interval review is still needed. Candidate remains experimental and was not installed. The electronic crossing example did not trigger this sustained path; this single example does not establish crossing rejection generally.

Freight source: [serøutōnin--deprivəd, 854498](https://freesound.org/people/ser%C3%B8ut%C5%8Dnin--depriv%C9%99d/sounds/854498/), CC0. Review clip covers the complete 24.54-second preview.

Steam source: [lakewoodsound, Shay #5: Departure, 752578](https://freesound.org/people/lakewoodsound/sounds/752578/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Review clip is a mono 16 kHz excerpt of seconds 0–30, without added gain. This steam-locomotive example is auxiliary, not a modern freight substitute. The source page has inconsistent publication and recording dates; exact recording date is not independently verified.

Crossing challenge: [WeenyBeany, 737301](https://freesound.org/people/WeenyBeany/sounds/737301/), CC BY 4.0. The creator describes a processed electronic crossing bell. This creator also supplied an existing wheel-on-rail development reference; no pristine held-out-family claim is made.

None of these creators have rows in the inherited manifest. This does not establish independence from original YAMNet pretraining. Copied results and catalog preserve source hashes and provenance.
'''
    (out/'external-bell-development-check.md').write_text(report,encoding='utf-8')
    for name in ('catalog.json','diagnostic.json','frozen-candidate.json'):
        (out/('external-bell-'+name)).write_bytes((root/name).read_bytes())
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
