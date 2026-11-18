"""Acquire public previews for development checking, never fit or auto-label."""
import csv,hashlib,json
from pathlib import Path
from importlib import import_module
from common import decode_bytes,write_wav16

def main():
    root=Path('experiments/bell-external-check');root.mkdir(exist_ok=True)
    collector=import_module('26_curate_rail');collector.ROOT=root
    sources=[(854498,('serøutōnin--deprivəd','development_review','Freight horn and bell; overlap must be reviewed')),
        (752578,('lakewoodsound','development_review','Steam locomotive bell; auxiliary train type, not modern diesel validation')),
        (737301,('WeenyBeany','crossing_challenge','Processed electronic crossing bell; not locomotive-bell positive or generic non-train negative'))]
    # Preserve exactly the candidate being checked before acquisition.
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('experiments/bell-candidate/expert.pkl'),Path('experiments/bell-candidate/results.json')]}
    (root/'frozen-candidate.json').write_text(json.dumps(frozen,indent=2))
    rows=list(csv.DictReader(open('data/manifest.csv',encoding='utf-8')))
    meta={}
    for part in ('dev','eval'):meta.update(json.loads(Path(f'data/meta/{part}_clips_info_FSD50K.json').read_text()))
    results=[]
    for entry in sources:
        r=collector.fetch(entry)
        related=[(c,row) for c,row in enumerate(rows) if meta.get(row['id'].split('/')[-1],{}).get('uploader','').casefold()==r['creator'].casefold()]
        r['legacy_creator_rows']={s:[c for c,row in related if row['split']==s] for s in ('train','val','test')}
        r['family']='creator:'+r['creator'].casefold()
        if r.get('downloaded'):
            folder=root/'sources'/str(r['id']);wave=decode_bytes((folder/'preview.mp3').read_bytes())
            write_wav16(str(folder/'audio.wav'),wave)
            r['decoded_duration_s']=len(wave)/16000;r['audio_sha256']=hashlib.sha256((folder/'audio.wav').read_bytes()).hexdigest()
        (root/'sources'/str(r['id'])/'metadata.json').write_text(json.dumps(r,indent=2))
        results.append(r);print(r['id'],r.get('downloaded'),r.get('error'),r['legacy_creator_rows'],flush=True)
        if r.get('http_status') in (403,429):break
    (root/'catalog.json').write_text(json.dumps(dict(sources=results,policy='Development-only, not pristine acceptance; metadata is not an interval label. No fitting or deployment. No acquisition of previously reserved source families.'),indent=2))

if __name__=='__main__':main()
