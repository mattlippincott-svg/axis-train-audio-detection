"""Find already-local TRAIN candidates from metadata, never infer audio labels."""
import csv,json,re,html
from pathlib import Path
from common import DATA,MANIFEST

def main():
    root=Path('experiments/local-rail-candidates');root.mkdir(exist_ok=True)
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    meta=json.loads((Path(DATA)/'meta/dev_clips_info_FSD50K.json').read_text())
    cat=json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text())
    excluded=set(cat['future_training_excluded_legacy_rows'])
    excluded.update(json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())['future_training_excluded_legacy_rows'])
    reserved={r['creator'].casefold() for r in cat['sources'] if r['role']=='reserved_review'}
    candidates=[]
    for index,r in enumerate(rows):
        if r['split']!='train' or r['label']!='1' or r['source']!='fsd50k' or index in excluded:continue
        sid=r['id'].split('/')[-1].replace('.wav','');m=meta.get(sid,{})
        if sid=='535384' or m.get('uploader','').casefold() in reserved:continue
        text=html.unescape(re.sub('<[^>]*>',' ',str(m.get('description',''))));title=m.get('title','');alltext=(title+' '+text+' '+' '.join(m.get('tags',[]))).lower()
        terms=[term for term in ('clack','clatter','clickety','clickety-clack','rail joint','wheels','squeal','screech','freight') if term in alltext]
        if not any(term in alltext for term in ('clack','clatter','clickety','rail joint','squeal','screech')):continue
        path=Path(DATA)/r['path']
        if not path.exists():continue
        us_hints=[term for term in ('united states','california','colorado','pennsylvania','bnsf','union pacific','norfolk southern','amtrak','seattle','chicago') if term in alltext]
        candidates.append(dict(manifest_row=index,source_id=sid,title=title,creator=m.get('uploader'),metadata_terms=terms,us_metadata_hints=us_hints,location_status='unconfirmed; names/operators alone do not verify location',description=text[:2500],path=r['path'],source_url='https://freesound.org/s/'+sid+'/',license=r['license'],role='train_review',review_status='UNREVIEWED_METADATA_CANDIDATE',previous_exposure='original TRAIN source; cannot become independent acceptance audio'))
    candidates.sort(key=lambda r:(not bool(r['us_metadata_hints']),-len(r['metadata_terms']),r['source_id']))
    (root/'catalog.json').write_text(json.dumps(dict(candidates=candidates,count=len(candidates),limits=['metadata can describe sounds absent from the available interval','no interval label approved','all sources previously available as training; no new independent families claimed','unconfirmed geography']),indent=2))
    for r in candidates[:15]:print(r['source_id'],r['title'],r['creator'],r['metadata_terms'],r['us_metadata_hints'],flush=True)
    print('TOTAL',len(candidates),flush=True)

if __name__=='__main__':main()
