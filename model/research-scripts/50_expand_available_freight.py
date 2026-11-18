"""Recheck previously blocked TRAIN candidates; never acquire reserved audio."""
import hashlib,json
from pathlib import Path
from importlib import import_module

def main():
    prior=json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())
    root=Path('experiments/us-freight-candidates/freesound-recheck-2026-11-10')
    collector=import_module('26_curate_rail');collector.ROOT=root
    results=[]
    for old in prior['sources']:
        if old['role']!='train_review':continue
        r=collector.fetch((old['id'],(old['creator'],old['role'],old['review_note'])))
        for key in ('family','legacy_creator_rows','legacy_source_rows'):
            if key in old:r[key]=old[key]
        if r.get('downloaded'):
            assert hashlib.sha256((root/'sources'/str(r['id'])/'preview.mp3').read_bytes()).hexdigest()==r['preview_sha256']
        (root/'sources'/str(r['id'])/'metadata.json').write_text(json.dumps(r,indent=2))
        results.append(r);print(r['id'],r.get('http_status'),r.get('downloaded'),r.get('page_duration_s'),r.get('acquisition_status'),flush=True)
        if r.get('http_status') in (403,429):break
    report=dict(sources=results,downloaded_count=sum(bool(r.get('downloaded')) for r in results),reserved_sources_not_requested=[r['id'] for r in prior['sources'] if r['role']=='reserved_review'],notes=['compressed public previews, not original WAV','no labels approved automatically','creator overlap must be excluded from evaluation before fitting new intervals','long source previews remain catalog-only under original acquisition cap'])
    (root/'catalog-audited.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
