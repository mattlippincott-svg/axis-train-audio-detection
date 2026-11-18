"""Acquire and audit a broader provisional rail-sound pool. Never trains.

Creator-level roles are assigned before acoustic screening. No candidate is
called clean or interval-labeled merely because a model or title suggests it.
"""
import csv,hashlib,html,json,re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import quote
import requests
from common import DATA,MANIFEST

ROOT=Path('experiments/rail-corpus-v2')
CANDIDATES={
274547:('skradz','reserved_review','train pass; creator warns of microphone handling at onset'),
592835:('TheFlyFishingFilmmaker','train_review','cargo train pass; clatter not interval verified'),
608679:('klankbeeld','train_review','two trains on bridge; same creator as other klankbeeld recordings'),
562108:('klankbeeld','train_review','electric train overhead; bridge coloration'),
636899:('klankbeeld','train_review','slow electric train'),
361021:('klankbeeld','train_review','freight train with city ambience'),
76078:('ceejay','train_review','passenger train in motion; check interior perspective'),
634659:('Dan_AudioFile','train_review','fast close electric pass through station'),
640306:('dibko','train_review','freight train; geography and clatter need review'),
660441:('chancemedia','reserved_review','freight train; reserve creator before acoustic screening'),
216642:('Zabuhailo','train_review','freight train; check horn and bell contamination'),
523976:('alex36917','reserved_review','Amtrak; same creator as long mixed passenger recording'),
639232:('alex36917','reserved_review','long mixed recording: clatter, squeal, horns and crossing bells'),
871372:('annie.maheux','train_review','onboard Canadian train; auxiliary perspective only'),
850660:('wmodes','reserved_review','freight yard: squeal, metal impacts, coupling and industrial ambience'),
}

def fetch(item):
    sid,(user,role,note)=item
    dest=ROOT/'sources'/str(sid);dest.mkdir(parents=True,exist_ok=True)
    url=f'https://freesound.org/people/{quote(user)}/sounds/{sid}/'
    prior=dest/'metadata.json'
    if prior.exists():
        saved=json.loads(prior.read_text())
        if saved.get('downloaded'):return saved
    record=dict(id=sid,creator=user,role=role,source_url=url,review_note=note,
                label_status='unreviewed candidate; not training ground truth',downloaded=False)
    try:
        r=requests.get(url,timeout=(10,25));record['http_status']=r.status_code
        if r.status_code!=200:raise RuntimeError(f'HTTP {r.status_code}')
        h=r.text;(dest/'page.html').write_text(h,encoding='utf-8')
        def meta(name):
            m=re.search(r'<meta[^>]+(?:name|property)="'+re.escape(name)+r'"[^>]+content="([^"]*)"',h,re.S)
            return html.unescape(m.group(1)) if m else None
        record.update(title=meta('og:title'),description=meta('twitter:description'),
            license_urls=sorted(set(re.findall(r'https?://creativecommons.org/(?:licenses|publicdomain)/[^"\s<>]+',h))))
        match=re.search(r'data-duration="([\d.]+)"',h)
        record['page_duration_s']=float(match.group(1)) if match else None
        urls=re.findall(r'https://cdn.freesound.org/previews/[^"\s<>]+-hq.mp3',h)
        if not urls:raise RuntimeError('No public HQ preview located')
        preview=html.unescape(urls[0]);record['preview_url']=preview
        # Avoid multi-hour downloads. Long mixed files stay catalog-only until
        # interval review; no truncation is silently treated as a full recording.
        if record['page_duration_s'] and record['page_duration_s']>600:
            record['acquisition_status']='catalog_only_long_mixed_recording'
        else:
            a=requests.get(preview,timeout=(10,45));a.raise_for_status()
            (dest/'preview.mp3').write_bytes(a.content)
            record.update(downloaded=True,preview_bytes=len(a.content),preview_sha256=hashlib.sha256(a.content).hexdigest(),
                          acquisition_status='public compressed preview; not original WAV')
    except Exception as exc:record['error']=str(exc)
    prior.write_text(json.dumps(record,indent=2),encoding='utf-8');return record

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=[]
        for future in as_completed([pool.submit(fetch,item) for item in CANDIDATES.items()]):
            r=future.result();records.append(r);print(r['id'],r['creator'],r.get('acquisition_status',r.get('error')),flush=True)
    metadata={}
    for part in ('dev','eval'):metadata.update(json.load(open(Path(DATA)/'meta'/f'{part}_clips_info_FSD50K.json')))
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    review=[]
    for r in sorted(records,key=lambda r:r['id']):
        related=[(i,row) for i,row in enumerate(rows) if metadata.get(row['id'].split('/')[-1].replace('.wav',''),{}).get('uploader','').casefold()==r['creator'].casefold()]
        r['legacy_creator_rows']={s:sum(row['split']==s for i,row in related) for s in ('train','val','test')}
        r['legacy_source_rows']=[i for i,row in enumerate(rows) if row['id'].split('/')[-1].replace('.wav','')==str(r['id'])]
        r['family']='creator:'+r['creator'].casefold()
        review.append(r)
    roles={}
    for r in review:
        if r['family'] in roles and roles[r['family']]!=r['role']:raise RuntimeError('Creator split conflict')
        roles[r['family']]=r['role']
    reserved_creators={r['creator'].casefold() for r in review if r['role']=='reserved_review'}
    training_creators={r['creator'].casefold() for r in review if r['role']=='train_review'}
    exclusions=[i for i,row in enumerate(rows) if metadata.get(row['id'].split('/')[-1].replace('.wav',''),{}).get('uploader','').casefold() in reserved_creators]
    evaluation_exclusions=[i for i,row in enumerate(rows) if row['split']!='train' and metadata.get(row['id'].split('/')[-1].replace('.wav',''),{}).get('uploader','').casefold() in training_creators]
    report={'sources':review,'source_count':len(review),'downloaded_count':sum(r['downloaded'] for r in review),
        'downloaded_creators':len({r['creator'] for r in review if r['downloaded']}),
        'creator_roles':roles,'future_training_excluded_legacy_rows':exclusions,
        'future_evaluation_excluded_legacy_rows':evaluation_exclusions,
        'permanently_excluded':{'535384':'user identified overlapping muscle-car engine'},
        'policy':['all intervals remain unreviewed','creator-level grouping is conservative; recording-session identity may need further merging',
                  'reserved candidates never enter fitting or model-assisted interval selection',
                  'exclude listed legacy creator rows from future fitting','frozen baseline/backbone may already contain related material',
                  'no models trained and no device changes in this acquisition step']}
    (ROOT/'catalog.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with open(ROOT/'interval-review-template.csv','w',newline='',encoding='utf-8') as h:
        fields=['source_id','creator','role','start_s','end_s','clatter','squeal','horn','bell','engine','air_release','speech_music','road_vehicle','handling_wind','review_status','notes']
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for r in review:
            if r['downloaded']:w.writerow(dict(source_id=r['id'],creator=r['creator'],role=r['role'],start_s=0,end_s=r.get('page_duration_s'),review_status='UNREVIEWED',notes=r['review_note']))
    print(json.dumps({k:report[k] for k in ('source_count','downloaded_count','downloaded_creators')},indent=2),flush=True)

if __name__=='__main__':main()
