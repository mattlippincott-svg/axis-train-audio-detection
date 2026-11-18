"""Build an auditable candidate manifest and local interval-labeling aid.

No model-generated label is promoted to ground truth. No purchase or training.
"""
import argparse,csv,hashlib,html,json,re,shutil
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path('experiments/rail-corpus-v2')
SUMMARIES={
76078:'Recorded between passenger carriages; wheel clatter and passing crossing alarms. Onboard perspective.',
216642:'A nearby freight passage includes a horn during the approach. Clatter intervals need review.',
274547:'Unprocessed train pass recorded during a film shoot; handling noise is reported near the beginning.',
361021:'Distant slow freight in a city setting, with nearby voices and road vehicles.',
523976:'Amtrak passage near a crossing; the creator reports a spoken greeting that needs excluding.',
562108:'Electric passenger trains heard from above while slowing near a station; birds are also reported.',
592835:'Cargo-train recording; exact clatter intervals and background sounds still require listening.',
608679:'Fast trains crossing a steel bridge in the Netherlands. Bridge resonance may dominate some sections.',
634659:'Short train passage through a station, with wind wash and rail chirps described by the creator.',
636899:'A slowly departing electric passenger train, with occasional birds.',
639232:'Long tourist-train onboard recording combining wheel noise, squeal, horns, and crossing bells.',
640306:'Freight passing a railway platform in Celje, recorded while waiting for another train.',
660441:'Close freight passage recorded in winter at Houston, British Columbia, Canada.',
850660:'Nighttime freight-yard recording with wheel squeal, coupling impacts, and industrial background.',
871372:'Inside a passenger train in the Canadian Prairies; rail clatter overlaps horns, carriage squeaks, and passenger activity.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--outputs',required=True);p.add_argument('--album-page',required=True);a=p.parse_args()
    dest=Path(a.outputs);dest.mkdir(exist_ok=True,parents=True);assets=dest/'rail-review-audio';assets.mkdir(exist_ok=True)
    catalog=json.loads((ROOT/'catalog.json').read_text());cards=[];hashes={}
    proposals=json.loads((ROOT/'interval-proposals.json').read_text())['proposals'] if (ROOT/'interval-proposals.json').exists() else []
    priority={216642:0,592835:1,640306:2,361021:3}
    ordered=sorted(catalog['sources'],key=lambda r:(r['role']=='reserved_review',priority.get(r['id'],20),r['id']))
    for record in ordered:
        if not record['downloaded']:continue
        sid=record['id'];folder=ROOT/'sources'/str(sid);wave,rate=sf.read(folder/'audio.wav',dtype='float32')
        digest=hashlib.sha256((folder/'audio.wav').read_bytes()).hexdigest();hashes.setdefault(digest,[]).append(sid)
        duration=len(wave)/rate
        record['decoded_audio']={'sample_rate':rate,'channels':1 if wave.ndim==1 else wave.shape[1],
            'duration_s':duration,'peak_dbfs':float(20*np.log10(max(float(np.abs(wave).max()),1e-12))),
            'samples_at_or_above_0_999_fraction':float(np.mean(np.abs(wave)>=.999)),
            'sha256':digest,'duration_matches_page':abs(duration-(record.get('page_duration_s') or duration))<.25}
        record['perspective_review']={76078:'between carriages: auxiliary onboard perspective',871372:'inside passenger train: auxiliary onboard perspective',
            562108:'overhead viewpoint: auxiliary perspective',608679:'steel bridge ambience: auxiliary perspective'}.get(sid,'review creator description before assigning deployment relevance')
        target=assets/f'{sid}.mp3';shutil.copyfile(folder/'preview.mp3',target)
        cards.append(dict(id=sid,creator=record['creator'],family=record['family'],role=record['role'],
            title=record.get('title') or str(sid),description=SUMMARIES[sid],
            notes=record['review_note']+'; '+record['perspective_review'],url=record['source_url'],duration=duration,
            audio=f'rail-review-audio/{sid}.mp3',licenses=record.get('license_urls',[]),
            suggestions=[dict(start_s=p['start_s'],end_s=p['end_s']) for p in proposals if p['source_id']==sid]))
    catalog['exact_decoded_duplicate_groups']=[ids for ids in hashes.values() if len(ids)>1]
    catalog['decoded_total_seconds']=sum(r.get('decoded_audio',{}).get('duration_s',0) for r in catalog['sources'])
    # Parse creator's catalog metadata only. Do not obtain paid master files or
    # stream audio from this album. The album/creator remains one reserved group.
    page=Path(a.album_page).read_text(encoding='utf-8')
    match=re.search(r'data-tralbum="([^"]+)"',page)
    album=json.loads(html.unescape(match.group(1)))
    tracks=[dict(id=t['track_id'],track_number=t['track_num'],title=t['title'],duration_s=t.get('duration'),
        source_url='https://freetousesounds.bandcamp.com'+t['title_link']) for t in album['trackinfo']]
    album_record={'source_url':album['url'],'creator':'FreeToUseSounds','family':'collection:ftus-american-freight-2020',
        'role':'reserved_us_collection_pending_acquisition','current_minimum_price_usd':album['current'].get('minimum_price'),
        'track_count':len(tracks),'downloaded':False,'purchased':False,'tracks':tracks,
        'note':'One compilation plus 31 listed recordings; no audio obtained. Older description says FREE but current purchase minimum is 5 USD. Possible relation to prior Colorado clip must be checked before assigning an acceptance role.'}
    (ROOT/'us-library-catalog.json').write_text(json.dumps(album_record,indent=2))
    (ROOT/'catalog-audited.json').write_text(json.dumps(catalog,indent=2))
    public_catalog=json.loads(json.dumps(catalog))
    for r in public_catalog['sources']:
        r.pop('description',None);r['description_summary']=SUMMARIES[r['id']]
    (dest/'rail-candidate-catalog.json').write_text(json.dumps(public_catalog,indent=2))
    # User-facing summary avoids reproducing the long catalog of keyword titles.
    (dest/'us-freight-library-summary.json').write_text(json.dumps({k:v for k,v in album_record.items() if k!='tracks'},indent=2))
    template=Path(__file__).with_name('rail_review_template.html').read_text(encoding='utf-8')
    page=template.replace('/*CANDIDATE_DATA*/[]',json.dumps(cards).replace('<','\\u003c'))
    (dest/'rail-audio-review.html').write_text(page,encoding='utf-8')
    with open(dest/'rail-interval-label-template.csv','w',newline='') as h:
        w=csv.writer(h);w.writerow(['source_id','creator','family','role','start_s','end_s','clatter','squeal','horn','bell','engine','air_release','speech_music','road_vehicle','handling_wind','review_status','notes'])
        for c in cards:w.writerow([c['id'],c['creator'],c['family'],c['role'],0,round(c['duration'],3),*(['unknown']*9),'UNREVIEWED',c['notes']])
    print(json.dumps({'downloaded_recordings':len(cards),'creators':len({c['creator'] for c in cards}),
        'downloaded_minutes':catalog['decoded_total_seconds']/60,
        'train_review_recordings':sum(c['role']=='train_review' for c in cards),
        'reserved_review_recordings':sum(c['role']=='reserved_review' for c in cards),
        'exact_duplicate_groups':catalog['exact_decoded_duplicate_groups'],
        'master_library_obtained':False},indent=2),flush=True)

if __name__=='__main__':main()
