"""Catalog targeted real field recordings, preserving original-source provenance.

Downloads public preview audio for internal experiments, not redistribution.
Titles/descriptions are provisional labels, not human-verified interval labels.
"""
import argparse,html,json,re
from pathlib import Path
import requests

ROOT=Path('experiments/rail-sounds')

def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    records={}
    for query in ('train clatter','train click clack','freight train wheels'):
        h=requests.get('https://freesound.org/search/',params={'q':query},timeout=30)
        h.raise_for_status()
        for block in re.findall(r'<div\s+class="bw-player"(.*?)</div>',h.text,re.S):
            d={k:html.unescape(v) for k,v in re.findall(r'data-([a-z-]+)="([^"]*)"',block)}
            sid=d.get('sound-id')
            if sid: records[sid]={k:d.get(k) for k in ('sound-id','username','title','duration','mp3')}
    (ROOT/'search-catalog.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='mp3'} for r in records.values()],indent=2))
    if a.download:
        chosen={769890:'WeenyBeany',700035:'JotrainG',181868:'cognito%20perceptu',
                199536:'mlekoman',592835:'TheFlyFishingFilmmaker'}
        # Explicit CLI selection is added after inspecting the catalog.
        for sid,user in chosen.items():
            if (ROOT/f'{sid}.json').exists() and (ROOT/f'{sid}.mp3').exists():continue
            page=f'https://freesound.org/people/{user}/sounds/{sid}/'
            r=requests.get(page,timeout=30)
            if r.status_code!=200:
                failure={'id':sid,'page':page,'status':r.status_code,'downloaded':False}
                (ROOT/f'{sid}-unavailable.json').write_text(json.dumps(failure,indent=2))
                print('Source unavailable',failure,flush=True)
                continue
            h=r.text
            (ROOT/f'{sid}.html').write_text(h,encoding='utf-8')
            urls=re.findall(r'https://cdn.freesound.org/previews/[^"\s<>]+-hq.mp3',h)
            if not urls:raise RuntimeError(f'No HQ preview: {sid}')
            audio=requests.get(html.unescape(urls[0]),timeout=60);audio.raise_for_status()
            (ROOT/f'{sid}.mp3').write_bytes(audio.content)
            description=re.search(r'<meta[^>]+name="twitter:description"[^>]+content="([^"]*)"',h)
            info={'id':sid,'page':page,'preview':urls[0],
                  'description':html.unescape(description.group(1)) if description else None,
                  'licenses':sorted(set(re.findall(r'https?://creativecommons.org/(?:licenses|publicdomain)/[^"\s<>]+',h))),
                  'label_status':'uploader description; intervals not human verified'}
            (ROOT/f'{sid}.json').write_text(json.dumps(info,indent=2),encoding='utf-8')

if __name__=='__main__':main()
