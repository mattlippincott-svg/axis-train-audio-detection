"""Discover source pages for US freight review; no audio labels or fitting."""
import html,json,re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests

ROOT=Path('experiments/us-freight-candidates')
QUERIES=('freight train USA','freight train Texas','freight train Pennsylvania',
         'freight train California','freight train Chicago','freight train Arkansas',
         'Norfolk Southern train','Union Pacific freight')

def search(q):
    r=requests.get('https://freesound.org/search/',params={'q':q},timeout=(10,25));r.raise_for_status()
    records=[]
    for b in re.findall(r'<div\s+class="bw-player"(.*?)</div>',r.text,re.S):
        d={k:html.unescape(v) for k,v in re.findall(r'data-([a-z-]+)="([^"]*)"',b)}
        if d.get('sound-id'):records.append(dict(id=int(d['sound-id']),creator=d.get('username'),title=d.get('title'),duration=d.get('duration'),query=q))
    return records

def main():
    ROOT.mkdir(exist_ok=True)
    found={};errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(search,q) for q in QUERIES]):
            try:
                for r in future.result():
                    if r['id'] not in found:found[r['id']]=r
                    else:found[r['id']].setdefault('other_queries',[]).append(r['query'])
            except Exception as e:errors.append(str(e))
    (ROOT/'search-catalog.json').write_text(json.dumps({'candidates':list(found.values()),'errors':errors},indent=2))
    print(json.dumps(list(found.values()),indent=2),flush=True)

if __name__=='__main__':main()
