"""Prepare a fresh acoustic-transfer check, not training data or deployment."""
import csv,hashlib,json,re
from pathlib import Path
import numpy as np
import soundfile as sf
from common import DATA,MANIFEST,read_wav16

def main():
    dest=Path('/lab/outputs');name='fresh-wheel-rail-evaluation-v1';manifest=dest/(name+'.json')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8-sig')).get('recording_status','').startswith('CAPTURED'):raise RuntimeError('Captured evaluation playback is immutable')
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')));meta={}
    for part in ('dev','eval'):meta.update(json.loads((Path(DATA)/'meta'/f'{part}_clips_info_FSD50K.json').read_text()))
    def sourceid(r):return r['id'].split('-')[1] if r['source']=='esc50' else r['id'].split('/')[-1].replace('.wav','')
    training_ids={sourceid(r) for r in rows if r['split']=='train'}
    cat=json.loads(Path('experiments/rail-corpus-v2/catalog-audited.json').read_text());reserved={r['creator'].casefold() for r in cat['sources'] if r['role']=='reserved_review'}
    reserved|={r['creator'].casefold() for r in json.loads(Path('experiments/us-freight-candidates/acquired/catalog.json').read_text())['sources'] if r['role']=='reserved_review'}
    reserved|={s.casefold() for s in json.loads(Path('experiments/rail-sounds/results.json').read_text())['reserved_authors']}
    parts=[np.zeros(5*16000,np.float32)];cursor=5.;segments=[];used=set()
    def add(w,record):
        nonlocal cursor
        assert len(w)==160000
        record.update(playback_start_s=cursor,playback_end_s=cursor+10,global_gain=.8)
        parts.append(w*.8);segments.append(record);cursor+=10
        parts.append(np.zeros(10*16000,np.float32));cursor+=10
    for target in ('truck','mechanical_fan'):
        pool=[]
        for index,r in enumerate(rows):
            if r['split']!='test' or r['label']!='0' or r['source']!='fsd50k' or sourceid(r) in training_ids:continue
            if target not in r['labels'].lower():continue
            m=meta.get(sourceid(r),{});creator=m.get('uploader','')
            if creator.casefold() in reserved or creator in used:continue
            text=m.get('title','')+' '+m.get('description','')
            if re.search(r'\b(train|trains|railway|railroad|tram|subway)\b',text,re.I):continue
            # Dataset tags alone are sometimes broader than the recording title.
            if target=='truck' and not re.search(r'\b(truck|trucks|lorry|lorries)\b',text,re.I):continue
            if target=='mechanical_fan' and (not re.search(r'\b(fan|ventilator|ventilation|hvac)\b',text,re.I) or re.search(r'\b(airplane|aeroplane|aircraft|flight|jet)\b',text,re.I)):continue
            path=Path(DATA)/r['path']
            if not path.exists() or sf.info(path).duration<10:continue
            pool.append((hashlib.sha256(('fresh-eval-20261110:'+r['id']).encode()).hexdigest(),index,r,m))
        assert pool,'No eligible held-out control '+target
        _,index,r,m=min(pool);used.add(m.get('uploader',''))
        add(read_wav16(Path(DATA)/r['path'])[:160000],dict(source_id=r['id'],manifest_row=index,expected_class='non_train',target=target,source_start_s=0,source_end_s=10,source_url='https://freesound.org/s/'+sourceid(r)+'/',creator=m.get('uploader'),title=m.get('title'),license=r['license'],dataset_labels=r['labels'],basis='original TEST negative, original source ID absent from TRAIN; metadata-screened, not newly human verified'))
    review=Path('experiments/us-freight-candidates/review-round-2')
    for sid,start,end,description in [(157864,17,27,'reviewer: high-speed repeated clacks'),(256259,2,12,'reviewer: slow repeated clacks, brief squeal near excerpt second eight'),(580528,52,62,'reviewer: flat-spot-like rhythmic wheel/rail sound')]:
        m=json.loads((review/'sources'/str(sid)/'metadata.json').read_text());w=read_wav16(review/'sources'/str(sid)/'audio.wav')[start*16000:end*16000]
        add(w,dict(source_id=sid,source_start_s=start,source_end_s=end,expected_class='train',description=description,creator=m['creator'],source_url=m['source_url'],license_urls=m['license_urls'],basis='exact user-reviewed interval; source audio fitted, acoustic replay not yet captured'))
    audio=np.concatenate(parts);assert len(audio)/16000==105
    path=dest/(name+'.wav');sf.write(path,audio,16000,subtype='PCM_16')
    report=dict(duration_s=105,recording_status='PREPARED_NOT_CAPTURED',audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),segments=segments,purpose='fresh microphone transfer evaluation; keep resulting capture outside fitting',gain_policy='one fixed factor 0.8; no per-clip normalization',silence_policy='five-second lead, ten-second gaps/tail; negatives precede trains to avoid event-release carryover',device_policy='record existing audio stream; installed app remains baseline, candidate evaluated offline',limits=['source-fitted train examples are not unseen-source acceptance','historically scored TEST controls are not new field false-alarm evidence','room/speaker/microphone conditions and input level must be logged'])
    manifest.write_text(json.dumps(report,indent=2));print([(s['expected_class'],s['source_id'],s['playback_start_s'],s['playback_end_s']) for s in segments],flush=True)

if __name__=='__main__':main()
