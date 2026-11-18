"""Prepare one short actual-microphone TRAINING capture, with known source timing.

No device interaction or recording. Existing microphone evaluation captures
remain unfitted. Train-only negative examples help separate the playback path
from the target class. Source dynamics are retained under one fixed global gain.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
from common import DATA,MANIFEST,read_wav16

def main():
    p=argparse.ArgumentParser();p.add_argument('--outputs',required=True);a=p.parse_args();dest=Path(a.outputs)
    prior=dest/'train-microphone-pairing-playback.json'
    if prior.exists() and json.loads(prior.read_text()).get('recording_status','').startswith('CAPTURED'):
        raise RuntimeError('Captured playback is immutable; choose a fresh output directory for another sequence')
    root=Path('experiments');corpus=root/'rail-corpus-v2';us=root/'us-freight-candidates'/'commons'
    human=json.loads((corpus/'human-interval-labels.json').read_text())['labels']
    us_human=json.loads((us/'human-interval-labels.json').read_text())['labels']
    labels=human+[r for r in us_human if r['source_id']=='commons-natchez-2011']
    sources=[]
    for sid,start,end in [('commons-natchez-2011',52,62),(76078,52,62),(636899,17,27)]:
        label=next(r for r in labels if r['source_id']==sid and r['start_s']==start and r['end_s']==end)
        assert label['review_status']=='HUMAN_REVIEWED' and label['rhythmic_clatter']=='present' and label['bell']=='absent' and label['horn']=='absent'
        base=us if isinstance(sid,str) else corpus
        w=read_wav16(base/'sources'/str(sid)/'audio.wav')[start*16000:end*16000]
        assert len(w)==160000
        if isinstance(sid,str):metadata=json.loads((base/'sources'/sid/'metadata.json').read_text())
        else:metadata=json.loads((base/'sources'/str(sid)/'metadata.json').read_text())
        sources.append((w,dict(source_id=sid,source_start_s=start,source_end_s=end,expected_class='train',
            label_basis='user-reviewed exact interval',creator=label['creator'],family=label['family'],
            source_url=metadata['source_url'],license_urls=metadata.get('license_urls',[metadata.get('license_url')]),
            reviewed_label=label)))
    rows=list(csv.DictReader(open(MANIFEST,encoding='utf-8')))
    def original_family(r):
        return r['id'].split('-')[1] if r['source']=='esc50' else r['id'].split('/')[-1].replace('.wav','')
    evaluation_families={original_family(r) for r in rows if r['split']!='train'}
    forbidden=set(json.loads((corpus/'catalog-audited.json').read_text())['future_training_excluded_legacy_rows'])
    forbidden.update(json.loads((root/'us-freight-candidates'/'acquired'/'catalog.json').read_text())['future_training_excluded_legacy_rows'])
    meta={}
    for part in ('dev','eval'):meta.update(json.load(open(Path(DATA)/'meta'/f'{part}_clips_info_FSD50K.json')))
    for target in ('washing_machine','truck'):
        for i,r in enumerate(rows):
            if i in forbidden or r['split']!='train' or r['label']!='0' or original_family(r) in evaluation_families:continue
            tags={v.lower() for v in r['labels'].split(',')}
            if target=='washing_machine' and (r['source']!='esc50' or 'washing_machine' not in tags):continue
            if target=='truck' and ('truck' not in tags or 'engine' not in tags or any('train' in t or 'horn' in t or 'siren' in t for t in tags)):continue
            w=read_wav16(Path(DATA)/r['path'])
            if len(w)<80000:continue
            sid=r['id'].split('/')[-1].replace('.wav','');m=meta.get(sid,{})
            info=dict(source_id=r['id'],source_start_s=0,source_end_s=5,expected_class='non_train',
                label_basis='existing TRAIN-split dataset negative; not a newly human-reviewed interval',
                dataset=r['source'],dataset_labels=r['labels'],manifest_row=i,manifest_split=r['split'],
                creator=m.get('uploader'),license=r['license'],family='legacy:'+r['source']+':'+r['id'],
                source_url='https://freesound.org/s/'+sid+'/' if r['source']=='fsd50k' else 'https://github.com/karolpiczak/ESC-50',
                title=m.get('title',r['id']),purpose=target)
            if r['source']=='esc50':
                info.update(original_source_url='https://freesound.org/s/'+original_family(r)+'/',dataset_credit='ESC-50, Karol J. Piczak; original recording identified by linked Freesound source',creator_status='original recording creator not resolved in local metadata')
            sources.append((w[:80000],info));break
        else:raise RuntimeError('No suitable training-only negative for '+target)
    parts=[np.zeros(80000,np.float32)];cursor=5.;timeline=[]
    for i,(wave,info) in enumerate(sources):
        duration=len(wave)/16000
        info.update(playback_start_s=cursor,playback_end_s=cursor+duration,playlist_global_gain=.8)
        timeline.append(info);parts.append(wave*.8);cursor+=duration
        if i<len(sources)-1:parts.append(np.zeros(32000,np.float32));cursor+=2
    parts.append(np.zeros(80000,np.float32));cursor+=5
    audio=np.concatenate(parts);assert len(audio)==58*16000 and np.abs(audio).max()<=.80001
    output=dest/'train-microphone-pairing-playback.wav';sf.write(output,audio,16000,subtype='PCM_16')
    report={'duration_s':cursor,'purpose':'paired training data acquisition; not a new model test',
        'source_segments':timeline,'audio_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'recording_status':'NOT_RECORDED','user_action':'wait for recorder armed, then play through the usual speakers at a comfortable steady level',
        'gain_policy':'one fixed factor 0.8 applied to every excerpt; no compression or per-clip normalization',
        'evaluation_protection':'existing Colorado and NS microphone references remain excluded from gradient training',
        'notes':['first and last five seconds are digital silence; capture will contain actual ambient sound',
                 'crossing-dominant clip 7 is absent','negatives come only from original training split',
                 'speaker playback must be captured and acoustically aligned before source labels are transferred']}
    (dest/'train-microphone-pairing-playback.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'duration_s':cursor,'segments':[{k:r.get(k) for k in ('source_id','expected_class','playback_start_s','playback_end_s')} for r in timeline]},indent=2),flush=True)

if __name__=='__main__':main()
