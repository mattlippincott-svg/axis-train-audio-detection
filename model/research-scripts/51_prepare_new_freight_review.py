import json,hashlib,shutil
from pathlib import Path
import soundfile as sf
from importlib import import_module
src=Path('experiments/us-freight-candidates/freesound-recheck-2026-11-10');dest=Path('experiments/us-freight-candidates/review-round-2');sources=[]
for sid in (157864,256259,383465,580528,667746):
 folder=src/'sources'/str(sid);r=json.loads((folder/'metadata.json').read_text());assert r['role']=='train_review' and r['downloaded']
 assert hashlib.sha256((folder/'preview.mp3').read_bytes()).hexdigest()==r['preview_sha256']
 info=sf.info(folder/'audio.wav');assert info.samplerate==16000 and info.channels==1 and abs(info.duration-r['page_duration_s'])<.15
 out=dest/'sources'/str(sid);out.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/'audio.wav',out/'audio.wav')
 r.update(decoded_duration_s=info.duration,decoded_sha256=hashlib.sha256((out/'audio.wav').read_bytes()).hexdigest(),changes='public MP3 preview decoded, downmixed to mono, resampled to 16 kHz PCM16; no normalization',original_preview_path=str(folder/'preview.mp3'))
 (out/'metadata.json').write_text(json.dumps(r,indent=2));sources.append(r)
(dest/'catalog-audited.json').write_text(json.dumps(dict(sources=sources,labels_approved=0,policy='training review only; no reserved sources acquired or screened'),indent=2))
m=import_module('28_propose_rail_intervals');m.ROOT=dest;m.main()

