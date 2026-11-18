"""Audit two publicly licensed US freight video sources and rank review clips.

Original creator/channel defines family; Commons import uploader does not.
No label is inferred from the file title or model predictions.
"""
import hashlib,json
from pathlib import Path
from importlib import import_module
import soundfile as sf

ROOT=Path('experiments/us-freight-candidates/commons')
SOURCES=[
    dict(id='commons-natchez-2011',creator='Isma Monfort Vialcanet',family='youtube:UCshP8x6Lcmagr8UH_Y10sjw',
        role='train_review',downloaded=True,source_url='https://commons.wikimedia.org/wiki/File:Freight_Train_passing_by_at_Natchez,_Louisiana.webm',
        original_url='https://www.youtube.com/watch?v=xRMTML4qDnM',
        download_url='https://upload.wikimedia.org/wikipedia/commons/0/01/Freight_Train_passing_by_at_Natchez%2C_Louisiana.webm',
        license_url='https://creativecommons.org/licenses/by/3.0/',location='Natchez, Louisiana, United States',
        location_basis='Commons file title and Louisiana categories',recorded_date='2011-12-12',
        expected_sha1='713ccdb7c89a4c12eec34ea62b3f46d4216f1b94',
        expected_duration_s=92.721,review_status='UNREVIEWED'),
    dict(id='commons-bnsf-snow-2016',creator='CRSWW RAIL FAN RXR / Brian Rock Railfan',family='youtube:UCbJ1AnhANs7SOrVFhMF0E3g',
        role='train_review',downloaded=True,source_url='https://commons.wikimedia.org/wiki/File:BNSF_freight_train_w_snow_on_trees_%26_ground_part_2.webm',
        original_url='https://www.youtube.com/watch?v=sBR8sesXZog',
        download_url='https://upload.wikimedia.org/wikipedia/commons/b/b6/BNSF_freight_train_w_snow_on_trees_%26_ground_part_2.webm',
        license_url='https://creativecommons.org/licenses/by/3.0/',location='United States; exact recording site not established',
        location_basis='Commons rail-transport and video country categories; no specific state inferred',recorded_date='2016-12-15',
        expected_sha1='1b57c5812c12682941f2116948f1242d7b743085',
        expected_duration_s=129.281,review_status='UNREVIEWED'),
]

def main():
    for r in SOURCES:
        folder=ROOT/'sources'/r['id'];raw=(folder/'source.webm').read_bytes()
        if hashlib.sha1(raw).hexdigest()!=r['expected_sha1']:raise RuntimeError('Original media checksum mismatch: '+r['id'])
        wave,rate=sf.read(folder/'audio.wav',dtype='float32')
        assert wave.ndim==1 and rate==16000
        r['duration_s']=len(wave)/rate;r['sha256']=hashlib.sha256(raw).hexdigest()
        assert abs(r['duration_s']-r['expected_duration_s'])<.1
        r['changes']='audio extracted from WebM, downmixed to mono and resampled to 16 kHz; source content not otherwise edited'
        (folder/'metadata.json').write_text(json.dumps(r,indent=2))
    report={'sources':SOURCES,'source_count':len(SOURCES),'downloaded_count':len(SOURCES),'original_creator_families':2,
        'labels_approved':0,'notes':['training candidates only; no reserved v2 source is included',
                                  'original video may contain non-train sound; content requires listening review',
                                  'all eight new Freesound candidates remain unacquired after HTTP 403']}
    (ROOT/'catalog-audited.json').write_text(json.dumps(report,indent=2))
    proposer=import_module('28_propose_rail_intervals');proposer.ROOT=ROOT;proposer.main()

if __name__=='__main__':main()
