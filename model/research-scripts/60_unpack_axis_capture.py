"""Unpack completed Axis RFC3640 AAC payloads to ADTS; do not decode here."""
import argparse,json,re
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-id',required=True);args=parser.parse_args();assert re.fullmatch(r'[a-z0-9-]+',args.capture_id)
    folder=Path('/lab/work/audio-diagnostics')/args.capture_id
    state=json.loads((folder/'capture-status.json').read_text(encoding='utf-8-sig'));assert not state['status'].startswith('CANCELLED')
    headers=(folder/'headers.txt').read_text();assert 'config=1188' in headers and 'audio/mpeg4-generic' in headers
    raw=(folder/'capture.aac').read_bytes();pos=frames=0;out=bytearray()
    while pos+4<=len(raw):
        bits=int.from_bytes(raw[pos:pos+2],'big');assert bits==16,('Unexpected AU header',pos,bits)
        size=int.from_bytes(raw[pos+2:pos+4],'big')>>3
        if pos+4+size>len(raw):break
        length=size+7;assert 7<length<8192
        out.extend(bytes([255,241,76,64|(length>>11),(length>>3)&255,((length&7)<<5)|31,252]));out.extend(raw[pos+4:pos+4+size]);pos+=4+size;frames+=1
    assert frames>0
    (folder/'capture.adts').write_bytes(out);report=dict(raw_bytes=len(raw),frames=frames,seconds_at_48khz=frames*1024/48000,trailing_bytes=len(raw)-pos)
    (folder/'unpack-report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)

if __name__=='__main__':main()
