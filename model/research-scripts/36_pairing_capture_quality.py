"""Align paired playback to microphone audio and measure capture usability."""
import json,hashlib,argparse,re
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfilt,correlate
from common import read_wav16

def db(w):return float(20*np.log10(np.sqrt(np.mean(np.asarray(w,dtype=np.float64)**2)+1e-12)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-id',default='paired-training-01');parser.add_argument('--playback-name',default='train-microphone-pairing-playback');args=parser.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+',args.capture_id):parser.error('Invalid capture identifier')
    if not re.fullmatch(r'[a-z0-9-]+',args.playback_name):parser.error('Invalid playback name')
    lab=Path('/lab');folder=lab/'work/audio-diagnostics'/args.capture_id
    sourcepath=lab/'outputs'/(args.playback_name+'.wav');planpath=lab/'outputs'/(args.playback_name+'.json')
    source=read_wav16(sourcepath);capture=read_wav16(folder/'capture.wav')
    plan=json.loads(planpath.read_text(encoding='utf-8-sig'));segments=plan.get('source_segments',plan.get('segments'))
    assert segments and hashlib.sha256(sourcepath.read_bytes()).hexdigest()==plan['audio_sha256']
    filt=butter(3,[250,3500],btype='bandpass',fs=16000,output='sos')
    a=sosfilt(filt,source);b=sosfilt(filt,capture);size=64000
    cs=np.r_[0.,np.cumsum(b*b)];den=np.sqrt(np.maximum(cs[size:]-cs[:-size],1e-20))
    anchors=[]
    for segment in segments:
        lo,hi=segment['playback_start_s'],segment['playback_end_s']
        candidates=np.arange(lo+.25,hi-4+.001,.25)
        st=max(candidates,key=lambda t:np.mean(a[round(t*16000):round(t*16000)+size]**2))
        start=round(st*16000);template=a[start:start+size]
        ncc=correlate(b,template,mode='valid',method='fft')/(den*np.sqrt(max(1e-20,np.sum(template*template))))
        index=int(np.argmax(np.abs(ncc)))
        anchors.append(dict(source_id=segment['source_id'],expected_class=segment['expected_class'],source_start_s=start/16000,
            capture_start_s=index/16000,correlation=float(ncc[index]),lag_s=(index-start)/16000))
    strong=[x for x in anchors if abs(x['correlation'])>=.2]
    lag=float(np.median([x['lag_s'] for x in strong])) if strong else 0.
    inliers=[x for x in strong if abs(x['lag_s']-lag)<=.15]
    accepted=len(inliers)>=3 and sum(x['expected_class']=='train' for x in inliers)>=2
    if accepted:lag=float(np.average([x['lag_s'] for x in inliers],weights=[abs(x['correlation']) for x in inliers]))
    report={'capture_id':args.capture_id,'playback_name':args.playback_name,'playback_sha256':plan['audio_sha256'],'capture_seconds':len(capture)/16000,'alignment_accepted':accepted,'lag_s':lag,
        'anchors':anchors,'corroborating_anchors':len(inliers),'capture_overall_rms_dbfs':db(capture),
        'capture_peak_dbfs':float(20*np.log10(max(1e-12,float(np.abs(capture).max())))),
        'clipped_sample_fraction':float(np.mean(np.abs(capture)>=.999)),
        'limitations':['uncalibrated digital audio levels, not dBA','silence-relative level is a room-background comparison, not calibrated SNR',
                      'aligned source labels do not rule out unexpected ambient interference']}
    if accepted:
        background=[]
        silence=[];cursor=0.
        for segment in sorted(segments,key=lambda r:r['playback_start_s']):
            if segment['playback_start_s']>cursor:silence.append((cursor,segment['playback_start_s']))
            cursor=segment['playback_end_s']
        if cursor<len(source)/16000:silence.append((cursor,len(source)/16000))
        for start,end in silence:
            for t in np.arange(start,end-.99,1):
                lo=round((t+lag)*16000);hi=lo+16000
                if lo>=0 and hi<=len(capture):background.append(db(capture[lo:hi]))
        bg=float(np.median(background));report['silence_median_rms_dbfs']=bg
        report['segments']=[]
        for segment in segments:
            lo=round((segment['playback_start_s']+lag)*16000);hi=round((segment['playback_end_s']+lag)*16000)
            complete=lo>=0 and hi<=len(capture);r=dict(segment,complete=complete,capture_start_s=lo/16000,capture_end_s=hi/16000)
            if complete:
                w=capture[lo:hi];windows=np.lib.stride_tricks.sliding_window_view(w,15600)[::7680]
                levels=20*np.log10(np.sqrt(np.mean(windows.astype(np.float64)**2,axis=1)+1e-12))
                r.update(rms_dbfs=db(w),peak_dbfs=float(20*np.log10(max(1e-12,float(np.abs(w).max())))),
                    above_volume_gate_fraction=float(np.mean(levels>=-65)),above_training_floor_fraction=float(np.mean(levels>=-70)),
                    median_window_dbfs=float(np.median(levels)),above_silence_median_db=db(w)-bg)
            report['segments'].append(r)
    (folder/'quality-and-alignment.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
