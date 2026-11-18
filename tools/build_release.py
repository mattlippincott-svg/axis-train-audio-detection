"""Build the shipped 0.6.3 source/model snapshot with Docker, on Windows or Linux."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def main():
    release=ROOT/'model/releases/0.6.3'
    manifest=json.loads((release/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen asset differs: {name}')
    dest=ROOT/'acap/app/model';dest.mkdir(exist_ok=True)
    shutil.copy2(release/'traindet.tflite',dest/'traindet.tflite')
    shutil.copy2(ROOT/'model/yamnet/yamnet_class_map.csv',dest/'yamnet_class_map.csv')
    image='train-detect-handoff:0.6.3'
    subprocess.run(['docker','build','--build-arg','ARCH=aarch64','-t',image,str(ROOT/'acap')],check=True)
    cid=subprocess.check_output(['docker','create',image],text=True).strip()
    try:
        build=ROOT/'build';build.mkdir(exist_ok=True)
        path=build/'Train_Audio_Detection_0_6_3_aarch64.eap'
        subprocess.run(['docker','cp',f'{cid}:/opt/app/Train_Audio_Detection_0_6_3_aarch64.eap',str(path)],check=True)
        print(path);print('SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    finally:subprocess.run(['docker','rm',cid],check=True)

if __name__=='__main__':main()
