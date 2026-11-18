"""Run host logic and actual model-score fixtures without connecting to a device."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
image='traindet-hosttest'
def main():
    if subprocess.run(['docker','image','inspect',image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        dockerfile='FROM debian:bookworm-slim\nRUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*\n'
        subprocess.run(['docker','build','-t',image,'-'],input=dockerfile,text=True,check=True)
    prefix=['docker','run','--rm','--security-opt','seccomp=unconfined','-v',f'{ROOT}:/src:ro','-w','/src',image,'bash','-c']
    subprocess.run(prefix+['gcc -std=gnu11 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer acap/tests/test_host.c acap/app/detector.c acap/app/fusion.c acap/app/wav.c -lm -o /tmp/test_host && setarch $(uname -m) -R /tmp/test_host'],check=True)
    commands=['gcc -std=gnu11 -O1 -Wall -Wextra -Werror acap/tests/bell_replay.c acap/app/detector.c -lm -o /tmp/bell_replay']
    for name,expected in [('freight-bell',1),('heldout-bell',1),('steam-no-bell',0),('electronic-crossing',0)]:commands.append(f'/tmp/bell_replay tests/fixtures/{name}.csv {expected}')
    subprocess.run(prefix+[' && '.join(commands)],check=True)
    subprocess.run(['node','--check','acap/app/html/app.js'],cwd=ROOT,check=True)
    subprocess.run(['node','acap/tests/test_preview.cjs'],cwd=ROOT,check=True)

if __name__=='__main__':main()
