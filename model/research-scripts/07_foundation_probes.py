"""Direct-source diagnostic probes; never training inputs or pristine holdouts."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
import sys
import json
from pathlib import Path
import numpy as np
from common import DATA, OUT, read_wav16
from importlib import import_module
experiment = import_module('06_foundation')
sys.path.insert(0, str(Path(__file__).parent.parent/'yamnet'))
import params
import yamnet


def main():
    from tf_keras.models import load_model
    root = Path(OUT).parent/'experiments'/'foundation'
    backbone = yamnet.yamnet_frames_model(params.Params())
    backbone.load_weights(os.path.join(DATA, 'yamnet.h5'))
    heads = {'baseline': load_model(root/'baseline'/'head.h5', compile=False),
             'candidate': load_model(root/'candidate-head.h5', compile=False)}
    report = {'note': 'Freesound HQ MP3 previews decoded to 16 kHz mono; diagnostic probes previously discussed, not pristine test data. No microphone path.', 'clips': {}}
    for path in sorted((root/'probes').glob('*.wav')):
        wave = read_wav16(path)
        variants = {}
        for target in (None, -45., -30., -20.):
            rms = np.sqrt(np.mean(wave.astype(np.float64)**2)+1e-12)
            gain = 1. if target is None else min(10**(target/20)/rms, .98/max(1e-9, np.max(np.abs(wave))))
            audio = wave*gain
            # Full waveform framing preserves contiguous time. Discard padded tail.
            scores, emb, _ = backbone(audio)
            n = max(1, 1+(len(audio)-15600)//7680)
            scores, emb = scores.numpy()[:n], emb.numpy()[:n]
            x = np.concatenate([emb, scores], axis=1)
            item = {'achieved_rms_db': float(20*np.log10(rms*gain)), 'gain_db': float(20*np.log10(gain)), 'heads': {}}
            for name, head in heads.items():
                p = head(x, training=False).numpy().ravel()
                item['heads'][name] = {'peak': float(p.max()), 'sustained_3s_score': experiment.event_score(p),
                                       'head_only_event': experiment.event_score(p) >= .509}
                if path.stem == 'ns-locomotives':
                    # User reports bell transition near 25-30 s; first 20 s is a
                    # provisional engine-only probe, pending exact annotation.
                    early = p[np.arange(len(p))*.48 < 20]
                    item['heads'][name]['first20s_event_score'] = experiment.event_score(early)
            variants['native' if target is None else str(target)] = item
            print(path.stem, target, item, flush=True)
        report['clips'][path.stem] = variants
    (root/'direct-source-probes.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
