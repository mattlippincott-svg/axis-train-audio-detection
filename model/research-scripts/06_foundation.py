"""Isolated baseline audit and fresh embedding-head experiment.

Never writes model/out or deploys. Validation selects calibration; test is reported
only after fitting. Existing weak labels are retained and reported as a limitation.
"""
import collections
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np

from common import MANIFEST, OUT

DEST = Path(OUT).parent / 'experiments' / 'foundation'


def smooth(p):
    # Causal startup behavior, matching the app's four-window moving average.
    c = np.r_[0., np.cumsum(p, dtype=np.float64)]
    i = np.arange(len(p))
    return (c[i + 1] - c[np.maximum(0, i - 3)]) / np.minimum(i + 1, 4)


def event_score(p):
    s = smooth(p)
    # ceil(3/.48): seven consecutive windows, not round(3/.48).
    if len(s) < 7:
        return 0.
    return float(np.lib.stride_tricks.sliding_window_view(s, 7).min(1).max())


def metrics(y, scores, threshold):
    yes = scores >= threshold
    tp, fp = int(np.sum(yes & y)), int(np.sum(yes & ~y))
    fn, tn = int(np.sum(~yes & y)), int(np.sum(~yes & ~y))
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=2*precision*recall/max(1e-12, precision+recall),
                negative_clip_false_positive_rate=fp/max(1, fp+tn))


def summarize(p, d, rows, split, threshold):
    chosen = (d['split'] == split) & (d['aug'] == 0)
    ids = np.unique(d['clip'][chosen])
    events = np.array([event_score(p[chosen & (d['clip'] == c)]) for c in ids])
    labels = np.array([int(rows[c]['label']) == 1 for c in ids])
    sources = np.array([rows[c]['source'] for c in ids])
    report = {'all': metrics(labels, events, threshold), 'by_source': {}}
    for source in sorted(set(sources)):
        m = sources == source
        report['by_source'][source] = metrics(labels[m], events[m], threshold)
    confusers = {}
    for tag in ('Truck', 'Bus', 'Aircraft', 'Engine', 'Speech', 'church_bells'):
        m = np.array([not labels[i] and tag in rows[c]['labels'].split(',')
                      for i, c in enumerate(ids)])
        if m.any():
            confusers[tag] = {'clips': int(m.sum()), 'false_positives': int(np.sum(events[m] >= threshold))}
    report['confusers'] = confusers
    return report, labels, events


def main():
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    import tensorflow as tf
    from tf_keras import Model, layers, models, optimizers, callbacks, regularizers, utils, backend
    tf.random.set_seed(20261103)
    np.random.seed(20261103)
    DEST.mkdir(parents=True, exist_ok=True)
    old = importlib.import_module('03_train_head')
    rows = list(csv.DictReader(open(MANIFEST, encoding='utf-8')))
    # Preserve quiet windows for honest evaluation; filtering applies only to fit weights.
    d = dict(np.load(Path(OUT) / 'embeddings.npz'))
    families = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if row['source'] == 'esc50':
            families[row['id'].split('-')[1]].append(i)
        elif row['source'] == 'fsd50k':
            families[row['id'].split('/')[-1].replace('.wav', '')].append(i)
    shared = {k: v for k, v in families.items()
              if len({rows[i]['source'] for i in v}) > 1}
    excluded = {i for ids in shared.values() for i in ids}
    keep = ~np.isin(d['clip'], list(excluded))
    d = {k: v[keep] for k, v in d.items()}
    for c in np.unique(d['clip']):
        if len(np.unique(d['split'][d['clip'] == c])) != 1:
            raise ValueError(f'Clip {c} crosses splits')
    audit = {'sources': dict(collections.Counter(r['source'] for r in rows)),
             'shared_freesound_families_excluded': len(shared),
             'shared_families_crossing_original_splits': sum(len({rows[i]['split'] for i in ids}) > 1 for ids in shared.values()),
             'manifest_rows_excluded': len(excluded),
             'clips': len(rows), 'windows': len(d['label']),
             'baseline_sha256': hashlib.sha256((DEST/'baseline'/'traindet.tflite').read_bytes()).hexdigest(),
             'limitations': ['Weak clip labels, not verified train-presence intervals',
                            'Existing held-out data has been used in prior experiments; not pristine',
                            'Infrabel has recording-level splits, not verified independent sites',
                            'No speed annotations or category-level freight ground truth',
                            'False-positive clip rate is not false alarms per hour']}
    (DEST/'audit.json').write_text(json.dumps(audit, indent=2))
    print('audit', audit, flush=True)
    w, b = old.yamnet_classifier()
    x = old.features(d['emb'], w, b)
    y = d['label'].astype(np.float32)
    tr, va = d['split'] == 0, d['split'] == 1
    original = d['aug'] == 0
    count = collections.Counter(zip(d['clip'][tr].tolist(), d['aug'][tr].tolist()))
    # Each recording/augmentation contributes equally, rather than long recordings
    # dominating the optimizer. Speech cleaning is confined to the training set.
    weight = np.array([1/max(1, count.get((int(c), int(a)), 1))
                       for c, a in zip(d['clip'], d['aug'])], dtype=np.float32)
    speech = x[:, 1024:][:, [0,1,2,3,4,5]].max(1)
    rail = x[:, 1024:][:, [322,323,324,325,326,327,328]].max(1)
    weight[(y == 1) & (speech >= .5) & (rail < .1)] = 0
    weight[d['rms_db'] < -70] = 0
    hard = np.array([bool(set(rows[c]['labels'].split(',')) & old.HARD_NEG_LABELS) for c in d['clip']])
    weight[(y == 0) & hard] *= 2
    weight[tr] /= weight[tr].mean()
    baseline = models.load_model(DEST/'baseline'/'head.h5', compile=False)
    def predict_batched(model):
        return np.concatenate([model(x[i:i+2048], training=False).numpy().ravel()
                               for i in range(0, len(x), 2048)])
    baseline_p = predict_batched(baseline)
    del baseline
    backend.clear_session()
    print('baseline predictions complete', flush=True)
    inp = layers.Input((1545,), name='features')
    # Fresh head deliberately does not receive YAMNet class scores, reducing direct
    # dependence on horn/train tags. This is an experiment, not an assumed improvement.
    z = layers.Lambda(lambda v: v[:, :1024], name='embedding_only')(inp)
    z = layers.LayerNormalization(name='embedding_normalization')(z)
    z = layers.Dropout(.25)(z)
    z = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(z)
    z = layers.Dropout(.3)(z)
    out = layers.Dense(1, activation='sigmoid', name='head_out')(z)
    candidate = Model(inp, out)
    candidate.compile(optimizer=optimizers.Adam(3e-4), loss='binary_crossentropy')
    class Batches(utils.Sequence):
        def __init__(self, mask, weighted):
            self.ids = np.flatnonzero(mask)
            self.weighted = weighted
        def __len__(self):
            return int(np.ceil(len(self.ids)/1024))
        def __getitem__(self, i):
            ids = self.ids[i*1024:(i+1)*1024]
            return (x[ids], y[ids], weight[ids]) if self.weighted else (x[ids], y[ids])
        def on_epoch_end(self):
            if self.weighted:
                np.random.shuffle(self.ids)
    training = Batches(tr, True)
    training.on_epoch_end()
    candidate.fit(training,
                  validation_data=Batches(va & original, False),
                  epochs=30, batch_size=1024, verbose=2,
                  workers=1, max_queue_size=2,
                  callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    p = predict_batched(candidate)
    _, vy, vs = summarize(p, d, rows, 1, .509)
    # Choose using validation ONLY. Optimize F1 of sustained head-only events.
    grid = np.linspace(.05,.95,91)
    scores = [metrics(vy, vs, t)['f1'] for t in grid]
    threshold = float(grid[int(np.argmax(scores))])
    dense = candidate.get_layer('head_out')
    kw, kb = dense.get_weights()
    target = .509
    shift = np.log(target/(1-target))-np.log(threshold/(1-threshold))
    dense.set_weights([kw, kb+shift])
    p = 1/(1+np.exp(-(np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))+shift)))
    candidate.save(DEST/'candidate-head.h5')
    result = {'candidate': 'embedding_only_layernorm_recording_balanced',
              'calibration_validation_threshold_before_shift': threshold,
              'evaluation': 'head-only causal four-window mean, seven consecutive windows >=0.509; no horn shortcut, no fusion or level gate',
              'limitations': audit['limitations']}
    for name, pred in [('baseline', baseline_p), ('candidate', p)]:
        result[name] = {label: summarize(pred, d, rows, split, .509)[0]
                        for label, split in [('validation',1),('test',2)]}
    (DEST/'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
