"""Report device-like results on held-out field railway recordings."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import MANIFEST, OUT  # noqa: E402
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("train_head", os.path.join(os.path.dirname(__file__), "03_train_head.py"))
train_head = module_from_spec(spec)
spec.loader.exec_module(train_head)


def main():
    from tf_keras.models import load_model

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    d = train_head.load()
    yamnet_w, yamnet_b = train_head.yamnet_classifier()
    x = train_head.features(d["emb"], yamnet_w, yamnet_b)
    model = load_model(os.path.join(OUT, "head.h5"), compile=False)
    test = (d["split"] == 2) & (d["aug"] == 0)
    for source in ("infrabel_cc0", "tram_curving_cc_by"):
        selected = np.array([rows[c]["source"] == source for c in d["clip"]])
        mask = test & selected
        if not mask.any():
            continue
        p = model.predict(x[mask], batch_size=4096, verbose=0).ravel()
        clips = d["clip"][mask]
        clip_ids = np.unique(clips)
        clip_max = train_head.clip_scores(p, clips)
        print(f"\nheld-out {source} clips: {len(clip_ids)}")
        for threshold in (0.3, 0.5, 0.7):
            detected = [clip_max[c] >= threshold for c in clip_ids]
            print(f"max 2 s-smoothed score >= {threshold:.1f}: {sum(detected)}/{len(detected)} ({np.mean(detected):.3f})")
        for seconds in (1.5, 3.0):
            windows = max(1, int(round(seconds / train_head.HOP_S)))
            detected = [train_head.event_detected(p[clips == c], 0.5, windows) for c in clip_ids]
            print(f"threshold 0.5 sustained {seconds:.1f}s: {sum(detected)}/{len(detected)} ({np.mean(detected):.3f})")
        print(f"window scores: median={np.median(p):.3f}, p90={np.percentile(p,90):.3f}, max={np.max(p):.3f}")

    # Check the existing held-out Freesound distribution separately so an
    # all-positive rail set cannot conceal a regression in confuser handling.
    for source in ("fsd50k", "esc50"):
        source_mask = test & np.array([rows[c]["source"] == source for c in d["clip"]])
        source_scores = model.predict(x[source_mask], batch_size=4096, verbose=0).ravel()
        train_head.evaluate(f"head test {source}", d["label"][source_mask], source_scores,
                            d["clip"][source_mask], rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
