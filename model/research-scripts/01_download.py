"""Download the first-swing training data and store it as 16 kHz mono PCM16 WAV.

Sources
  * ESC-50 (GitHub zip): 40 `train` clips + 1960 diverse negatives.
  * FSD50K (per-file via the Hugging Face mirror Fhrozen/FSD50k): every clip with a
    rail label (Train / Rail_transport / Subway) as positives, plus capped samples
    of known confuser classes (road vehicles, aircraft, engines, weather) and a
    random sample of everything else as negatives.

Writes data/manifest.csv: path,source,id,label,split,labels,license
Splits: ESC-50 folds 1-3 train, 4 val, 5 test. FSD50K dev split train/val, eval -> test.
Infrabel Batch5 is weakly labeled railway-positive. Tram-curving recordings are
CC-BY-4.0 railway-positive, with 2023-08-12 held out by collection date.
"""
import csv
import io
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from common import CLIPS, DATA, MANIFEST, decode_bytes, write_wav16

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
FSD_URL = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/{split}/{fname}.wav"
FSD_META_URL = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/{source}/{fname}"
YAMNET_WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
# A first CC0 batch of real trackside recordings, covering passenger/freight
# pass-bys and rail events such as braking, acceleration, horns and wheel squeal.
INFRABEL_BATCH_URL = "https://fr.ftp.opendatasoft.com/infrabel/Train_Sounds/Batch5.zip"
# Field recordings of trams passing a curve, useful for non-horn rolling,
# flanging and squeal sounds. Zenodo record 11540120, CC BY 4.0.
TRAM_ARCHIVE_URL = "https://zenodo.org/api/records/11540120/files/Wav%20files.zip/content"
TRAM_META_URL = "https://zenodo.org/api/records/11540120/files/datasets.csv/content"
META = os.path.join(DATA, "meta")

RAIL = {"Train", "Rail_transport", "Subway_and_metro_and_underground"}
# (label, max dev clips, max eval clips)
CONFUSERS = [
    ("Truck", 110, 44), ("Bus", 120, 40), ("Car_passing_by", 126, 60),
    ("Traffic_noise_and_roadway_noise", 152, 60), ("Motorcycle", 100, 40),
    ("Race_car_and_auto_racing", 60, 20), ("Accelerating_and_revving_and_vroom", 80, 30),
    ("Vehicle_horn_and_car_horn_and_honking", 100, 40), ("Car", 120, 40),
    ("Aircraft", 120, 45), ("Fixed-wing_aircraft_and_airplane", 101, 45),
    ("Engine", 150, 50), ("Idling", 80, 30), ("Engine_starting", 40, 15),
    ("Boat_and_Water_vehicle", 80, 30), ("Thunder", 120, 40), ("Thunderstorm", 80, 30),
    ("Wind", 150, 50), ("Rain", 120, 40), ("Waves_and_surf", 60, 20), ("Siren", 80, 30),
    ("Power_tool", 60, 20), ("Drill", 40, 15), ("Mechanical_fan", 60, 20),
    ("Church_bell", 40, 15), ("Explosion", 40, 15), ("Crowd", 60, 20), ("Speech", 80, 30),
    ("Music", 80, 30), ("Walk_and_footsteps", 40, 15), ("Bird", 60, 20), ("Dog", 40, 15),
]
RANDOM_NEG = {"dev": 900, "eval": 250}


def log(msg):
    print(msg, flush=True)


def ensure_yamnet_weights():
    path = os.path.join(DATA, "yamnet.h5")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log("YAMNet weights already present, skipping download")
        return
    log("Downloading YAMNet weights ...")
    os.makedirs(DATA, exist_ok=True)
    tmp_path = path + ".download"
    with requests.get(YAMNET_WEIGHTS_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(tmp_path) == 0:
        os.remove(tmp_path)
        raise RuntimeError("downloaded YAMNet weights are empty")
    os.replace(tmp_path, path)
    log(f"YAMNet weights downloaded: {os.path.getsize(path)} bytes")


def ensure_fsd_metadata():
    files = (
        ("dev.csv", "labels/dev.csv"),
        ("eval.csv", "labels/eval.csv"),
        ("dev_clips_info_FSD50K.json", "metadata/dev_clips_info_FSD50K.json"),
        ("eval_clips_info_FSD50K.json", "metadata/eval_clips_info_FSD50K.json"),
    )
    os.makedirs(META, exist_ok=True)
    for filename, source in files:
        path = os.path.join(META, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        url = FSD_META_URL.format(source=source.rsplit("/", 1)[0], fname=source.rsplit("/", 1)[1])
        log(f"Downloading FSD50K metadata: {filename}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        tmp_path = path + ".download"
        with open(tmp_path, "wb") as f:
            f.write(r.content)
        os.replace(tmp_path, path)


def esc50(rows):
    out_dir = os.path.join(CLIPS, "esc50")
    meta_path = os.path.join(META, "esc50.csv")
    zf = None
    if os.path.exists(meta_path) and os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= 2000:
        log("ESC-50 already present, skipping download")
        meta = list(csv.DictReader(open(meta_path, encoding="utf-8")))
    else:
        log("Downloading ESC-50 zip ...")
        r = requests.get(ESC50_URL, timeout=600)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        meta_name = next(n for n in zf.namelist() if n.endswith("meta/esc50.csv"))
        open(meta_path, "wb").write(zf.read(meta_name))
        meta = list(csv.DictReader(open(meta_path, encoding="utf-8")))
        audio_prefix = meta_name.replace("meta/esc50.csv", "audio/")
    split_of = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
    for i, m in enumerate(meta):
        path = os.path.join(out_dir, m["filename"])
        if not os.path.exists(path):
            write_wav16(path, decode_bytes(zf.read(audio_prefix + m["filename"])))  # zf set when missing
        rows.append({
            "path": os.path.relpath(path, DATA).replace("\\", "/"), "source": "esc50",
            "id": m["filename"], "label": int(m["category"] == "train"),
            "split": split_of[int(m["fold"])], "labels": m["category"],
            "license": "esc50 (CC BY-NC 3.0 dataset; per-clip in ESC-50 LICENSE)",
        })
        if i % 500 == 0:
            log(f"  esc50 {i}/{len(meta)}")
    log(f"ESC-50 done: {sum(r['label'] for r in rows if r['source']=='esc50')} positives")


def fsd_select(seed=1234):
    rnd = random.Random(seed)
    picked = {}
    for split in ("dev", "eval"):
        rows = list(csv.DictReader(open(os.path.join(META, f"{split}.csv"), encoding="utf-8")))
        for r in rows:
            r["labelset"] = set(r["labels"].split(","))
        pos = [r for r in rows if r["labelset"] & RAIL]
        neg_pool = [r for r in rows if not (r["labelset"] & RAIL)]
        for r in pos:
            picked[(split, r["fname"])] = (r, 1)
        col = 1 if split == "dev" else 2
        for conf in CONFUSERS:
            cands = [r for r in neg_pool if conf[0] in r["labelset"]]
            rnd.shuffle(cands)
            for r in cands[:conf[col]]:
                picked.setdefault((split, r["fname"]), (r, 0))
        rnd.shuffle(neg_pool)
        n = 0
        for r in neg_pool:
            if n >= RANDOM_NEG[split]:
                break
            if (split, r["fname"]) not in picked:
                picked[(split, r["fname"])] = (r, 0)
                n += 1
    return picked


def fsd_fetch(split, fname):
    path = os.path.join(CLIPS, "fsd50k", split, f"{fname}.wav")
    if os.path.exists(path):
        return path
    url = FSD_URL.format(split=split, fname=fname)
    last = None
    for attempt in range(8):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 429:  # HF anonymous rate limit: back off and retry
                time.sleep(float(r.headers.get("Retry-After", 0)) or min(120, 10 * 2 ** attempt))
                last = "429 Too Many Requests"
                continue
            r.raise_for_status()
            write_wav16(path, decode_bytes(r.content))
            return path
        except Exception as e:  # noqa: BLE001 - retry any network/decode error
            last = e
    raise RuntimeError(f"{url}: {last}")


def fsd50k(rows):
    lic = {}
    for split in ("dev", "eval"):
        info = json.load(open(os.path.join(META, f"{split}_clips_info_FSD50K.json"), encoding="utf-8"))
        lic.update({(split, k): v.get("license", "") for k, v in info.items()})
    picked = fsd_select()
    log(f"FSD50K: {len(picked)} clips selected, "
        f"{sum(1 for v in picked.values() if v[1])} positives")
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fsd_fetch, s, f): (s, f) for (s, f) in picked}
        for fut in as_completed(futs):
            s, f = futs[fut]
            r, label = picked[(s, f)]
            try:
                path = fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                log(f"  FAIL {e}")
                continue
            split = r["split"] if s == "dev" else "test"
            rows.append({
                "path": os.path.relpath(path, DATA).replace("\\", "/"), "source": "fsd50k",
                "id": f"{s}/{f}", "label": label, "split": split,
                "labels": r["labels"], "license": lic.get((s, f), ""),
            })
            done += 1
            if done % 250 == 0:
                log(f"  fsd50k {done}/{len(picked)} (failed {failed})")
    log(f"FSD50K done: {done} ok, {failed} failed")


def infrabel_batch(rows):
    """Add Infrabel's CC0 trackside recordings as train-positive clips.

    The released ZIP uses generic filenames and provides no per-file event
    labels. Each clip is therefore weakly labeled as railway-positive, while
    source, batch and original filename are retained for auditability.
    """
    archive = os.path.join(DATA, "infrabel", "Batch5.zip")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    if not os.path.exists(archive) or os.path.getsize(archive) == 0:
        log("Downloading Infrabel CC0 Batch5 ...")
        tmp_path = archive + ".download"
        with requests.get(INFRABEL_BATCH_URL, stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, archive)
    out_dir = os.path.join(CLIPS, "infrabel", "batch5")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".wav"))
    shuffled = names.copy()
    random.Random(505).shuffle(shuffled)
    n_test = max(1, round(len(names) * 0.1))
    n_val = max(1, round(len(names) * 0.1))
    split_by_name = {n: "test" for n in shuffled[:n_test]}
    split_by_name.update({n: "val" for n in shuffled[n_test:n_test + n_val]})
    split_by_name.update({n: "train" for n in shuffled[n_test + n_val:]})
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for name in names:
            original = os.path.basename(name)
            dest = os.path.join(out_dir, original)
            if not os.path.exists(dest):
                write_wav16(dest, decode_bytes(zf.read(name)))
            rows.append({
                "path": os.path.relpath(dest, DATA).replace("\\", "/"),
                "source": "infrabel_cc0", "id": f"Batch5/{original}",
                "label": 1, "split": split_by_name[name],
                "labels": "railway pass-by / rail sound (weak label)",
                "license": "CC0 1.0 Universal; Infrabel Train Sounds Batch5",
            })
            count += 1
    log(f"Infrabel Batch5 done: {count} CC0 railway-positive clips")


def tram_curving(rows):
    """Add CC-BY-4.0 tram pass-bys, holding out a collection day for evaluation.

    These field recordings capture trams traversing a curve, including rolling,
    flanging, and squeal. The dataset metadata's direction field is not a
    train/non-train label; all recordings document tram passages.
    """
    archive = os.path.join(DATA, "tram-curving.zip")
    metadata = os.path.join(META, "tram_datasets.csv")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    os.makedirs(META, exist_ok=True)
    if not os.path.exists(archive) or os.path.getsize(archive) == 0:
        log("Downloading Zenodo tram-curving audio (CC BY 4.0) ...")
        tmp_path = archive + ".download"
        with requests.get(TRAM_ARCHIVE_URL, stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, archive)
    if not os.path.exists(metadata) or os.path.getsize(metadata) == 0:
        log("Downloading Zenodo tram-curving metadata ...")
        r = requests.get(TRAM_META_URL, timeout=300)
        r.raise_for_status()
        with open(metadata, "wb") as f:
            f.write(r.content)

    with open(metadata, encoding="utf-8-sig", newline="") as f:
        meta_rows = list(csv.DictReader(f))
    out_dir = os.path.join(CLIPS, "tram_curving")
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for m in meta_rows:
            original = f"{m['Date']}.wav"
            if original not in names:
                raise RuntimeError(f"tram-curving metadata audio missing from archive: {original}")
            dest = os.path.join(out_dir, original)
            if not os.path.exists(dest):
                write_wav16(dest, decode_bytes(zf.read(original)))
            day = m["Date"][:10]
            split = "test" if day == "2023-08-12" else "train"
            labels = ("tram pass-by; rolling noise, flanging noise, squeal noise; "
                      f"direction={m['directional']}; weather={m['weather']}")
            rows.append({
                "path": os.path.relpath(dest, DATA).replace("\\", "/"),
                "source": "tram_curving_cc_by", "id": original,
                "label": 1, "split": split, "labels": labels,
                "license": "CC BY 4.0; Railway tram curving noise datasets, Zenodo 11540120",
            })
            count += 1
    log(f"Zenodo tram-curving done: {count} CC-BY-4.0 tram pass-bys; 2023-08-12 held out")


def main():
    rows = []
    os.makedirs(META, exist_ok=True)
    ensure_yamnet_weights()
    ensure_fsd_metadata()
    esc50(rows)
    fsd50k(rows)
    infrabel_batch(rows)
    tram_curving(rows)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "id", "label", "split", "labels", "license"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["path"]))
    for split in ("train", "val", "test"):
        s = [r for r in rows if r["split"] == split]
        log(f"{split}: {len(s)} clips, {sum(r['label'] for r in s)} positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
