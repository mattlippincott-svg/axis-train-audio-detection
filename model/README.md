# Model development

The frozen 0.6.3 model and its export parameters are in `releases/0.6.3/`. `manifest.json` records hashes; `traindet.tflite` is the exact model checked on the C1111-E. The trained general head, pretrained YAMNet weights, and numeric expert coefficients are included so another developer can rebuild the graph without the old machine or private recordings.

Input: float32 `[1,15600]`, mono 16 kHz. Output tensors are float32 `[1,2]` (unchanged train probability, learned bell-informed probability, in that order) and `[1,521]` (original YAMNet tags). The loader also supports old one-probability models, with bell availability false.

Input conditioning is boost-only toward −26 dBFS, capped at 36 dB gain and 0.98 peak. The general head is combined with a reviewed rhythmic-rail expert. The bell expert stays separate and uses the five-window trigger in the C detector; the application does not simply merge its score into the old train head.

## Re-export the frozen model

From the repository root:

```sh
docker build -t traindet-handoff-ml model
docker run --rm -v "${PWD}:/work" -w /work traindet-handoff-ml python model/export_frozen.py
```

The export and parity report go to `build/model/`. Export checks fixed test vectors against the shipped TFLite graph. Graph serialization can differ while equivalent outputs pass parity. The regular app build uses the immutable shipped model; replacing it with a newly trained/exported candidate should be an explicit versioned change.

## Further training

Read `training/README.md` and `training/model-provenance.json`. Raw datasets, audio previews, captures, feature caches and the original training environment's experiment folders are excluded. The research scripts are preserved in `research-scripts/`, but they use historical relative experiment paths and `/lab` capture mounts and are not a one-command reproduction of the full training history.

The frozen export above is self-contained. Full refitting is a separate undertaking: reacquire and review data using the recorded source IDs and licenses, preserve source-family exclusions, rebuild feature caches, and keep validation/test audio outside fitting. Do not call fitted freight excerpts independent test evidence. Do not treat bell-absent train sounds as generic non-train negatives.
