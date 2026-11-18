# Labels, provenance and limits

`model-provenance.json` summarizes the frozen rhythmic expert and bell expert. `labels/` preserves human listening decisions and source IDs. The original source metadata in historical labels can describe a pre-fit review state; the shipped 0.6.3 experts did fit the approved training references described here.

Bell fitting uses source 28777 (Corsica_S, original confirmed bell clip) and **only seconds 9–12** of source 854498 (serøutōnin--deprivəd, confirmed clear bell). Horn-affected portions of the latter are excluded as bell positives. The reviewed **0–30 s** of source 752578 (lakewoodsound) has no audible bell and is used as a contrast for the bell expert only. Its unreviewed remainder is not labelled. This recording is never promoted to a generic non-train negative.

Source 127451 (arfei, GO Train bell) stayed outside fitting. The earlier microphone replay is development comparison material, not a pristine independent-site test. It is not included here. Source 737301 (WeenyBeany, processed electronic crossing bell) is a separate development challenge and did not fit the bell expert.

The rhythmic rail expert uses eight reviewed source families, including the 35–85 s intermodal YouTube reference. Flat-spot-like wheel thumps and joint click-clacks share the audible rhythmic-rail target; no mechanical defect diagnosis is claimed. Squeal and general wheel-on-rail sounds are useful context, but the inherited general head does not guarantee that a squeal alone can never trigger.

Published development validation was reused during calibration. The short bell path added six positive clip/version detections over the current general path, with the same six negative detections in that reused set. This is neither a field false-alarm rate nor new independent acceptance evidence. The bell expert has only two positive recording families, and the original whole-clip labels include unlabelled overlaps.

Future priorities: capture representative site backgrounds and train passages; record camera/mic/firmware settings; label train-present intervals and cue types; retain confusers such as road trucks, horns, engines, ordinary bells, speech, weather and mechanical impacts; split by recording session/site; evaluate detection delay, misses and false alarms per monitored time. Pressure releases and prime movers should not become unconditional triggers on the basis of the current data.
