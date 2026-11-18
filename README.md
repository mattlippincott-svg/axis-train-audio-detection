# Axis train audio detection — development handoff

Version **0.6.3**, a development ACAP that recognizes train-related audio from an Axis device's microphone and raises train-present events. It adds a dedicated short locomotive-bell path to the earlier horn and rolling-stock model.

This is a separate handoff snapshot derived from [mrdnlabs/axis-audio-train-detection](https://github.com/mrdnlabs/axis-audio-train-detection), base commit `23d1076f91aab4b486068ead36a2fca56431cf29`. Original attribution and licenses are retained. The original working directory and Git history were preserved; this snapshot does not include that history or its private development environment.

## Start here

- **Install and operate:** [handoff guide](docs/HANDOFF.md).
- **Target camera:** [Q1656-LE compatibility and onsite checks](docs/Q1656-LE.md).
- **Build:** Python 3, Docker, and `python tools/build_release.py`. This uses the bundled frozen model; no training data or GPU is required.
- **Tests:** Python 3, Docker, Node.js, and `python tools/test_release.py`.
- **Model export and further training:** [model notes](model/README.md) and [reviewed labels/provenance](training/README.md).
- **Older investigations:** `docs/archive/` and `model/research-scripts/` are historical development material, not the current quick-start pipeline.

## What is verified

The aarch64 package was installed and checked on a C1111-E with an external T8351 microphone. Model export parity, actual-device analyze parity, host detector tests, UI-preview replay tests, and speaker/microphone bell replays passed. In the freight replay, the new bell path initiated the event before horn evidence qualified.

**Q1656-LE has not been tested with this build.** Its ARTPEC-8/aarch64 platform and external audio input are suitable for the current package, subject to the actual firmware and onsite audio/performance checks. This is not field false-alarm validation.

## Current behavior

- General train probability and all 521 YAMNet sound-tag outputs are unchanged from 0.6.2.
- Bell triggering uses a separate learned score: five consecutive windows at least 0.869, approximately a two-second hold. `BellTriggers` turns it on/off. It is not driven directly by generic Bell tags.
- Horn triggering is enabled, including other horns without a rail-context requirement.
- Electronic crossing triggering remains separate, unavailable in this model, and off.
- Bell-triggered events are identified as bells in history, status and CSV output. Horn evidence can be added later to the same event.

The packaged laboratory settings are documented in [HANDOFF.md](docs/HANDOFF.md). They are starting settings for tuning, not proven defaults for every location.

## Files and ownership

Source is MIT-licensed with third-party components and notices retained. The model includes material trained on non-commercial datasets; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). No source audio, microphone recordings, device addresses, passwords or raw training caches are included. The repository should begin private, with the installable `.eap` and rollback attached as release assets rather than committed as build products.
