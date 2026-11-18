# Developer and operator handoff

Baseline: **0.6.3 bell laboratory build**, tested on a C1111-E with an original T8351 external microphone. Intended next device: Q1656-LE with an external mic, awaiting onsite verification. The original coworker repository and local working directory are preserved separately.

## Installable files

Use the 0.6.3 `.eap` attached to the release. Keep the 0.6.2 intermodal `.eap` as rollback. SHA256 values are in the release's `SHA256SUMS.txt`. Both are aarch64 builds. Neither the P1367 nor all Axis cameras are implied to be supported.

In the device UI, go to Apps, add the `.eap`, start the app, and open its settings page. Configure the external microphone in the device audio UI first. The ACAP does not set microphone power, gain or camera security policy. For the Q1656-LE, complete [the target-camera checklist](Q1656-LE.md).

Do not apply a firmware update or change app-signing/security policy without the site's authorization. The current package is a development, unsigned build.

## Packaged laboratory settings

| Setting | Value |
|---|---|
| Enabled | yes |
| Sensitivity | 10 (general head threshold 0.869) |
| TriggerSeconds | 3 |
| SmoothingSeconds | 2 |
| ReleaseSeconds | 8 |
| CooldownSeconds | 10 |
| MaxEventSeconds | 600 |
| MinLevelDbfs | −90 |
| ScoreSource | head |
| HornTriggers | yes |
| HornSensitivity | 50 (threshold 0.331) |
| HornIncludesOtherHorns | yes |
| BellTriggers | yes (fixed threshold 0.869; five raw consecutive windows) |
| CrossingTriggers | no; dedicated crossing model unavailable |
| SpeechSuppression | 0 |
| RequireRailSound | no |
| MinRiseDb | 0 |

These intentionally permissive laboratory settings helped microphone replay testing. Existing camera parameters may survive an upgrade; check effective values in `status.cgi` rather than assuming package defaults were applied. Model conditioning and microphone gain are different: the current model conditions every analysis window internally, but the camera's analog/digital input settings still determine noise and clipping.

## Build and tests

From the repository root, with Docker running and Python 3 available:

```sh
python tools/build_release.py
```

The frozen model and parameter files are included. The build stages the correct model and class map, then produces `build/Train_Audio_Detection_0_6_3_aarch64.eap` using ACAP Native SDK 12.11.0. The 0.6.3 source/model snapshot is equivalent to the tested package, but timestamps/package metadata can cause a rebuilt EAP's hash to differ from the original.

For the host tests, also install Node.js, then run:

```sh
python tools/test_release.py
```

The tests cover general detector behavior, five-window bell eligibility, toggle/disable/gate handling, missing-hop reset, release/cooldown/cap, isolated real-model score fixtures and UI-preview parity. These fixtures are model outputs, not audio recordings.

Do not run the archived original training pipeline expecting it to recreate 0.6.3. See `model/README.md` for the self-contained frozen export and the limits of historical research scripts.

## Verify a device

After installation, inspect `/local/train_detect/status.cgi` using device authentication. Check the audio and inference health fields and `config.bell_available`. A bell score of zero with `bell_available=false` can mean an old one-probability model was packaged. General YAMNet Bell scores and the learned `scores.train_bell` are different signals.

`analyze.cgi` accepts mono PCM16 16 kHz WAVs and returns per-window train and bell scores. The default analyze mode does not inject events. `mode=inject` replaces microphone input and exercises events; reserve it for explicitly coordinated tests. The historical `tools/device.py verify` performs event/injection checks and assumes older fixture layouts; it is not the new handoff's unattended health check. Configure its ignored `.env.devices` file before using any legacy device tooling. No credential file is shipped.

Speaker/microphone playback already demonstrated a standalone bell start on the C1111-E for the reviewed freight clip. The other bell replay was recognized by both paths, with horn starting first. Neither replay is evidence of a field false-alarm rate. The UI can show generic horn evidence for a physical bell; inspect event attribution and the dedicated bell score.

## Preserve and evolve

Create a private repository owned by `mattlippincott-svg`, then share it with the intended developer through GitHub once their account is known. Do not add the coworker's repository as a push destination or overwrite its history. Tag the handoff baseline, attach the EAP/rollback assets, and make future changes on branches.

Keep onsite recordings and credentials outside Git. Store new labels with source/session/site separation, relevant audio settings and measured results. Capture quiet periods and non-train confusers as well as successful passages. Report false alarms per monitored time and detection delay rather than relying only on clip accuracy.

Priority follow-up work: Q1656-LE audio/runtime validation; longer independent locomotive-bell examples; US freight clacks/rolling sounds; ordinary bell and road-vehicle confusers; startup/reboot persistence and behavior under normal video/analytics load. Prime movers and pressure releases remain unreliable standalone train cues. Crossing detection requires a dedicated future model and must retain its own toggle.
