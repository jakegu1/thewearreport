# Methodology

> **Draft.** This describes the method as designed. Measured accuracy figures will be
> added once the engine has run in production. Changes are listed at the end.

## What we measure

For each sweep of London's public traffic cameras we count:

- people detected
- people with an outer layer (coat or jacket), or without one
- people with bare legs (shorts or a short skirt), or without
- open umbrellas

Each sweep is tagged with the weather at that time: temperature, feels-like temperature
and precipitation. Aggregating many sweeps gives the share of people wearing each item at
each temperature, for example "at 12 °C, 71% wore a coat".

We do **not** measure anything finer than these coarse attributes (sleeve length, colours,
brands). At 352×288 pixels they are not reliable, and the product does not need them.

## Pipeline

| Step | What happens | Notes |
|---|---|---|
| Camera registry | Download the current list of TfL JamCam cameras | Refreshed every sweep; only cameras TfL lists as available are used (787 in a September 2026 test) |
| Fetch | Download the latest still from each camera | In memory only; one attempt per camera with a hard time limit; a failed camera is counted and skipped, never retried; only complete JPEG images are decoded |
| Detect | YOLOX-m (Apache-2.0) on CPU via ONNX Runtime; classes *person* and *umbrella* | About 190 ms per frame on a 4-core machine; chosen over YOLOX-s, which found about a quarter fewer people on the same frames |
| Classify | Coarse attributes per person crop | Method chosen in a later milestone; see Changelog |
| Aggregate | Counts per sweep, joined to weather | Published on the `data` branch as one JSON record per sweep (concatenates into JSON Lines) |

## Privacy

1. Camera frames and person crops exist only in memory. They are never written to
   disk, logs, caches, CI artifacts, git or analytics services.
2. No face detection, face recognition, re-identification, or tracking of anyone
   across frames.
3. Only aggregate counts are published. The site never shows camera images or crops.
   To see a camera, follow the link to TfL's own camera page.
4. If a vision model is used for attribute labelling, it receives person crops only
   (never full frames), from a provider with data-processing terms, within a fixed
   monthly budget, and only the resulting labels are kept.
5. Accuracy checks are done by hand, live, on a sample of current frames. A person views the
   crops in a window on their own computer; nothing is saved, and only the tallies and each
   box's height in pixels are kept. Before launch at least 300 boxes are checked, across
   daylight, dusk and rain; after that, 100 a month and after every change to the detector
   or its thresholds. For one-off tests of automatic clothing labels, the same crops of
   single people may be sent, in memory, to an open-weights vision model at a hosting
   provider whose terms exclude storing, logging or training on them. The spot-check tool
   refuses to run in CI.

CI enforces rule 1 in two ways. A static check blocks image-writing calls in `engine/`.
An end-to-end test runs a full sweep against a local fake camera server and checks that
it creates no files except the aggregate output and leaves no image bytes in its working,
home or temporary directories. Deliberate leaks in the test suite prove that these checks
catch them. The detector's runtime library (ONNX Runtime) has its built-in telemetry
switched off, and a test checks that loading it writes nothing.

## Sampling and known biases

- **Cameras watch roads, not pavements.** In early tests about 40% of cameras showed at
  least one person and 5–7% showed five or more. People close to the camera are
  over-represented.
- **Distant people are missed.** In a manual check of one frame every detection was a
  real person, but about half of the visible (mostly distant) people were not detected.
  That check used the earlier YOLOX-s model; YOLOX-m, used since September 2026, detects
  30–40% more people on the same frames, and its precision is measured by the spot-checks.
  This lowers the count but should not bias the share wearing a coat, because detection
  depends on distance, not on clothing. We will test this assumption with spot-checks.
- **One city.** London's population is not everyone's. Clothing norms at a given
  temperature differ between cultures and climates. Pages say where the data comes from.
- **Near-field people only.** We count and classify only people whose box is at least
  31 pixels tall in a 352×288 image: the smallest size at which most boxes can be checked
  by eye and at least 90% of them are people. Smaller boxes are ignored. The threshold is
  recalculated when the hand checks grow.
- **Daytime only.** Sweeps run in London daytime, and only sweeps that start while the sun
  is no more than 6° below the horizon are counted.
- **Weather is a forecast for one point in central London** (the Met Office hourly time
  step nearest the sweep), not a measurement at each camera.

## Uncertainty and minimum sample

- Every percentage is shown with its sample size `n`, time range and source.
- Every percentage has a 95% confidence interval (Wilson score interval).
- Below `n = 200`, pages show "collecting data (n observed so far)" instead of a
  percentage. Nothing is imputed, padded or estimated.

## Validation

- **Detection precision:** at least 300 boxes checked by hand before launch, then 100 a
  month and after every change to the detector. Target ≥ 90% for near-field boxes; below
  85% raises an alert.
- **Attribute accuracy:** at least 200 near-field people compared with human labels.
  Target ≥ 85% per attribute before an attribute is published.
- **Umbrellas:** validated during rain (at least 30 umbrellas).

Results will be published here with dates and sample sizes.

## Sources and attribution

- Camera images: Transport for London JamCam feed. Powered by TfL Open Data.
- Weather (UK): Met Office Weather DataHub. Powered by Met Office data.
- Weather (US visitors): National Weather Service (public domain).
- Detector: YOLOX by Megvii, Apache-2.0.

## Changelog

- 2026-09 — Method drafted; feasibility tested with two full sweeps (822 and 931 people).
- 2026-09 — Fetcher and weather adapter built. Facts updated: one attempt per camera per
  sweep (no retries), feels-like temperature recorded, weather point described exactly.
- 2026-09 — Detector switched from YOLOX-s to YOLOX-m after a same-frame comparison on 200
  live frames (30–40% more people detected, about 190 ms per frame).
- 2026-09 — Accuracy checks redesigned: an open-weights vision model on the same machine
  judges detection crops in memory, with known-answer crops mixed in as controls. The
  manual mode remains.
- 2026-09 — Accuracy checks are done by hand: the automatic judge did not reach the required
  accuracy (it never recognised a box that was not a person). Counts are limited to
  near-field people.
