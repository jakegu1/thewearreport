# Methodology

This page explains how every number on The Wear Report is made, how sure it is, and what it does
not show. The aggregate data is published in the engine repository. Changes are listed at the end.

## In short

- **Who is counted:** people seen on London's public traffic cameras (TfL JamCam) in daylight. The
  detector counts everyone it finds at least 31 pixels tall. Each sweep of the cameras is counted
  separately, so one person can appear in several sweeps.
- **What is measured automatically:** how many people are out at each temperature, and open
  umbrellas (umbrella figures are not shown until they have been checked in rain).
- **What is checked by hand:** whether a person's outermost layer is a coat or jacket, and whether
  their legs are bare. A person looks at crops of people at least 46 pixels tall and answers yes,
  no or "can't tell". On the site, "checked by hand" means judged yes or no. These answers are the
  only source of the coat and bare-legs percentages.
- **When a percentage appears:** only after at least 200 people have been checked by hand, in at
  least 10 spot checks on at least 5 different days, for that temperature band. Before that, pages
  show how many people have been checked so far.

## What we measure

For each sweep of London's traffic cameras we record:

- **People:** the people the detector finds that are at least 31 pixels tall in the camera's
  352×288 image (our "near-field" people; see the limits below).
- **Umbrellas:** open umbrellas the detector finds. The figure is umbrellas per person, not the
  share of people holding one, because two people can share an umbrella.
- **Weather:** temperature, feels-like temperature and precipitation from the Met Office hourly
  forecast for central London nearest the sweep.

From spot checks we record, for each person shown:

- **Outer layer:** yes if the outermost layer is a coat or jacket that opens at the front and has
  sleeves (overcoats, puffer jackets, raincoats, blazers, denim, leather and fleece jackets); no if
  it is a T-shirt, shirt, sweater, cardigan, hoodie (with or without a zip) or a sleeveless vest;
  "can't tell" otherwise.
- **Bare legs:** yes for shorts or a short skirt.

We do not measure anything finer (colours, brands, sleeve length). At this image size it is not
reliable.

## How the numbers are made

| Step | What happens |
|---|---|
| Cameras | The current list of TfL JamCam cameras is downloaded every sweep; only cameras TfL lists as available are used. |
| Fetch | The latest still from each camera is downloaded into memory. A camera that fails is counted and skipped. |
| Detect | YOLOX-m (Apache-2.0) runs on CPU and finds people and umbrellas. |
| Count | People at least 31 pixels tall are counted. Sweeps that start when the sun is more than 6° below the horizon are not counted. |
| Weather | Each sweep is joined to the Met Office hourly forecast nearest to it. |
| Spot checks | A person runs one pass over a current sweep of London cameras. Every person at least 46 pixels tall is shown, one crop at a time, in a window on their own computer, and is answered. Only the answers and each crop's height in pixels are kept. |
| Join | Each spot check takes the temperature of the counted sweep that started closest to it, within 20 minutes. Spot checks with no such sweep are left out. |
| Publish | Counts and answers are added up by temperature band. Only these totals are published. |

Since 4 October 2026, spot checks start between 09:30 and 15:00 London time, at most four a day and
at least 30 minutes apart. Four earlier pilot spot checks (30 September to 2 October 2026) started
between 09:32 and 17:21 London time and are included. Pages that show a percentage list how many
spot checks were made at each hour of the day.

## How sure are the percentages?

- Every number is shown with its sample size, date range and source.
- Coat and bare-legs percentages come with a 95% interval: a Wilson interval computed on an
  effective sample size that allows for differences between spot checks. From 4 October 2026,
  spot checks on the same London day are grouped together, because they share the same weather and
  the same crowd.
- When the intervals of two neighbouring temperature bands overlap, the page says the difference is
  not clear.
- Below the minimum (200 people, 10 spot checks, 5 days), no percentage is shown. Nothing is
  imputed, padded or estimated.

## What the percentages do and do not show

- **Only people whose clothing could be seen.** Many crops are too small or blurred to judge, and
  those people are left out. In London spot checks from 30 September to 4 October 2026, 69 of 175
  people (39%) could be judged for an outer layer. If people in coats were easier or harder to judge
  than others, the percentage would lean one way.
- **One person checks.** All answers so far come from one person following the written rules above.
  How consistently those rules are applied has not been measured yet; a repeat-crop check is planned
  before we publish any study.
- **Near the camera, on roads.** Traffic cameras watch roads, so people close to the camera are
  over-represented and distant people are missed. Detection depends on distance, not clothing, so
  this should change the counts more than the shares.
- **London only.** Clothing at a given temperature differs between cities and cultures. Every page
  says where its data comes from.
- **A forecast for one point.** The temperature is the Met Office hourly forecast for central
  London, not a measurement at each camera.

## How accurate are the counts?

- **Detection precision:** one person judged, box by box, whether each near-field box (at least 31
  pixels tall) was a person. In London, in three sessions on 28–29 September 2026 (two in daylight,
  one at dusk, none in rain), 207 of 208 judged boxes were people (99.5%; 95% lower bound 97.3%); 47
  more could not be judged. Target: at least 90%; below 85% raises an alert. We have not yet reached
  the 300 boxes, including rain, that we planned before launch. We plan to re-check 100 boxes a
  month (first by 31 October 2026) and after every change to the detector.
- **Umbrellas:** to be checked during rain (at least 30 umbrellas) before umbrella figures appear.
- **Automatic clothing labels:** from 30 September to 3 October 2026, the same crops were also sent
  to an open-weights vision model (Qwen3-VL, hosted by DeepInfra). On 132 crops a person could not
  judge, it still gave an answer on 97; of its "coat" answers checked against a person's, 9 of 14
  were right. It is not used for any published number.

## Privacy

1. Camera frames and person crops exist only in memory. They are never written to disk, logs,
   caches, CI artifacts, git or analytics services. Spot checks use the tool's window mode, which
   writes no file.
2. No face detection, face recognition, re-identification, or tracking of anyone across frames.
3. Only totals of counts and answers are published. The site never shows camera images or crops.
   To see a camera, follow the link to TfL's own camera page.
4. Spot checks run on the checker's own computer. For tests of automatic labels, crops of single
   people may be sent, in memory, to an open-weights vision model at a hosting provider whose terms
   exclude storing, logging or training on them. They may also be passed, in memory, to a small
   open-weights image model on the checker's computer, which keeps only its answers and running
   totals, never images or per-person image features. The spot-check tool refuses to run in CI.

CI enforces rule 1 for sweeps: a static check blocks image-writing calls in the engine, and an
end-to-end test runs a full sweep against a local fake camera server and checks that no image bytes
are left anywhere. The detector's runtime has its telemetry switched off.

## Sources and attribution

- Camera images: Transport for London JamCam feed. Powered by TfL Open Data.
- Weather (UK): Met Office Weather DataHub. Powered by Met Office data.
- Weather (US visitors, home page forecast): National Weather Service (public domain).
- Detector: YOLOX by Megvii, Apache-2.0.
- Data: sweep totals and spot-check answers are published in the engine repository.

## Changelog

- 2026-10 — Coat and bare-legs percentages come from hand spot checks only (one person, written
  rules, crops at least 46 pixels tall); display rule set to 200 people, 10 spot checks and 5 days.
  An automatic vision model was tested and is not used. Counts are described as detections per
  sweep; the weather is a forecast.
- 2026-09 — Detection precision measured by hand: 207 of 208 near-field boxes were people. Counts
  limited to people at least 31 pixels tall.
- 2026-09 — Detector switched from YOLOX-s to YOLOX-m after a same-frame comparison.
- 2026-09 — Method drafted; feasibility tested with two full sweeps.
