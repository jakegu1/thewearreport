# The Wear Report — engine

> Pre-release: the pipeline is being built and nothing is published yet. Site: thewearreport.com (not live yet).

An open pipeline that measures **what people actually wear outside**. It uses public
traffic cameras and publishes **aggregate statistics only**, for example "at 12 °C,
71% of people seen wore a coat (n = 4,210)". It never stores images, never identifies
anyone and never tracks individuals.

## How it works

1. **Fetch.** During daytime, fetch the latest still frame from each of London's
   ~800 TfL JamCam traffic cameras. Frames are held in memory only.
2. **Detect.** Find people and umbrellas on CPU with
   [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) via ONNX Runtime.
3. **Classify.** Classify a few coarse attributes per detected person: outer layer
   (coat or jacket) yes/no, bare legs yes/no. Nothing finer than that.
4. **Aggregate.** Combine the counts with the current weather and publish per-sweep and
   daily aggregates. Images and crops are discarded as soon as they are processed.

The full method, including its known biases and limits, is in
[METHODOLOGY.md](./METHODOLOGY.md).

## Privacy

- Frames and person crops exist only in memory and are never written to disk, logs,
  caches, artifacts or git.
- No face detection, face recognition, re-identification or tracking across frames.
- Only aggregate counts are published.

[METHODOLOGY.md § Privacy](./METHODOLOGY.md#privacy) describes the rules in full. CI
enforces them: a static check blocks image-writing calls, and an end-to-end test
asserts that a sweep creates no files except its aggregate output.

## Data

Aggregates will be published on the `data` branch as JSON Lines, with a JSON Schema.
The data license is in [DATA-LICENSE.md](./DATA-LICENSE.md).

Powered by TfL Open Data. Contains Met Office data. US weather from the National
Weather Service.

## Repository layout

```
engine/      Python package: registry, fetch, detect, attributes, aggregate, publish
spike/       Verified reference implementation (read, don't import)
tools/       Repository checks (public guard)
.github/     CI and the scheduled sweep
```

## License

Code: [Apache-2.0](./LICENSE). Data: see [DATA-LICENSE.md](./DATA-LICENSE.md).

## Contributing

This project is maintained by one person and is not accepting pull requests yet.
To report a security or privacy problem, use GitHub's private vulnerability reporting
(see [SECURITY.md](./SECURITY.md)).
