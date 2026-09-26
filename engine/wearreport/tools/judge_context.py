"""The judge context experiment: does a wider crop or a larger render help a hosted judge?

  python -m wearreport.tools.judge_context --variants m0.5,m1.0,m2.0,m1.0-r480
      --models di-qwen3-vl-235b,di-gemma-4-31b --max-requests N [--subset screen|all]

Each crop variant (VARIANTS) renders the gold set as the judge sees it, with another margin
around the box or another render target height; the degradation is unchanged, so the box
is still shrunk to the item's height_px and passed through JPEG at its jpeg_quality, and a
wider margin adds surroundings, never pixels on the person. `m0.5` is the baseline: the
spot-check tool's crop, byte for byte.

Every variant is run on every model named, through the DeepInfra backend of
`wearreport.tools.judge` (DeepInfraClassifier): the same prompt, the same gold-set-only
guard, and one request budget (`--max-requests`, required) for the whole run. For each
variant and model it prints the report of the T-033 bake-off (requests, tokens, cost,
accuracy on confident answers, unsure rate, precision error on the 80/90/95% mixes, the
height bands and the confusion table), then the change against `m0.5` for the same model.

The screen subset (the default) decides nothing: it names each variant and model whose
screen result is within SCREEN_ACCURACY_WITHIN of the accuracy bar with at most
SCREEN_MAX_UNSURE unsure, the only ones a full-set run may be spent on. With `--subset
all`, each is also scored on the held-out items, split by source photo
(`held_out_by_source`): the items whose photo has no item in the screening subset. Both
must pass the unchanged quality bar. Either way this is a diagnostic: it chooses no model.
The output is counts, tokens and costs only: no item, file, URL, image or request.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from wearreport.tools import goldset, judge


@dataclass(frozen=True, slots=True)
class Variant:
    """How much surrounds the box (`margin`, of its width and height, each side) and the
    height the crop is enlarged towards (`target_height`)."""

    name: str
    margin: float
    target_height: int


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant("m0.5", goldset.CROP_MARGIN, goldset.CROP_TARGET_HEIGHT),  # today's crop
        Variant("m1.0", 1.0, goldset.CROP_TARGET_HEIGHT),
        Variant("m2.0", 2.0, goldset.CROP_TARGET_HEIGHT),
        Variant("m1.0-r480", 1.0, 480),  # a larger enlargement only
    )
}
BASELINE = "m0.5"
SCREEN_ACCURACY_WITHIN = 0.05  # of MIN_ACCURACY, for a screen result to earn a full-set run
SCREEN_MAX_UNSURE = 0.15
DIAGNOSTIC = "diagnostic only: no model is chosen"


def held_out_by_source(manifest: goldset.Manifest) -> tuple[goldset.Item, ...]:
    """The gold-set items whose source photo has no item in the screening subset, so no
    held-out item shares a scene with an item the prompt may be tuned on."""
    screen = {item.source for item in goldset.screening_subset(manifest)}
    return tuple(item for item in manifest.items if item.source not in screen)


def eligible(scores: judge.Scores) -> bool:
    """Whether a screen result earns a full-set run: within SCREEN_ACCURACY_WITHIN of the
    accuracy bar on confident answers, with at most SCREEN_MAX_UNSURE unsure."""
    within = round(100 * (judge.MIN_ACCURACY - SCREEN_ACCURACY_WITHIN), 6)
    return (
        scores.n > 0
        and round(100 * scores.accuracy, 6) >= within
        and scores.unsure_rate <= SCREEN_MAX_UNSURE
    )


def gold_crops(variant: Variant, subset: str) -> judge.Crops:
    """The gold set (or its screening subset) rendered with `variant`, each crop marked as
    licensed, with its height and whether it is held out by source photo."""

    def crops() -> Iterator[judge.Crop]:
        manifest = goldset.load_manifest()
        held = {item.id for item in held_out_by_source(manifest)}
        items = goldset.screening_subset(manifest) if subset == "screen" else manifest.items
        for item, image in goldset.iter_gold(
            manifest, items=items, margin=variant.margin, target_height=variant.target_height
        ):
            # A crop of the licensed gold set: it may be sent to a hosted model.
            yield item.label, judge.mark_licensed(image), item.height_px, item.id in held

    return crops


def _pick[T](text: str, registry: dict[str, T], what: str) -> list[T]:
    """The registry entries named in a comma-separated list, each once; raises ValueError
    for a name that is not a key of the registry (a path, a URL or a typo)."""
    names = list(dict.fromkeys(n.strip() for n in text.split(",") if n.strip()))
    if not names:
        raise ValueError(f"no {what} named")
    for name in names:
        if name not in registry:
            raise ValueError(f"unknown {what} {name[:40]!r} (known: {', '.join(registry)})")
    return [registry[n] for n in names]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m wearreport.tools.judge_context",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variants", required=True, help=f"comma-separated, of: {', '.join(VARIANTS)}")
    ap.add_argument(
        "--models", required=True, help=f"comma-separated, of: {', '.join(judge.DEEPINFRA)}"
    )
    ap.add_argument("--subset", choices=("screen", "all"), default="screen")
    ap.add_argument(
        "--max-requests",
        type=int,
        required=True,
        help="stop after N requests in all, every variant and model together (retries count)",
    )
    return ap


def _points(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100 * x:+.1f}"


def _change(variant: Variant, name: str, run: judge.Run, base: judge.Run | None) -> str:
    """The change of a variant's result against the baseline's, for the same model."""
    head = f"{variant.name} / {name}"
    if base is None or base.error is not None or not base.answers:
        return f"{head}: no {BASELINE} run to compare with"
    if run.error is not None or not run.answers:
        return f"{head} against {BASELINE}: n/a (this run did not finish)"
    now, then = judge.score(run.answers), judge.score(base.answers)
    errors = ", ".join(
        f"{mix:.0%} "
        + (
            "n/a"
            if math.isnan(now.precision_error[mix]) or math.isnan(then.precision_error[mix])
            else f"{now.precision_error[mix] - then.precision_error[mix]:+.1f}"
        )
        for mix in judge.MIXES
    )
    return (
        f"{head} against {BASELINE}: accuracy on confident answers "
        f"{_points(now.accuracy - then.accuracy)} points, unsure "
        f"{_points(now.unsure_rate - then.unsure_rate)} points, precision error {errors} points"
    )


def _report(variant: Variant, run: judge.Run, subset: str) -> list[str]:
    title = f"{variant.name} / {judge._title(run.name)}"
    setting = (
        f"crop variant {variant.name}: margin {variant.margin:g} of the box on each side, "
        f"render target {variant.target_height} px"
    )
    if run.error is not None:
        head = f"stopped after {len(run.answers)} crops" if run.answers else "not run"
        return [f"== {title}", setting, f"{head}: {run.error}", *judge._usage(run)]
    scores = judge.score(run.answers)
    lines = judge.report(title, scores, run.seconds)
    lines[1:1] = [setting]
    lines += judge._usage(run) + judge._by_height(run)
    per_crop = run.seconds / scores.n if scores.n else math.inf
    if subset == "screen":
        verdict = "eligible" if eligible(scores) else "not eligible"
        lines.append(
            f"{variant.name} / {run.name}: {verdict} for a full-set run (within "
            f"{100 * SCREEN_ACCURACY_WITHIN:g} points of {judge.MIN_ACCURACY:.0%} on "
            f"confident answers, at most {SCREEN_MAX_UNSURE:.0%} unsure)"
        )
        return lines
    held = judge._held_out(run)
    held_passes = False
    if held:
        held_scores = judge.score(held)
        lines += judge.report(
            f"{variant.name} / {run.name}, held out (by source photo)",
            held_scores,
            per_crop * len(held),
        )
        held_passes = judge.passes(held_scores, per_crop)
    both = scores.n > 0 and judge.passes(scores, per_crop) and held_passes
    lines.append(
        f"{variant.name} / {run.name}: full set and held-out items: "
        + ("PASS" if both else "FAIL")
        + ("" if held else " (no held-out items)")
    )
    return lines


def _total(runs: Sequence[judge.Run]) -> str:
    metered = [r for r in runs if r.usage is not None]
    cost = sum(
        judge.cost_usd(h, r.usage.input_tokens, r.usage.output_tokens)
        for r in metered
        if r.usage is not None and (h := judge.priced(r.name)) is not None
    )
    missing = sum(r.usage.missing for r in metered if r.usage is not None)
    return (
        f"requests made {sum(r.usage.requests for r in metered if r.usage is not None)}, "
        f"input tokens {sum(r.usage.input_tokens for r in metered if r.usage is not None)}, "
        f"output tokens {sum(r.usage.output_tokens for r in metered if r.usage is not None)}, "
        f"cost ${cost:.4f}"
        + (f" (a lower bound: usage missing from {missing} replies)" if missing else "")
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    endpoint: str | None = None,
    crops: Callable[[Variant], judge.Crops] | None = None,
) -> int:
    """Run the experiment. `crops` replaces the rendered gold set of a variant and
    `endpoint` the DeepInfra origin (tests)."""
    args = build_parser().parse_args(argv)
    if args.max_requests < 1:
        print("judge_context: --max-requests must be at least 1", file=sys.stderr)
        return 2
    try:
        variants = _pick(args.variants, VARIANTS, "variant")
        models = _pick(args.models, judge.DEEPINFRA, "model")
        manifest = goldset.load_manifest()
    except (ValueError, goldset.GoldsetError) as exc:
        print(f"judge_context: {exc}", file=sys.stderr)
        return 2

    def emit(line: str) -> None:
        print(line, flush=True)

    held = held_out_by_source(manifest)
    moved = len(judge.held_out(manifest)) - len(held)
    emit(DIAGNOSTIC)
    emit(
        f"held-out items by source photo: {len(held)} of {len(manifest.items)}; "
        f"{moved} items move out of the held-out set (they share a source photo with an "
        "item of the screening subset)"
    )
    budget = judge.RequestBudget(args.max_requests)
    source = crops or (lambda variant: gold_crops(variant, args.subset))
    runs: dict[tuple[str, str], judge.Run] = {}
    for variant in variants:
        variant_crops = source(variant)
        for model in models:

            def open_classifier(model: judge.HostedCandidate = model) -> judge.Classifier:
                return judge.DeepInfraClassifier(model, budget=budget, endpoint=endpoint)

            run = judge.evaluate(model.name, open_classifier, variant_crops)
            runs[variant.name, model.name] = run
            for line in _report(variant, run, args.subset):
                emit(line)

    emit(f"== change against {BASELINE}, the current crop, for the same model")
    for variant in variants:
        for model in models:
            if variant.name != BASELINE:
                base = runs.get((BASELINE, model.name))
                emit(_change(variant, model.name, runs[variant.name, model.name], base))
    emit(_total(list(runs.values())))
    if args.subset == "screen":
        emit("screen only: no pass decision (a full-set run needs --subset all)")
    emit(DIAGNOSTIC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
