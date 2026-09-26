"""The judge bake-off on vision models hosted by DeepInfra (`judge.DEEPINFRA`).

  python -m wearreport.tools.judge_deepinfra --bakeoff --max-requests N
      [--models A,B] [--subset all|screen] [--limit N]

The same harness, gold set, quality bar, held-out rule and cost report as
`python -m wearreport.tools.judge --bakeoff --backend bedrock`; see `wearreport.tools.judge`.
"""

from __future__ import annotations

from wearreport.tools.judge import deepinfra_main

if __name__ == "__main__":
    raise SystemExit(deepinfra_main())
