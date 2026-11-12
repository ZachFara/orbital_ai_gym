# 0006 Registered environment ids are immutable benchmarks

## Context

Published results (the 48/48 moon landing, the autopilot reference scores)
are only meaningful against a fixed environment. The roadmap includes rule
changes such as landing legs and a touchdown hold.

## Decision

- A registered id (`ToMoon-v0`, …) never changes physics, reward,
  observation layout or termination after it has been published with
  results. Bug fixes that change scores count as changes.
- Rule changes ship as a new version id (`-v1`) with the old version kept
  and still tested.
- Constructor knobs may be added with defaults that reproduce the
  registered behaviour exactly.

## Consequences

- Benchmark tables stay comparable over time.
- Some duplication between versions is accepted; shared code lives in the
  base classes and bodies, not in copied envs.
