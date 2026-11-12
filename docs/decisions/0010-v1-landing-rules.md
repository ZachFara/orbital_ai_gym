# 0010 v1 landing rules: legs, contact limits and a touchdown hold

## Context

Under v0 a landing is one number at one instant: body-relative speed under
25 m/s when the rocket reaches the surface, and the episode ends there. A
lander arriving sideways at 20 m/s, or on its side, "lands". The owner
asked for landing legs (2026-11-12): a touchdown that a real lander would
survive, and proof that it stays down.

## Decision

- `TouchdownRules` (`orbital_gym/touchdown.py`), defaults from the owner:
  at contact the vertical speed is at most 5 m/s, the horizontal speed at
  most 3 m/s and the tilt off the local vertical at most 15 degrees, else
  the episode ends as `crashed` with `info["touchdown_limit"]` naming the
  limit. A contact inside the limits puts the rocket on its legs: it rests
  on the surface at its contact bearing, carried round with the tidally
  locked moon.
- The episode ends as a landing only after 5 s on the legs with the engine
  effectively off (thrust under 10 % of local weight). The engine coming on
  starts the hold over; thrust above local weight lifts off, free flight
  resumes and the clock resets. The site bonus is paid when the hold ends.
- Two observation channels are appended: on-the-legs (0/1) and hold
  progress (0..1). `info` gains `contact`, `hold_progress`,
  `touchdown_limit`.
- Speeds are measured against the moon's centre, as every other speed in
  the env is (the v0 limit, the instruments, the HUD); the ground's own
  rotation (80 m/s at the surface) is not a landing error. A lander at rest
  on the legs reads zero and lifts off with no sideways kick.
- The contact check runs every physics substep (0.1 s), not once per 1 s
  step: under the easy moon's 9.4 m/s² a step-end check would read a
  contact that happened 0.9 s earlier as a 9 m/s crash.
- `ToMoon-v1` is `ToMoonEnv(touchdown=TouchdownRules())`;
  `MoonLandingFromOrbit-v1` the practice id over it with its own expert
  bank. v0 ids are untouched (decision 0006); `touchdown=None` reproduces
  them exactly.
- The scripted pilots sit still on the legs (engine off, nose up) when
  `info["contact"]` is set, which never happens under v0.

## Consequences

- The v0 reference policy was trained on v0's single-speed rule; C2
  fine-tunes it under v1.
- Mars gets the same rules through the same knob when its v1 is defined;
  the limits may need their own numbers for a thin atmosphere.
- Gymnasium prints "out of date, consider v1" when `ToMoon-v0` or
  `MoonLandingFromOrbit-v0` is made. v0 stays the published benchmark; the
  warning is Gymnasium's convention, not a deprecation here.
