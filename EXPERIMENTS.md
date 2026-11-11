# Moon-landing experiment log

Every attempt is an MLflow run in `mlflow.db` (experiment `moon-landing`);
`python scripts/explog.py` prints the table, `mlflow ui --backend-store-uri
sqlite:///mlflow.db` opens the dashboard. This file mirrors each finished run as a
short entry: hypothesis, outcome, diagnosis. Newest at the bottom.

## Prior attempts (before this log existed, reconstructed from the code)

- **train_ppo.py / train_guided.py (PPO, BC + JSRL prefix).** Reached orbit;
  on the moon mission the policy exploited the duration reward by loitering
  (fixed in the env), then a BC clone of the direct-descent autopilot arrived
  at ~450 m/s: the autopilot rides the suicide-burn boundary with no margin,
  so the landing is unlearnable by imitation plus small-noise exploration.
- **train_her.py (TD3+BC + HER, residual, orbit-first).** Documents a long
  chain of fixes: critic collapse, BC anchor drift, escape exploit when only
  crashes were penalised, dense per-step level reward wrecking landings, a
  60:1 reward-scale ratio. The orbit-first setup (park in low moon orbit, then
  learn deorbit + descent with hover-relative throttle) was reached but no
  trained model survives (models/ is empty after the machine reorganisation),
  so its result is unknown.

## Baseline measurements (2026-11-10)

| flight | end | rel speed | miss | dv used | steps |
|---|---|---|---|---|---|
| direct autopilot | landed_moon | 8.7 m/s | 285 km | 6523 m/s | 2956 |
| direct + divert | landed_moon | 16.7 m/s | 301 km | 8170 m/s | 2675 |
| orbit-first, site 0 / 90 / 180 / -90 deg | landed_moon_site x4 | ~1 m/s | 6-7 km | ~8400 m/s | 5200-6200 |

The orbit-first autopilot solves the whole mission for any site. The RL task
worth posing is therefore the segment after parking-orbit entry: when to
deorbit and how to fly the powered descent to a random site. The env steps at
~18k steps/s per process.

## 2026-11-10 23:11 - smoke  (`966dbf53`, FINISHED)

**Hypothesis:** smoke test of the pipeline

**Outcome:** level reached 0.00; final on-site 0%, soft 0%, crash 0%, median miss 3065.9 km, 61,440 steps in 0 min

**Diagnosis:** (fill in with `mlflow` tag edit or explog) see curriculum level history: 

**Key metrics:** final/level=0, final/on_site=0, final/soft=0, final/crash=0, final/timeout=1, final/miss_km_med=3.07e+03, final/rel_med=2.33e+03, final/dv_mean=3.94e+03, final/native=-471, final/full_mission=2.01e+03

## 2026-11-10 23:25 - ppo-curriculum-v1  (`b782a9be`, KILLED)

**Hypothesis:** PPO from scratch with a reachable-state curriculum (expert states, ttl-capped by level), moon-frame actions, hover throttle, difference-form shaping on site distance + speed, terminal +10 on-site / graded crash. Expect the curriculum to climb from touchdown to full descent; the open question is whether deorbit timing (level > 0.5) is learnable by pure RL.

**Outcome:** killed at ~1M steps: curriculum never left level 0 (on-site rate flat at 55-65%, crash 10-18%, episodes 3-4 steps). Throughput 7.4k steps/s with 8 envs.

**Diagnosis:** the level-0 start states are the expert's last ~15 steps, but the expert itself touches down 6-8.5 km from the commanded site (site radius 8 km; two of 24 sites it misses outright), so a large share of level-0 episodes are off-site before the policy acts and the 80% on-site threshold is unreachable by construction. Fix: center the site on the expert's ACTUAL touchdown bearing for descent-phase start states (site noise then grows with level).

## 2026-11-10 23:50 - ppo-curriculum-v2  (`f01fd90d`, KILLED)

**Hypothesis:** v1 with the bank fix: descent-phase start states aim at the expert's actual touchdown bearing (site noise grows with level), so level 0 is achievable. Same PPO + reachable-state curriculum.

**Outcome:** killed at ~4.5M steps: passed level 0 immediately, then stuck at level 0.05 (ttl<=24) with on-site 60-69%, crash 30-40%; policy std only fell 0.34 -> 0.33

**Diagnosis:** two things. (1) The env's hover_control throttle is hover*exp(3*a0): PPO's exploration std 0.33 on a0 is +-1 g of random vertical acceleration per second, so stochastic rollouts near the ground crash regardless of the mean policy, and the deterministic policy learned to hover-oscillate at 0-15 m rather than commit (traces: alt bouncing 0.2 -> 12 m for hundreds of steps, eventual 40-95 m/s impact). (2) A quarter of the expert's ttl<=20 states are still 0.5-1 km up falling at 150-190 m/s (its descent is a vertical dive with a last-second suicide burn), which need immediate full throttle; those are legitimately hard but learnable. Fix for (1): a gentler throttle map (linear below hover, quadratic above) and smaller/smoother exploration (log_std_init -2, optionally gSDE).

## 2026-11-10 23:22 - abl-exp-std1  (`53d3c225`, FINISHED)

**Hypothesis:** ablation @ fixed level 0.1: env's exp throttle, std 0.37 (the v2 setting) - control

**Outcome:** level reached 0.10; final on-site 3%, soft 16%, crash 84%, median miss 206.0 km, 1,507,328 steps in 3 min

**Diagnosis:** (fill in with `mlflow` tag edit or explog) see curriculum level history: 

**Key metrics:** final/level=0.1, final/on_site=0.0312, final/soft=0.156, final/crash=0.844, final/timeout=0, final/miss_km_med=206, final/rel_med=219, final/dv_mean=3.43e+03, final/native=-29.7, final/full_mission=2.45e+03

## 2026-11-10 23:25 - abl-lin-std1  (`3b289efa`, FINISHED)

**Hypothesis:** ablation @ fixed level 0.1: linear/quadratic throttle map, std 0.37

**Outcome:** level reached 0.10; final on-site 0%, soft 0%, crash 100%, median miss 213.8 km, 1,507,328 steps in 3 min

**Diagnosis:** (fill in with `mlflow` tag edit or explog) see curriculum level history: 

**Key metrics:** final/level=0.1, final/on_site=0, final/soft=0, final/crash=1, final/timeout=0, final/miss_km_med=214, final/rel_med=1.47e+03, final/dv_mean=3.39e+03, final/native=-92.9, final/full_mission=2.39e+03

## 2026-11-10 23:29 - abl-lin-std2  (`c54a63c9`, FINISHED)

**Hypothesis:** ablation @ fixed level 0.1: lin throttle map, std 0.135

**Outcome:** level reached 0.10; final on-site 0%, soft 0%, crash 0%, median miss 845.0 km, 1,507,328 steps in 3 min

**Diagnosis:** (fill in with `mlflow` tag edit or explog) see curriculum level history: 

**Key metrics:** final/level=0.1, final/on_site=0, final/soft=0, final/crash=0, final/timeout=0, final/miss_km_med=845, final/rel_med=3.23e+03, final/dv_mean=3.94e+03, final/native=-105, final/full_mission=2.37e+03

## 2026-11-10 23:32 - abl-lin-sde  (`9257aa6d`, FINISHED)

**Hypothesis:** ablation @ fixed level 0.1: lin throttle map, gSDE (smooth exploration, resampled every 8 steps)

**Outcome:** level reached 0.10; final on-site 0%, soft 0%, crash 0%, median miss 269.9 km, 1,507,328 steps in 3 min

**Diagnosis:** (fill in with `mlflow` tag edit or explog) see curriculum level history: 

**Key metrics:** final/level=0.1, final/on_site=0, final/soft=0, final/crash=0, final/timeout=1, final/miss_km_med=270, final/rel_med=1.06e+03, final/dv_mean=0, final/native=-3.81, final/full_mission=2.48e+03

## Ablation 1 (fixed curriculum level 0.1, 1.5M steps each): throttle map x exploration

| run | throttle | log_std_init | on-site (train, end) | crash |
|---|---|---|---|---|
| abl-exp-std1 | env exp (K=3) | -1.0 | **92%** | 6% |
| abl-lin-std1 | lin/quad | -1.0 | 81% | 12% |
| abl-lin-std2 | lin/quad | -2.0 | 62% | 29% |
| abl-lin-sde | lin/quad | -1.5 + gSDE/8 | 1% | 79% |

Reading: the v2 diagnosis (exponential throttle gain too sensitive to PPO
noise) was wrong, or at least not the binding constraint. Larger exploration
noise learns faster here; gSDE's slowly varying noise just produces long
dithering episodes. The v2 stall at level 0.05 remains unexplained by this
ablation; v3 simply starts the curriculum at level 0.1 with the control
settings. Note the automatic "final" eval in these four runs is the
from-orbit eval and is meaningless at level 0.1.

## Guidance channels validated (2026-11-11)

`conic_impact()` in `orbital_gym/rl/moon_landing.py` predicts, from the
moon-relative state alone, whether the coasting path reaches the surface,
when, and at what speed (radial/tangential). Checked against brute-force
engine-off coasting in the simulator across bank states from 5 to 400 steps
before touchdown: time to impact within one step, impact speed within ~1%
(the residual is the planet's tide over long coasts). The expert's own
dive states have a full-thrust stop ratio of only 0.13-0.41, i.e. the
"knife-edge" landing was a controller property, not a physics one.

## 2026-11-11 - ppo-curriculum-v3  (`1a8460fc`, KILLED as baseline)

**Hypothesis:** full curriculum with the ablation-1 winner (env exp throttle, log_std -1), starting at level 0.1 (skipping the level 0-0.05 states where v1/v2 stalled). Expect levels to advance every ~1-2M steps; watch whether progress stalls where the expert's dive states (ttl 30-120) or the deorbit-timing states (level > 0.5) enter.

**Outcome:** stopped at 9.6M steps as the baseline for v4. Curriculum: 8k:0.10, 1040k:0.15, 1302k:0.20, 1785k:0.25, 2220k:0.30, 2768k:0.35, 3522k:0.40, 6848k:0.45, 9797k:0.50. Soft-landing rate 95-99% from level 0.3 on (the knife-edge touchdown is learned), but on-site rate plateaued at 60-71% at levels 0.4-0.45 for 6M steps; from-orbit eval never left the parking orbit (0% soft).

**Diagnosis:** lateral site precision is the bottleneck for the base observation: mean miss 6.4-6.9 km against an 8 km radius once the site offset noise reaches ~8 km (level 0.4). The base obs encodes the predicted miss only as sin/cos of a ~0.05 rad angle, a weak signal. v4 (same settings + zero-effort guidance channels incl. signed miss on an asinh scale) cleared level 0.4 in 0.29M steps vs 3.3M here.

## 2026-11-11 - ppo-zem-v4  (`868ee8a7`, KILLED to fold in fixes)

**Hypothesis:** v3 settings plus 8 zero-effort guidance channels in the observation (ballistic impact speed and its radial/tangential split, time to impact along the conic, signed predicted miss on an asinh scale, stop ratio at full thrust, braking delta-v vs remaining, reaches-surface flag). Expect faster curriculum progress and better site precision than v3 (which plateaued at level 0.4, 69% on-site), since site targeting becomes nulling a gauge. 4 envs x 2048 steps, in parallel with v3.

**Outcome:** stopped at 3.6M steps to fold in two fixes. Curriculum: 8k:0.10, 1572k:0.15, 1687k:0.20, 1916k:0.25, 2244k:0.30, 2301k:0.35, 2359k:0.40, 2646k:0.45 -- cleared level 0.4 in 0.29M steps where v3 needed 3.3M, confirming the guidance channels. At 0.45 for 1M steps: deterministic on-site 74% / soft 97%; stochastic 62% / 90% (the gate reads the stochastic number).

**Diagnosis:** breakdown by initial offset between site and ballistic impact point: 0-4 km offset -> 64% on-site, median miss 7 km; 12-18 km offset -> 91%, miss 1.3 km. The policy applies a learned prior about where the site usually sits relative to the expert's trajectory instead of nulling the miss gauge. Cause: the only lateral gradient is the 3D site-distance potential, which is dominated by altitude during a descent. v5 adds a potential on the predicted miss itself and gates level advances on a 48-episode deterministic probe.

## 2026-11-11 - ppo-zem-v5  (`a8f09e6c`, KILLED to fold in fixes)

**Hypothesis:** v4 (guidance channels) plus (a) a potential-based shaping term on the predicted zero-effort miss (w=2, 20 km scale, capped 200 km) so lateral error has gradient throughout the descent, and (b) level advances gated on a 48-episode deterministic probe at the current level instead of the noisy rollouts. Expect the site-precision plateau at 0.4-0.45 to clear and the from-orbit eval to show on-site landings once level > 0.6.

**Outcome:** stopped at 2.7M steps, just after reaching level 0.5. Curriculum: 8k:0.10, 786k:0.15, 1310k:0.20, 1466k:0.25, 1597k:0.30, 1613k:0.35, 1679k:0.40, 1703k:0.45, 2678k:0.50 -- level 0.4 cleared in 25k steps (v4 0.29M, v3 3.3M); ~1M steps at 0.45. Deterministic @0.45: on-site 61%, soft 99%, crash 0.5%. The per-rollout 48-episode probe halved throughput (4.1k fps vs 7.5k).

**Diagnosis:** the 0.45 wall is a curriculum artifact: for post-deorbit starts the expert touches down 13-48 km upstream of the ballistic impact point, and the site noise (sigma 0.12*level rad, 19 km at level 1) sometimes puts the site at or beyond that point. Upstream sites: 81-94% on-site; at/beyond: 0-9%, since reaching them means staying aloft longer, which the real from-orbit task never needs (there the policy picks the deorbit time). Also the policy spends 1.2-2.1 km/s per episode (hover detours; expert ~1.1). v6: cap descent-start site sigma at 0.04 rad, probe every 3rd rollout, fuel cost 0.0005/(m/s).

## 2026-11-11 - ppo-zem-v6  (`c1b68c0b`, KILLED to fold in fixes)

**Hypothesis:** v5 plus: descent-start site offset noise capped at 0.04 rad (6.4 km) so every curriculum start is reachable by braking (v5 showed sites beyond the ballistic impact point are a different, unneeded manoeuvre); deterministic probe only every 3rd rollout (throughput); fuel cost 0.0005 per m/s to discourage 2 km/s hover detours. Expect the curriculum to pass 0.5 and the from-orbit eval to show on-site landings as orbit-coast states enter (level > 0.5).

**Outcome:** stopped at 3.6M steps. Curriculum: 8k:0.10, 368k:0.15, 958k:0.20, 983k:0.25, 1228k:0.30, 1400k:0.35, 1622k:0.40, 1646k:0.45, 1695k:0.50 -- reached level 0.5 (full descent incl. the deorbit burn) at 1.7M steps, the fastest yet; then 1.9M steps at 0.5 with on-site 45-64%, soft 71-81%, crash 12-17%, timeout 8-13%, probe ~0.5-0.6, no clear trend.

**Diagnosis:** the new states at 0.5 start in orbit just before the expert's deorbit, and the expert's 'deorbit' is a full retro burn (ecc 0.02 -> 1.0 in 20 s: stop, drop, brake). Two signal problems there: (1) while the coasting path misses the moon, the predicted-miss channel and potential point at the periapsis of a near-circular orbit, i.e. +-320-460 km with sign flips; (2) the miss potential was linear with a 200 km cap, so the orbit->touchdown swing was 20 units against +10 for landing on-site and ~2 for the last 20 km. v7: asinh-shaped miss potential held constant (worst case) while not reaching the surface, and the obs channel reports the along-track site offset from the sub-point in orbit.

## 2026-11-11 - ppo-zem-v7  (`e1d2caa0`, KILLED to fold in fixes)

**Hypothesis:** v6 plus: miss potential asinh-shaped (orbit->touchdown swing ~1.7w, last 20 km ~0.7w instead of 20:2) and held constant while the coasting path does not reach the surface (no periapsis garbage, no reward dip when the retro burn starts); the obs miss channel reports the along-track site offset from the sub-point while in orbit. Expect level 0.5 (deorbit burn) to clear and the from-orbit eval to produce on-site landings as coast states enter.

**Outcome:** stopped at 2.6M steps. Curriculum: 8k:0.10, 466k:0.15, 1155k:0.20, 1302k:0.25, 1376k:0.30, 1671k:0.35, 1695k:0.40, 1744k:0.45, 1769k:0.50 -- same wall at 0.5 as v6 (on-site 49-59%, soft 76-79%, timeout 12%). First ever from-orbit on-site landing in a periodic eval (6% at 1M), not sustained.

**Diagnosis:** deterministic breakdown at 0.5: starts inside the descent land on-site 70-75%; starts in orbit before the retro burn (ttl 205-222) time out 44/59 with ZERO thrust, and the few self-initiated deorbits land 100-350 km off. The shaping reshape was not the binding constraint: with gamma 0.995 a +10 landing ~1000 steps away is worth 0.07 now, so waiting for the right phase has no value and a 25 s blind retro burn is never discovered; the timeout penalty 600 steps out is worth 0.5. v8: gamma 0.9995 (half-value horizon ~1400 steps = one lunar orbit), gae-lambda 0.98, and the curriculum spreads the retro burn + the 40 s before it over levels 0.5-0.65 instead of one jump.

## 2026-11-11 - ppo-zem-v8  (`d8c9c331`, KILLED to fold in fixes)

**Hypothesis:** v7 plus gamma 0.9995 (was 0.995) and gae-lambda 0.98, so a landing up to one lunar orbit (~1400 steps) away keeps half its value and waiting for the right deorbit phase is worth something; curriculum levels 0.5-0.65 now cover the retro burn and the 40 s of coast before it in three steps. Expect the policy to start committing to deorbits from orbit states and the from-orbit eval to climb.

**Outcome:** stopped at 3.6M steps. Curriculum: 8k:0.10, 417k:0.15, 1302k:0.20, 1400k:0.25, 1646k:0.30, 2310k:0.35, 2383k:0.40, 2408k:0.45, 2506k:0.50, 2703k:0.55 -- first run past level 0.5 (in 200k steps), then 1M steps at 0.55 with on-site ~58%, probe 0.48-0.69, timeouts creeping 5% -> 10%, mean miss 29 -> 49 km.

**Diagnosis:** deterministic breakdown at 0.55: descent starts fine (ttl<192: 70-80% on-site). Orbit starts with the site in the lead window: the policy DOES deorbit now (gamma fix worked) but 150-340 steps late, lands 100-250 km off or burns 2.2-3.7 km/s and crashes dry. Orbit starts with a re-randomised far site: waits correctly, then hits MY episode cap (ttl+400 = ~610 steps) while the site needs up to ~1300 steps to come round -> guaranteed -10, i.e. the curriculum punished waiting. Also only ~7% of episodes at 0.55 started in the burn-commit states (sampler band 60-100% of the cap). v9: orbit starts get the full 1800-step window; sampler puts half the starts in the newest 15% of the range; probe 24 episodes every 5 rollouts.

## 2026-11-11 - ppo-zem-v9  (`6cd24c89`, KILLED to fold in fixes)

**Hypothesis:** v8 plus: orbit-coast starts (re-randomised site) get the full 1800-step window instead of ttl+400, so waiting up to a full orbit for the right phase is possible and not punished; the start sampler puts half of the episodes in the newest 15% of the time range (the burn-commit states were ~7% of episodes at 0.55); probe 24 episodes every 5 rollouts. Expect the deorbit timing to be learnable now and the from-orbit eval to climb once the level passes ~0.65.

**Outcome:** stopped at 2.7M steps. Curriculum: 8k:0.10, 409k:0.15, 1597k:0.20, 1679k:0.25 -- stalled twice (1.1M steps at 0.15, 0.9M at 0.25) with the stochastic on-site rate at 80-95% but the deterministic probe at 4-29%.

**Diagnosis:** not a v9-specific bug: traced the mean vs the sampled policy from the same start. The mean policy flies a clean braking descent, reaches 4 m at 2.5 m/s, then HOVERS at 2-5 m for 30-150 s translating sideways to shave a 3 km miss that is already inside the 8 km site radius (sampled policy: touches down ~30 s earlier). With gamma 0.9995 a hover step costs 0.05% of the landing bonus, so surface perfectionism is ~free, and the hover is fragile (earlier probes: 130-240-step episodes ending 10-16 km off). v10 adds a 0.02/step cost below 500 m altitude.

## Orbit-phase zero-effort miss (2026-11-11)

`MoonLandingEnv._zem()` now defines the predicted miss in every regime: the
conic impact point against where the site will be at impact (the moon is
tidally locked; the env's predictor ignores the 81 m/s site motion, a
16 km bias early in a fall), or, while the path does not reach the surface,
the touchdown of a stop-and-drop started now (burn drift at 0.9 a_max plus
site motion over burn + 1.2 x free-fall time), blended by horizontal speed.
Along the expert's flights it reads 3.6-5.3 km at the step the autopilot
fires its deorbit burn, stays continuous through the burn, and converges to
the true 7 km miss at touchdown. Used by v11 as both observation and
potential.

## 2026-11-11 - ppo-zem-v10  (`87e0228c`, KILLED as control)

**Hypothesis:** v9 plus a 0.02/step cost while below 500 m altitude, so that with gamma 0.9995 the mean policy commits to touchdown instead of hovering at the surface (v9: mean policy hovered 30-150 s at 2-5 m, probe gate stuck at 4-29% while sampled rollouts were 80-95%). Everything else as v9 (full window for orbit starts, newest-15% start sampler).

**Outcome:** stopped at 7.6M steps as the control for v11. Curriculum: 8k:0.10, 573k:0.15, 1392k:0.20, 1556k:0.25, 1597k:0.30, 2088k:0.35, 2334k:0.40, 2662k:0.45, 2703k:0.50, 2990k:0.55, 5120k:0.60. First run to clear 0.55 (at 5.1M); then 2.5M steps at 0.6 with on-site 45-48%, probe 33%, mean miss 70 km. From-orbit eval on-site/soft by step: 0M:0%/0%, 1M:0%/0%, 1M:0%/31%, 2M:0%/0%, 2M:0%/0%, 3M:0%/0%, 3M:0%/0%, 4M:0%/62%, 4M:6%/31%, 5M:0%/44%, 5M:0%/38%, 6M:6%/62%, 6M:0%/88%, 7M:0%/75%, 7M:12%/69% -- deorbits reliably (69-88% soft) but lands 150-200 km off: no timing.

**Diagnosis:** the hover cost fixed the probe gate (levels 0.1-0.5 in 2.7M steps, no deterministic/stochastic gap), and the orbit-start window fix removed the timeouts. What remains is deorbit timing, which here is learned only from the terminal reward difference (+10 on-site vs +1..5 off) hundreds of steps after the decision; that is slow and plateaus. v11 (same run + the stop-and-drop zero-effort miss in orbit) had half the mean miss at level 0.5 and is the continuation.

## 2026-11-11 01:29 - smoke2  (`3512510d`, FINISHED)

**Hypothesis:** smoke test of BoundedPPO

**Outcome:** level reached 0.05; final on-site 0%, soft 0%, crash 0%, median miss 2377.0 km, 20,480 steps in 0 min

**Diagnosis:** (fill in with `mlflow` tag edit or explog) see curriculum level history: 7k:0.05

**Key metrics:** final/level=0.05, final/on_site=0, final/soft=0, final/crash=0, final/timeout=0, final/miss_km_med=2.38e+03, final/rel_med=2.54e+03, final/dv_mean=3.94e+03, final/native=-355, final/full_mission=2.13e+03

## 2026-11-11 - ppo-zem-v11  (`0cc420ca`, KILLED to fold in fixes)

**Hypothesis:** v10 plus the zero-effort miss extended to the orbit phase: while the coasting path misses the moon, the gauge (obs channel and potential) is the touchdown of a standard stop-and-drop started now (burn drift + site motion during the fall), blended with the ballistic impact point by horizontal speed. Along the expert's flight it reads 3.6 km at the exact step the autopilot deorbits, with no tuning. Expect the deorbit timing to be learned from a dense gradient instead of the terminal reward alone; v10 continues in parallel as the control.

**Outcome:** stopped at 6.2M steps. Curriculum: 8k:0.10, 409k:0.15, 1228k:0.20, 1392k:0.25, 1433k:0.30, 1474k:0.35, 1515k:0.40, 1556k:0.45, 1679k:0.50, 2621k:0.55 -- fastest to 0.55 (2.6M vs v10's 5.1M; at 0.5 half of v10's mean miss), then collapsed at 0.55: on-site 25-34%, soft 55-59%, crash 25-32%, timeout 12-16%. First from-orbit eval at 500k already showed 12% on-site / 50% soft with a 27 km median miss (never seen before), not sustained.

**Diagnosis:** traced from a pre-deorbit state: the expert's burn scores +10.2 shaped return, the policy coasts with throttle pinned at -1.00 and times out at -8.2, so the reward is right and the policy cannot act on it. Policy mean for the throttle at the burn states is -2.1 with std 0.08 (started 0.37): 13 sigma from the engine-on threshold, so no sampled action ever fires the engine and the policy gradient toward burning is zero -- a saturation trap of an unbounded Gaussian mean with 'off' at the action boundary. It also stops mid-burn when handed mid-burn states. v12: BoundedPPO (penalty on |mean| > 1, std floor 0.2), everything else as v11.

## 2026-11-11 - ppo-zem-v12  (`b495d5a4`, KILLED to fold in fixes)

**Hypothesis:** v11 (orbit-phase ZEM gauge) trained with BoundedPPO: a penalty on the Gaussian mean outside [-1,1] and an exploration-std floor of 0.2, so 'engine off' can never become an absorbing state at the deorbit (v11: mean throttle -2.1, std 0.08 at the burn states, zero chance of ever firing). Expect the burn commit to be discoverable and the from-orbit on-site rate to climb past level 0.55.

**Outcome:** stopped at 4.7M steps. Curriculum: 8k:0.10, 409k:0.15, 983k:0.20, 1433k:0.25, 1515k:0.30, 1638k:0.35, 1720k:0.40, 1761k:0.45, 2170k:0.50, 3235k:0.55, 5406k:0.60 -- no collapse at 0.55 this time (bound loss stayed 0, std at the 0.2 floor) but a plateau like v10: on-site 38-50%, soft 79%, crash 17-20%, mean miss 40-49 km. From-orbit eval 6% on-site / 44% soft.

**Diagnosis:** new local optimum: the policy never fires a real retro burn; its throttle idles near hover (exp map: a0~-0.5..0 is 0.2-1x hover thrust, not off), so it spirals down on a trickle for 300-800 steps spending 2-3 km/s and lands 15-300 km off (expert-site orbit starts: median |gauge| seen before 'burning' 0.6 km -- the window passes and it does not commit; 72/103 land off-site). Bounding the mean removed the engine-off trap but a full burn is still ~1.7 action units from idle, a valley a unimodal Gaussian does not cross from reward alone. v13 anchors the policy to the fixed expert bank with a BC term inside PPO (coef 0.2) plus a BC warm start, keeping bounds, std floor and the ZEM gauge.

## Milestone: first learned full-mission landings (2026-11-11, v13 @ 4M steps)

`scripts/fly_moon_policy.py` with v13's best checkpoint, flown from the
launch pad (scripted prefix to the parking orbit, then the policy) on 16
sites: 6/16 `landed_moon_site` (0.3-7 km miss, 4-20 m/s, mission return
3040-3090 = the autopilot's), 7/16 soft but 15-64 km off, 3/16 crashed.
Pattern: sites that need a short wait after orbit entry land on target;
sites that need the longest wait (near 0 deg) get an early deorbit, a long
fuel-burning correction and a dry-tank crash. Consistent with the curriculum
being at level 0.6, where starts are at most ~17 s of coast before the
burn: long waits are not trained yet.

## Milestone: 79% on-site, 100% soft, full mission from the pad (2026-11-11, v13 @ 5M steps)

Checkpoint `models/ppo-zem-v13/best_5M_79pct.zip` (from-orbit eval: 81%
on-site, 100% soft, median miss 4.7 km). Full mission from the launch pad on
24 sites (`scripts/fly_moon_policy.py`): 19/24 `landed_moon_site` at 0.4-10.6
m/s and 0.7-7.6 km from the site, mission returns 3037-3090 (the autopilot
scores 3040-3090); 5/24 soft but off-site (sites -45..+15 deg, the ones that
need the longest wait in orbit: the policy deorbits 170-330 s after orbit
entry instead of waiting ~1200 s, since the curriculum at level 0.65 has
only trained coasts of up to ~40 s); 0 crashes. Training continues toward
level 1.0 (full-orbit waits).

## 2026-11-11 - ppo-zem-v13  (`fa583cac`, FINISHED: curriculum complete)

**Hypothesis:** v12 plus imitation of the expert's from-orbit flights: 20 epochs of behavior cloning before PPO and a fixed-bank BC anchor (coef 0.2) in every PPO minibatch, so the bang-bang retro burn at the gauge zero is in the policy from the start and cannot be forgotten while the curriculum repairs the touchdown; PPO is free to beat the expert where the advantage is clear (it already lands closer than the expert's 7 km). Expect the deorbit timing to hold and the from-orbit on-site rate to climb as the level passes 0.55.

**Outcome:** SUCCESS of the approach. Curriculum completed to level 1.0 at 8.4M steps (8k:0.10, 573k:0.15, 655k:0.20, 1105k:0.25, 1392k:0.30, 1515k:0.35, 1638k:0.40, 1720k:0.45, 1761k:0.50, 1843k:0.55, 2211k:0.60, 4956k:0.65, 5529k:0.70, 6225k:0.75, 6635k:0.80, 7045k:0.85, 7372k:0.90, 8192k:0.95, 8396k:1.00). From-orbit eval (on-site/soft/median miss): 0.5M:6%/6%/49km, 1.0M:12%/12%/52km, 1.5M:0%/31%/74km, 2.0M:6%/75%/176km, 2.5M:12%/88%/23km, 3.0M:0%/62%/103km, 3.5M:19%/88%/20km, 4.0M:38%/81%/19km, 4.5M:25%/62%/11km, 5.0M:81%/100%/5km, 5.5M:44%/100%/8km, 6.0M:88%/94%/2km, 6.5M:75%/81%/3km, 7.0M:31%/81%/9km, 7.5M:62%/94%/6km, 8.0M:75%/100%/5km, 8.5M:62%/100%/4km, 9.0M:31%/100%/11km. Full mission from the pad with the 5M checkpoint: 19/24 on-site, 24/24 soft, 0 crashes. Stopped at 9.1M to resume from the best checkpoint as a low-lr polishing run (v14); on-site rate between checkpoints swings 31-88% at constant lr while soft stays ~100%.

**Diagnosis:** what made it work, in order of discovery: BC warm start + fixed-bank BC anchor gave the bang-bang retro burn at the gauge zero from step 0 (pure PPO never found it); the orbit-phase zero-effort miss made the timing a dense gradient; bounded means + std floor kept exploration alive; gamma 0.9995 made waiting worth it; the hover cost made the mean policy commit to touchdown. Remaining weakness: checkpoint-to-checkpoint variance of the on-site rate.

## 2026-11-11 - ppo-zem-v14-polish  (`bcf11590`, KILLED to fold in fixes)

**Hypothesis:** polish: resume v13's best checkpoint (88% on-site from orbit) at level 1.0 with lr 1e-4 (was 3e-4), std floor 0.1 (was 0.2), BC anchor kept, 32-episode evals every 250k. Expect the on-site rate to stabilise above 80% instead of swinging 31-88% between checkpoints.

**Outcome:** stopped at 1.9M steps. 32-site from-orbit evals (on-site/soft/median miss): 0.25M:81%/97%/4.0km, 0.50M:78%/84%/3.8km, 0.75M:28%/94%/9.6km, 1.00M:84%/100%/2.8km, 1.25M:88%/94%/3.1km, 1.50M:72%/81%/1.6km, 1.75M:31%/97%/10.8km, 2.00M:88%/100%/3.6km. Best 84%/100%/2.8 km at 1M (best_84pct.zip). The 31-88% swing persisted at lr 1e-4.

**Diagnosis:** compared the best and a weak checkpoint site by site: identical deorbit timing (median gauge at burn -2.1 vs -2.7 km, same burn-start steps) but the weak one lands 8-14 km off on nearly every site where the strong one lands 0.2-5 km off, so the swing is the descent's lateral precision, not the timing. Consistent with the BC anchor (expert profile, which itself lands 7 km off target) fighting the policy gradient. v15: anchor 0.05, std floor 0.05. Separate residual failure (2/16 sites): when the site is just behind at orbit entry, the policy fires 17-18 s late instead of waiting a full orbit.

## Milestone: 92% on-site, 100% soft, 0 crashes from the pad on 48 sites (2026-11-11, v15 @ 0.5M)

Checkpoint `models/ppo-zem-v15-polish/best_91pct.zip` (from-orbit 32-site
eval: 91% on-site, 100% soft, median miss 1.1 km, native return 516 vs the
expert's ~524). Full mission from the launch pad on 48 sites: 44/48
`landed_moon_site`, 4/48 soft off-site, 0 crashes, mean mission return 3043
(autopilot 3040-3090). The 4 misses are the consecutive sites -45..-22.5 deg,
where the site sits just behind the vehicle at orbit entry: the policy fires
its retro burn 180-290 s in instead of waiting one full orbit (~1460 s) for
the next pass. Training continues at level 1.0 with the weak anchor.

## Milestone: 94% on-site from the pad on 48 sites (2026-11-11, v15 @ 2.75M)

Checkpoint `models/ppo-zem-v15-polish/best_94pct.zip` (from-orbit 32-site
eval: 94% on-site, 100% soft, median miss 2.3 km, native return 525 = the
expert's). Full mission from the launch pad on 48 sites: 45/48
`landed_moon_site`, 3/48 soft off-site (sites -45, -37.5, -30 deg: the
"site just behind at orbit entry" cases, fired 150-400 s in instead of
waiting an orbit), 0 crashes, mean mission return 3048 (autopilot
3040-3090).

## Note: v15 @ 6M checkpoint (`best_97pct.zip`)

From-orbit eval 97% on-site / 100% soft / native 533. From the pad on 48
sites: 46/48 on-site, but 1 crash (+120 deg: tank ran dry at 184 m/s, 1 km
from the site) and 1 soft off-site (-45 deg). Mean mission return 3047.
Recommended deliverable stays `best_94pct.zip` (45/48 on-site, 48/48 soft,
0 crashes, 3048): a zero-crash policy beats one extra on-site landing.

## 2026-11-11 - ppo-zem-v15-polish  (`85a14ddc`, FINISHED)

**Hypothesis:** polish 2: resume v14's best (84% on-site, 100% soft, 2.8 km) with the BC anchor weakened to 0.05 (was 0.2) and std floor 0.05 (was 0.1), lr 1e-4. The on-site swing was traced to descent precision fighting the anchor, so a weaker anchor should let the policy settle on its own precise descent while still never forgetting the retro burn.

**Outcome:** polish complete. 32-site from-orbit evals over 10M steps: on-site 88-97% (median 94%), soft 94-100%; best 97%/100%/native 533 at 10.75M. From the pad on 48 sites: best_94pct.zip 45/48 on-site, 48/48 soft, 0 crashes, mean mission return 3048 (recommended); best_97pct.zip 46/48 on-site but 1 dry-tank crash. The checkpoint swing is gone (anchor 0.05); the run began to drift after 9.5M (56% at 10M) and was stopped.

**Diagnosis:** remaining failure mode: when the site is just behind the vehicle at orbit entry (sites around -45..-30 deg) the policy fires 150-400 s in instead of waiting a full orbit; those starts are rare in the curriculum bank (only the flights whose deorbit came right after capture). Next step if pursued: oversample long-wait orbit starts, or a discrete throttle to drop the imitation anchor entirely.

## FINAL: 48/48 on-site from the launch pad (2026-11-11, v15 @ 10.75M)

Correction to the v15 entry above: the interrupt at 10M did not stop the
process; it ran on to 12.5M and was then stopped. Its 32-site from-orbit
evals between 10.25M and 12.25M: 91-100% on-site, 97-100% soft; two
checkpoints (10.75M, 11.5M) at 100%/100% with native return 541 (expert
~524). Checkpoint `models/ppo-zem-v15-polish/best_100pct.zip` flown as the
full mission from the launch pad on all 48 sites: 48/48 `landed_moon_site`,
0 crashes, mean mission return 3065 (autopilot 3040-3090). This is the
deliverable. Other preserved checkpoints: best_94pct (45/48, 0 crashes),
best_97pct (46/48, 1 crash).

## New environment: ToMars-v0 (2026-11-11)

Three-stage rocket, a star at the centre of a compact system (planet year
3.9 h, Hohmann transfer 2.8 h, planet 0.35% of the star's mass), a
Mars-like target with a thin atmosphere, a departure window that does not
return within the episode. `orbital_gym/missions/to_mars.py`, presets
`STARS`/`SYSTEMS`/`ROCKETS["mars"]`, renderer system view, tests in
`tests/test_to_mars.py`. The scripted `MarsAutopilot` lands softly
(1 m/s, ~100 km off the site, 15,300 steps, 3.1 km/s spare). Things learned
building it: patched conics are useless here (the star's tide does ~1 km/s
of work during the climb-out, so the departure is planned by simulating the
full force model ahead and pricing the rest with free-arrival-time
Lambert); a free-time Lambert re-planner must not hop between arrival
times (1.8 km/s wasted before it was pinned); the moon pilot's capture loop
descends at <= 25 m/s, so capture must happen at the parking altitude.
No RL attempt yet.
