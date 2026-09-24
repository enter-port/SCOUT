# κ calibration investigation

Git 归档范围、代码迁移与核验记录见 [ARCHIVE.md](ARCHIVE.md)。

Status: all launched evaluations are complete; square round1/2 remain untested
because their checkpoints were unavailable. Final audit covers 13 initial
evaluations, 13 paired DP controls and 53 SCOUT arms. This file preserves the
chronological research notes below.
Use [RESULTS.md](RESULTS.md) for the latest validated result tables and
[METHODS.md](METHODS.md) for current definitions and interpretation. Early
statements that an arm is pending are historical, not the current job state.

## Evaluation protocol

- Tasks: can, square, coffee, threading, tool_hang.
- Start with base DP and dyn checkpoints, training seed 233.
- Initial unguided evaluation: 100 scenes, eval seeds 42–141.
- Freeze each task's initial failure set. Evaluate DP and SCOUT on that same
  set with five attempts per scene, four workers, 25 environments per worker.
- Primary metric: number of initially failed scenes rescued by pass@5.
  SCOUT must rescue more scenes than DP; strictly more than 1.5 times DP is
  the requested excellent result. Total pass@5 also includes initial successes.
- First sensitivity probe: calibrate eta to R in [0.009, 0.011] at κ=2.5,
  then hold eta fixed for κ in {1, 2.5, 5}. This does **not** hold R fixed
  when κ changes. Candidate calibration rules need to account for this coupling.
- Do not infer that every cross-task rule is impossible from a small κ sweep.
  Validate a promising rule on round1–2; if no useful cross-task rule emerges,
  evaluate fixed-task cross-round calibration on identical checkpoint pairs.

Server outputs are under
`/mnt/workspace/baojiachun/scout/experiments/2026_09_24_kappa_calibration/`.
No pre-existing campaign scripts were replaced on the server.

## Source definitions checked against code

The cost is `-min(KL(q(z|s,a) || q(z|s,a0)), κ)`. The baseline `a0` is the
unguided intent at the first guided denoising step of each action chunk.
The gradient is masked when the current KL exceeds κ. The current code does
not project the next action back to a KL ball, so uncapped KL can exceed κ.

`scripts/coffee/cfk_rcalib.py` defines

```
R = mean_guided_steps(eta * sqrt(1-alphabar_t) * meanabs(capped_gradient))
    / meanabs(core abs_actions)
```

The denominator is the raw core action scale, not the noisy action scale.
The diagnostic script records both ratios without changing the production
definition of R. The 128-observation batch and RNG seed 0 match the existing
calibrators.

`scripts/threading/thm2_c_calib.py` defines C as the mean **uncapped** KL over
all guided steps and batch rows. It updates κ by `κ * C_target / C`.

## Confirmed existing calibration issues

1. `--kappa0` was used both as the round's starting κ and as the base κ used
   to measure C_target. Both round drivers pass the previous round's κ.
   Thus the supposedly fixed base reference drifts after κ first changes.
   In threading seed233, the same base pair and eta yielded C_target 9.9055
   at κ=2.5, then 14.3591 when the previous κ became 3.17258. Coffee seed233
   changed from 4.1167 to approximately 2.998 when base κ became 1.88678.
   Local fix: separate `--base-kappa` (default 2.5) from `--kappa0`; both
   current round drivers explicitly pass `--base-kappa 2.5`.
2. The multiplicative C update can oscillate across its target. For example,
   threading `ATY-exp2` (using round1 checkpoints) probes
   `(κ,C)=(2.5,6.02),(4.111,12.84),(3.173,8.69)`
   against C_target 9.91, and reports nonconvergence. The caller still uses
   the last κ. Local opt-in `--solver bracket` expands a bounded interval
   and bisects in log κ. It returns only an actually measured κ, records
   why it stopped, and marks an unreachable target as nonconverged.
   `--require-converged` can reject such a result after writing diagnostics.
   Seven numerical tests cover nonlinear response, a flat unreachable
   target, a discontinuous response, the probe budget, and invalid values.
   Convergence alone does not establish that matching C improves pass@5.
3. The campaign first recalibrates eta and then changes kappa. Its final R
   can therefore change. This is a property of the sequential method, not
   by itself a confirmed bug. We distinguish joint R/C matching from the
   original sequential R-then-C order. Both record final R; only the joint
   method requires final R to remain in the target band.

The local base-reference fix has passed Python syntax and shell syntax
checks. It has not been deployed into active server campaigns.

## Current base inputs and completed controls

| Task | eta at κ=2.5 | R | Initial success | DP rescued | DP pass@5 |
|---|---:|---:|---:|---:|---:|
| can | 2.299270 | 0.010600 | 57/100 | 8 | 0.65 |
| square | 2.464778 | 0.010237 | 37/100 | 9 | 0.46 |
| coffee | 1.341886 | 0.010138 | 69/100 | 16 | 0.85 |
| threading | 4.692822 | 0.010134 | 47/100 | 16 | 0.63 |
| tool_hang | 2.622425 | 0.010115 | 31/100 | 35 | 0.66 |

First completed SCOUT results: can at fixed eta=2.299270, κ=1 rescues 10
scenes (DP 8), pass@5=0.67. This meets the minimum criterion but not the
excellent criterion. Its core R is only 0.003531. A second κ=1 arm uses
eta=7.910159 to restore R=0.009647 and rescues 16 scenes, pass@5=0.73.
At κ=2.5 and R=0.010600, can rescues 15 scenes, pass@5=0.72. Both satisfy
the excellent criterion. Both preserve all eight scenes rescued by DP,
and the two SCOUT rescue sets overlap on 14 scenes. This is evidence for
a useful κ range after controlling R on **can**, not yet a cross-task result.
Coffee at fixed eta=1.341886, κ=1 rescues 19 scenes (DP 16), pass@5=0.88,
also meeting only the minimum criterion.
Coffee at κ=2.5 rescues 23 scenes, pass@5=0.92 (23/16=1.4375 times DP).
It improves on κ=1 but remains below the strict excellent threshold of 25.
The completed fixed-eta κ=5 arms rescue 17 on can (17/8=2.125) and 24 on
coffee (24/16=1.5, which is **not** strictly excellent). Coffee at κ=1 with
R restored to 0.009061 rescues 21. Threading at fixed eta and κ=1 rescues
19, versus DP's 16.
Threading at κ=2.5 rescues 26 versus DP16, pass@5=0.73 (excellent, 1.625x).
Square at fixed eta and κ=1 rescues 19 versus DP's 9, pass@5=0.56,
which satisfies the excellent criterion despite core R=0.003708.
Tool_hang's completed DP control rescues 35 of 69 initial failures;
its excellent threshold is therefore 53, and its minimum threshold is 36.
Tool_hang at fixed eta and κ=5 rescues only 3, pass@5=0.34, clearly failing
the minimum criterion. A smaller-P matched-R arm is being evaluated.

The paired accounting script `scripts/calibration/kappa_report.py` checks
checkpoint identity, all 100 scenes, the frozen failure set, five attempts,
four shards, and agreement between rescue counts and per-scene records.
It reports scenes gained and lost relative to the same DP control. A
missing summary is pending, never a zero score. Downloaded metadata is in
`snapshot/`; its contents are a snapshot, not a live view of the server.

Checkpoint provenance is recorded in each server `eval/summary.json` and
`k*/kl_probe.json`. The base source families are CAN-entropy-s233,
SQUARE-entropy-s233, COFFEE-MG-p5-s233, THREADING-THM3-s233, and TOOLHANG-s233.

## Mechanism diagnostics

`scripts/calibration/kappa_diagnostics.py` measures one real denoising pass
on the same frozen core batch for each condition:

- eta=0: natural DP KL drift from the intent anchor;
- the calibrated fixed eta at κ=0.5,1,2.5,5,10;
- actual R, noisy-action-relative injection, uncapped C, KL quantiles,
  and fraction of rows/steps above the cap.

These are candidate explanatory variables, not substitutes for pass@5.
The full per-row KL arrays are saved in compressed NumPy files for later
analysis. In particular, natural KL provides a possible reference scale
that does not depend on the guided trajectory being calibrated.

The task configurations use different guidance starts: can/square use 100,
while coffee/threading/tool_hang use 50. Their intent anchors are therefore
captured at different noise levels. The following differences cannot be
attributed purely to the task or dyn model.

| Task | Guided steps | Natural C (eta=0) | C at κ=2.5, R≈0.01 | Cap fraction | R at κ=1, fixed eta | R at κ=5, fixed eta |
|---|---:|---:|---:|---:|---:|---:|
| can | 100 | 19.8643 | 20.6934 | 0.7776 | 0.003531 | 0.019856 |
| square | 100 | 4.0985 | 4.5169 | 0.4223 | 0.003708 | 0.017521 |
| coffee | 50 | 0.2530 | 4.2185 | 0.4273 | 0.007195 | 0.012755 |
| threading | 50 | 0.0331 | 9.9055 | 0.2645 | 0.007831 | 0.012407 |
| tool_hang | 50 | 0.0290 | 6.3799 | 0.2261 | 0.007157 | 0.013065 |

The optional `--target-r 0.01` mode recalibrates eta for every κ and writes
`diagnostics_rmatched/`. All five tasks converged at κ=1 and κ=5. These
measurements allow subsequent rollout comparisons to control R rather than
only holding eta fixed. Fixed-eta cap fraction is almost constant with κ
for coffee and threading; matching R changes that relationship, so it must
be considered separately.

At matched R, can's mean uncapped C is 20.18, 20.69, and 20.74 for κ=1,
2.5, and 5, against natural C=19.86. Thus this C statistic barely responds
to a fivefold change in kappa on can. On threading and tool_hang, the top
10% of KL entries contribute roughly 87% and 88% of C at κ=2.5. These
observations explain why neither raw mean C nor a common cap-hit fraction
is currently justified as a universal calibration statistic.

`scripts/calibration/kappa_eval_arm.py` evaluates such a calibrated pair
with the original four-worker protocol. It checks the physical GPU UUID
and idle memory, requires a converged calibration and matching checkpoint
metadata, and refuses to overwrite an existing output directory.

## Primary candidate: joint R and scaled cost cap P

The actual injection contains `eta * grad(min(f, kappa))`. If an otherwise
identical cost changes by a scalar factor `f_new = c * f_old`, then

```
kappa_new = c * kappa_old
eta_new = eta_old / c
P = eta * kappa  (invariant)
```

preserves the injection at every denoising step. This is an exact
conditional scaling identity; learned cost shapes can also change, so it
does not prove that one P works across tasks or rounds.

The candidate **P≈6 and R≈0.01**, each with a 10% relative band, was fixed
in `potential_cap_protocol.json` before completing the five-task base
comparison. It was proposed during this investigation, not specified
before seeing all preliminary data. Round1–2 are still needed for further
validation. Existing core pairs can be reused only when both bands hold;
selection uses proximity to P, not rollout outcomes.

| Task | Selected κ | Selected η | R | P | Result source |
|---|---:|---:|---:|---:|---|
| can | 2.5 | 2.299270 | 0.010600 | 5.7482 | k2.5 |
| square | 2.5 | 2.464778 | 0.010237 | 6.1619 | k2.5 |
| coffee | 7.071068 | 0.848528 | 0.009162 | 6.0000 | p6 |
| threading | 1 | 5.992432 | 0.009531 | 5.9924 | rmatched_k1 |
| tool_hang | 2.5 | 2.622425 | 0.010115 | 6.5561 | k2.5 |

Coffee's P=6 arm completed with **25 rescues versus DP16**, pass@5=0.94,
which is excellent (1.5625x). Its round1 confirmation has started with a
fresh joint calibration and a new paired DP control. Remaining base tasks
continue concurrently; this is not yet a five-task validation of P=6.

`scripts/calibration/kappa_pcalib.py` searches on the frozen core batch
with `eta=P/kappa`, measures actual R, brackets in both directions, and
bisects in log kappa. It refuses to use a nonconverged pair for rollout.
This is numerical calibration on core data, not an environment-success
grid search. Seven scalar-solver tests include decreasing and locally
nonmonotone responses in addition to the original failure cases.

If a common P fails, the corresponding fixed-task hypothesis is to hold
that task's successful **base** P and R fixed across rounds. This also
avoids moving the reference after every round. Neither hypothesis has
passed the required full validation yet.

## Alternative scale: local action sensitivity (candidate only)

`scripts/calibration/kappa_action_sensitivity.py` perturbs expert action
chunks isotropically in the DP's normalized action coordinates. Core
axis-angle actions are converted to rotation-6d using the training dataset's
existing conversion. It uses 16 random directions per state, each with
exact RMS epsilon, and evaluates posterior KL on the frozen core states.

```
S = E[KL(q(z|s,a + delta_a) || q(z|s,a))] / epsilon**2
equivalent local action RMS radius = sqrt(kappa / S)
```

The primary diagnostic uses epsilon=0.05; epsilon=0.01 and 0.1 check local
quadratic stability. These are unclipped perturbations in DP coordinates,
not a physical Cartesian distance. The probe runs on CPU with two threads
and low scheduling priority, and hides CUDA before importing torch.

| Task | S at epsilon=0.05 | Equivalent local RMS radius for κ=2.5 |
|---|---:|---:|
| can | 17.4875 | 0.378 |
| square | 2.9159 | 0.926 |
| coffee | 410.2764 | 0.0781 |
| threading | 55.8752 | 0.212 |
| tool_hang | 466.2574 | 0.0732 |

Small-perturbation S is fairly stable across the measured epsilon range,
but the radii for a common κ differ substantially. Rollout data is still
needed to assess whether optimal values align after this normalization.
A fixed-task candidate is `kappa_round = kappa_base * S_round / S_base`,
followed by recalibrating eta at that κ. This candidate has not yet been
validated on round1–2.

The CPU checkpoint inventory produced the following **unvalidated** caps
from that formula. These are predictions from model sensitivity, not
values selected using environment success outcomes:

| Task | S round1 / base | Predicted κ round1 | S round2 / base | Predicted κ round2 |
|---|---:|---:|---:|---:|
| can | 1.007 | 2.519 | 2.047 | 5.118 |
| coffee | 4.383 | 10.958 | 7.889 | 19.723 |
| threading | 8.058 | 20.144 | 15.003 | 37.507 |
| tool_hang | 0.277 | 0.693 | 0.320 | 0.799 |

Each round uses its own DP normalizer, so these perturbations have equal
size in DP sampling coordinates, not necessarily in physical action units.
The round checkpoint paths are recorded in `round_candidates.json` and
the server's `roundN/task/checkpoints.json`. Square round1/2 checkpoints
have not yet been located. No round rollout result is implied by this table.

## Reproducibility and resources

- Local branch `param-dev`, starting commit after fast-forward:
  `abc5c11cb651be0099ede84aba51deffe9c6e803`.
- Server HEAD: `597e34cab33e5652ec12fd7bf1b9ecbc738c21d3` with pre-existing
  local changes. Git blob hashes of guidance, eta calibration, cap probe,
  rollout, vector rollout, and shard driver were verified equal to local
  HEAD before interpreting results.
- Initial jobs: port1022 GPU2 coffee, GPU3 can, GPU4 square; port1024 GPU0
  threading and GPU6 tool_hang. After the coffee/can jobs completed,
  port1022 GPU2 and GPU3 were reused for tool_hang κ=5 and square κ=2.5.
  A newly free port1022 GPU0 runs tool_hang κ=2.5. Matched-R κ=1 arms run
  on port1024 GPU7 (square), GPU2 (threading), and GPU4 (tool_hang), each
  started only after checking physical UUID and idle memory. Port1022
  GPU7 is prohibited and was not used. CPU sensitivity probes hide CUDA.
- New files and outputs stay within `/mnt/workspace/baojiachun`.


## Results update — 2026-09-24 04:24 CST

Common P≈6 produced base rescue counts can15/DP8, square27/DP9,
coffee25/DP16, threading22/DP16, tool_hang18/DP35. It is not a usable
five-task rule. This rejects the tested candidate, not all possible rules.
Coffee round1 P6 rescues16/DP12 (pass@5 .95/.91), minimum quality.
Coffee round2 DP rescues11 of13 initial failures, making the excellent
criterion unattainable; its SCOUT result is pending.

Can round1 cannot reach R=.01 at P=6: sixteen core probes within the
allowed cap range reach a maximum R=.0046056. No failed calibration is
used for rollout. Alternative: retain the base cap's natural unguided KL
quantile q=.406328125, giving round1 kappa13.826037921458482 and round2
kappa7.752820452861488, then recalibrate eta. Corresponding eta/R are
approximately .9669/.01023 and1.775/.01089. Paired DP controls rescue13
and2; SCOUT results are pending. The base reference stays fixed.

Threading uses its excellent base P=11.732053760076715 rather than the
common P6 candidate. Round1 core calibration gives kappa11.892071150027213,
eta.986544195041237, R.01004472812874251. Both rounds require rollout proof.

An additional direct-scale hypothesis uses eight independent unguided DP
samples at the same128 core observations. Set kappa to median directed KL
between a final DP sample and a different sample's first-guidance anchor.
The seed0 same-sample final-to-anchor KL reproduces the earlier natural
probe, checking instrumentation. Base medians are can4.838144,
square.921471, coffee.09548945, threading.010235, tool_hang.00759784.
A coffee pilot at kappa.09548945248953886, eta25.5252, R.0091442 is pending.
Core statistics alone do not validate this rule.
