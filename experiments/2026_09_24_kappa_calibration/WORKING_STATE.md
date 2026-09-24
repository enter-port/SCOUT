# FINAL STATE — 2026-09-24 07:40 CST

All launched evaluations completed. No owned kappa tmux sessions remain on port1022 or1024. Final audit:13 initial evals,13 DP controls,53 SCOUT arms (66 paired retry records), each100 seed42–141 scenes / frozen failures /5 retries /4workers. All66 records passed scene, checkpoint, actual eta/kappa, mode and shard checks. Last tool fixed-kappa results: round1 13/DP40, round2 14/DP37; joint C round1 21/40, round2 12/37. All fail.

Final results in RESULTS.md, definitions/usage in METHODS.md, machine-readable successful fixed-task candidates in validated_candidates.json. Latest downloaded metadata720files (~9MB); snapshot/FINAL_AUDIT.json is authoritative audit. Raw rolloutHDF5 retained on server in owned experiment directory, roughly626GiB. Runtime versions and final source fingerprints retained.

Conclusions: two-stage fixedkappa1 (eta calibrated at2.5 then frozen) passes all5BASE tasks, but coffeeR2 9/11 fails transfer. MatchedRk1/P6/median failtoolbase. Can naturalq base15/8,R1 15/13,R2 4/2; can median18/8,15/13,3/2; threading median17/16,18/17,25/23. Allthese fixed-task series passminimum; threading gains small. No tested stable cross-round method forcoffee/tool. Square base done; round1/2 checkpoint unavailable, question already asked. Do not claim proof that all universal rules are impossible.

Code: local param-dev C calibrator now separates base-kappa from initial kappa, optional bounded solver and convergence rejection; old active campaign files on server were not replaced. New research scripts deployed under scripts/calibration/kappa_*. kappa_diversity_eval.py supports --calibrate-only; kappa_diagnostics.py supports explicit --eta. kappa_report.py checks actual guidance parameters as well as scenes/model identities. Solver7tests passed, syntax checked, real core and rollout paths validated. No commit or PR created.

There is no remaining scheduled job or conditional continuation. Earlier notes below are historical, not instructions to relaunch work.

---

# MOST RECENT STATE: use LIVE_STATE.json (just refreshed from server)

Tool round2 pbase's original GPU1 became occupied during its initial-eval
wait. The guard stopped BEFORE creating the output directory. Original
log preserved under incidents/. This is resolved: round2 SCOUT is running
on1024 GPU0 after can direct-median completed18/DP8. Never touch GPU1's
new process. Tool round1 SCOUT is running on1022 GPU5.

All tool base arms except direct-median complete: k1 fixedeta41/DP35,
Rmatched-k1 34/35, k2.5 18/35, k5 3/35, P2.5 atR~.01 38/35.
Direct median base now can18/8, coffee26/16, threading17/16; square/tool
pending. Do not confuse threading's barely passing17 with its excellent
base k2.5=26. Threading round1 Pbase completed19/DP17 (minimum).
Threading round2 initial62/100, DP23; Pbase pending.

Added actual two-round threading test of existing C idea, jointly keeping
C and R: round1 on1022 GPU2 gives kappa5 eta1.2122250580820007,
R.00976394985969172 C9.7382230758667 target9.905536651611328; rollout
running. Round2 on1022 GPU3 gives kappa2.5 eta≈1.443 R≈.01053 C≈9.167;
rollout running. Core legacy diagnostics lacked R_converged; new C-reference
validation now accepts measured finite R in band when this flag is absent,
while explicitly false is still rejected. Threading reference eta and R
were checked against converged base eta.json. Original pre-load rejection
log preserved under incidents/; no rollout was started before that fix.

DP-diversity core probes finished for ALL8 available round checkpoints:
can median20.217892122034094 /10.028137789268861;
coffee .2544265806225559 /.328091817797859;
threading .045603644249817527 /.04856655461256025;
tool .0009355129485825362 /.0010773368077966797.
Their R=.01 dose calibrations are now running sequentially on1024 GPU6
(session kappa_round_median_dose, controller E/round_median_dose_probes.py).
E/round_median_dose_probes.done.json appears when all eight are done.
No round direct-median ENVIRONMENT arm has started yet.

NEXT: once this core controller completes and cards free, evaluate coffee
round1/2 direct median as a fixed-task fallback (base is excellent, other
coffee transfers failed round2). Use kappa_diversity_eval.py; it reuses
core measurements. This is separate from claiming the common five-task
rule has passed. If all five base median arms pass, validate all available
rounds with the same predictor. Square early checkpoints are still missing.
Finish active Pbase/C arms and DP controls; no final answer yet.

The paired report now also validates guided/off mode and dyn checkpoint
identity against source metadata. All completed arms passed this audit.
Also audited dyn training provenance for all8 rounds; four legacy paths
(can/tool) have moved, but relative data paths agree AND each model's120
frozen ResNet tensors equal the selected DP's EMA encoder. JSON evidence
is E/round_checkpoint_pair_audit.json. Audit ran onCPU only; no model changed.

Snapshot/results files are still older than live server results; refresh
near final. Final RESULTS.md should be written once actual outcomes settle;
README/METHODS.md need updating (currently contain interim results).

---

# Latest operation update (2026-09-24 about 05:00 CST)

LIVE_STATE.json GPU map now includes TWO newly freed cards after checking
both memory and empty compute-process lists: 1022 GPU5 and1024 GPU1.
They run tool_hang round1 and round2 Pbase respectively. Both calibrations
completed: kappa=.08838834764831845, eta=29.66935038184085,
R=.0072407673572808424 / .007404827016396008, target=.007156729252424086.
Round1 initial eval completed46/100, so DP control on1022 GPU4 and SCOUT
pbase on1022 GPU5 are now running concurrently. Round2 initial eval still
running; its Pbase controller on1024 GPU1 waits for that OWN initial eval.
Do not double allocate a waiting controller's card even if memory is low.
The arm runner rechecks UUID/occupancy after its wait and will abort if busy.
New --wait-for-eval-seconds supports this workflow; syntax checked and
runtime reached round1 SCOUT successfully.

Threading round1 DP completed17, pbase running. Excellent threshold26.
All FIVE direct-median base arms are now scheduled: coffee completed26/16;
can1024 GPU0, square1022 GPU0, threading1024 GPU6, tool1024 GPU2 running.
Tool base P2.5 still on1022 GPU2. Tool controls on1022 GPU4/1024 GPU4.
Threading Pbase round1 on1022 GPU3; round2 on1024 GPU7.

Next action is MONITOR and finish these experiments; no new arm is currently
required while all11 cards are reserved. If direct-median works on the
five base tasks, validate its round1/2 predictions. If it fails, report the
counterexample and continue supported task-specific transfer outcomes.
Existing can natural-quantile transfer passed all three checkpoints.
Coffee P6 and jointC/R both failed strict minimum at round2; preserve these
negative outcomes. Square early checkpoint absence remains unresolved.

---

# Latest checkpoint: 2026-09-24 04:46 CST

Read LIVE_STATE.json in this directory for the CURRENT GPU map and next
steps. This supersedes all older maps below. All nine investigation cards
are active; no original task controllers remain except tool P2.5.

New completed results:
- can natural_q: round1 15/DP13 (minimum), round2 4/DP2 (excellent but small).
  Base reference k2.5 was15/DP8. This is a task-specific positive transfer.
- coffee P6: round1 16/DP12, round2 9/DP11 (fails).
- coffee joint C/R preserving excellent base P6 C=9.84748: round1 17/DP12,
  round2 11/DP11 (ties, fails strict minimum). New C/R mode actually ran.
- coffee direct DP KL median: kappa.09548945248953886 eta25.5252,
  R.0091442, rescued26/DP16 (excellent). This promising direct rule now
  runs at base on can, square, threading, tool_hang as well.
- tool fixedeta k1:41/DP35 (minimum), whereas Rmatched-k1:34/DP35 (fails).
  Tool p2.5 still running. Tool direct median kappa.007597838207715576,
  eta155.19147, R.00919883, C.91926; rollout running.
- threading round2 Pbase: kappa20 eta.5866026880 R≈.01022, initial eval
  running. Round1 Pbase initial59/100, DP running then SCOUT.

Next: tool fixed-task transfer should preserve successful k1's P=2.6224248560499186
AND R=.007156729252424086. pbase_anchor.json written in both rounds.
No tool pbase calibration/eval launched yet; wait for an actually free card.
Use kappa_pcalib.py --potential-cap ... --target-r ... --out potential_pbase.json,
then kappa_eval_arm.py directly on the round source with arm pbase once the
initial eval is complete. DP controls run separately on1022 GPU4/1024 GPU4;
do not let checkpoint_probe try to restart their incomplete DP directories.

New script kappa_diversity_eval.py handles direct-median pipeline. New
kappa_pcalib.py --c-reference does nested R and C calibration. Updated
checkpoint_probe accepts positive custom R targets within10%, supports
--controls-only. Syntax checks passed; old P cache reuse regression passed;
C/R runtime succeeded on two coffee checkpoints. No need repeated checks
without new changes. Seven scalar-solver unit tests had already passed.

All new scripts are deployed. Existing tracked C-calibrator/campaign fixes
remain LOCAL ONLY. README/METHODS.md and snapshot need final updating.
No final response yet: finish running arms and necessary round validation.

---

# Current checkpoint (2026-09-24 04:24 CST)

This section supersedes older running-job notes below.

Completed base counts: can DP8/k1 10/k2.5 15/k5 17/Rmatched-k1 16;
square DP9/k1 19/k2.5 27/Rmatched-k1 24;
coffee DP16/k1 19/k2.5 23/k5 24/Rmatched-k1 21/P6 25;
threading DP16/k1 19/k2.5 26/k5 11/Rmatched-k1 22;
tool_hang DP35/k2.5 18/k5 3 (k1, Rmatched-k1, P2.5 pending).
Common P6 is therefore NOT supported on all five tasks (tool_hang fails).
Do not infer that all possible common rules are impossible.

Coffee round1 P6: kappa20 eta0.3, R0.009877, initial79, DP12,
SCOUT16 (minimum, not excellent), pass@5 .95 vs .91.
Coffee round2 P6: kappa20 eta0.3, R0.009999, initial87, DP11;
SCOUT pending. Excellent threshold17 exceeds13 initial failures.
Can natural-quantile transfer: fixed base q=.406328125; round1 kappa13.826,
eta.9669, initial65, DP13; round2 kappa7.75282, eta1.775, initial82,
DP2. Both SCOUT arms pending. Round1 P6 failed core calibration and was
NOT rolled out. Threading round1 Pbase=11.732053760076715 calibrated to
kappa11.892071150027213, eta.986544195041237, R.01004472812874251.

GPU map (check UUID/memory AND live controllers before every new use):
1022 GPU0: can round2 natural_q; GPU2: tool base P2.5;
GPU3: threading round1 Pbase (after completed five-task DP-diversity probe);
GPU4: tool round1 CONTROLS ONLY (after coffee round1 fully completed).
1024 GPU0: coffee round2 P6; GPU2: can round1 natural_q;
GPU4: tool base Rmatched-k1; GPU6: original tool base controller k1;
GPU7: coffee base kl_median, then queued threading round2 Pbase (&& chain).
1022 GPU7 forbidden. 1024 GPU7 permitted. Other GPUs belong to others.

The five-task DP-diversity core probe completed. Primary candidate kappa
is pooled median KL(final independent DP draw || first-guidance anchor
from another draw), eight draws, same128 core obs. Medians:
can4.838144, square.921471, coffee.09548945248953886,
threading.010235, tool.007597838207715576. Coffee pilot eta25.5252,
R.0091442 is running; this statistic is NOT validated by core data alone.
Its completion releases the queued threading round2 Pbase calibration.

New --controls-only option in kappa_checkpoint_probe.py freezes initial
scenes and runs DP without pretending guidance is calibrated. The arm
runner skips R convergence only for --guide-off. Both new scripts were
syntax checked and deployed; tracked campaign fixes are still local only.

Keep working through final rollout results; do not finalize from pending
arms. Consider tool round2 controls when a current tool card is fully free,
then run task-specific round1/2 only after a useful base point is confirmed.
Square round1/2 checkpoints remain missing; user clarification is pending.

---

# Active investigation — not a final conclusion

The user's task is still incomplete. Continue experiments, then validate
the selected calibration on round1–2; do not stop at the current base results.

## Latest decision

Investigate P=eta*kappa jointly with R≈0.01. Cost rescaling f→c f,
kappa→c kappa, eta→eta/c preserves the actual injection exactly, conditional
on a pure scalar cost change. Shared P≈6 is a hypothesis, not an established
cross-task result. Both P and R allow 10% relative error. The hypothesis and
five-task result mapping were recorded on the server before full base results.

P=6 base mapping: can k2.5, square k2.5, coffee p6, threading rmatched_k1,
tool_hang k2.5. Coffee's newly solved pair is kappa=7.071067811865476,
eta=0.8485281374238569, R=0.009162235188661109. Its rollout completed with
25 rescues versus DP16 (excellent, 1.5625x), pass@5=0.94, exit_code=0.
Because coffee's base candidate passed, its round1 P=6 confirmation has
started on 1022 GPU4 while the remaining base tasks continue. This remains
useful as fixed-task transfer if the common P fails on another task.
Coffee round1 P=6 calibration has converged to kappa=20, eta=0.3,
R≈0.009877; its fresh initial eval has started successfully.

Important negative result: can round1 P=6 failed to reach R=0.01 in 16
core probes; maximum observed R≈0.004606 at kappa=10. Natural mean KL
grew from 19.8643 to 67.7245. No environment rollout used that rejected pair.
The replacement is natural KL quantile transfer: preserve the base cap's
CDF value q=0.406328125, giving round1 kappa=13.826037921458482, then
eta≈0.9669 and R≈0.01023. Its paired eval is running on 1024 GPU2.
This method is guarded against extreme base quantiles (.05<q<.95), so it
does not automatically apply to coffee/threading/tool_hang where natural
KL is almost always below the base cap.

Tool_hang fixed-eta k5 completed with **3** rescues versus DP **35**, so a
smaller task-specific P=2.5 is now being calibrated and evaluated. Do not
infer the final P=6 result from k5: P=6 maps to k2.5, still running.

## Completed rescue counts

- can: DP 8; fixed-eta k1=10, k2.5=15, k5=17; R-matched k1=16.
- square: DP 9; fixed-eta k1=19. Other arms still running.
- coffee: DP 16; fixed-eta k1=19, k2.5=23, k5=24; R-matched k1=21; p6=25.
- threading: DP 16; fixed-eta k1=19, k2.5=26, k5=11; R-matched k1=22.
- tool_hang: DP 35; fixed-eta k5=3. Other arms running.

Strict excellent thresholds: can 13, square 14, coffee/threading 25,
tool_hang 53. The generic report also records whether thresholds are
attainable given each round's failure count.

## Current GPU ownership by this investigation

Always recheck physical UUID and idle memory before new allocations. A GPU
briefly empty between phases of a live controller is still reserved by it.
Never use port1022 GPU7. No other person's files or jobs may be changed.

- 1022 GPU0: tool_hang k2.5, independent arm.
- 1022 GPU2: tool_hang P=2.5 calibration, then arm p2.5 (reused after k5).
- 1022 GPU3: square k2.5, independent arm.
- 1022 GPU4: coffee round1 P=6 calibration → initial eval → DP → SCOUT.
  Session kappa_coffee_r1_p6; root logs coffee_round1_p6_calib.driver.log
  and coffee_round1_p6.driver.log. Base coffee p6 is fully completed.
- 1024 GPU0: coffee round2 P=6 calibration → initial eval → DP → SCOUT.
  Session kappa_coffee_r2_p6; threading base controller fully completed.
- 1024 GPU2: can round1 natural_q transfer → initial eval → DP → SCOUT.
  Session kappa_can_r1_natural; threading rmatched_k1 fully completed.
- 1024 GPU4: tool_hang rmatched_k1, independent arm.
- 1024 GPU6: tool_hang original base controller, currently k1 after DP.
- 1024 GPU7: square rmatched_k1, independent arm. This GPU7 is allowed.

Square's original controller stopped safely after k1 because independent
k2.5 had already created its output directory. This expected queue stop
is not an evaluation failure. Square fixed-eta k5 has not started.

## Paths and tools

Server repo: /mnt/workspace/baojiachun/scout
Experiment: experiments/2026_09_24_kappa_calibration
Python: /mnt/workspace/baojiachun/.venv_mg/bin/python
Use ssh port1022 directly. Access port1024 via nested SSH through1022,
with UserKnownHostsFile=/mnt/workspace/baojiachun/.ssh_known_hosts_1024.
SCP requires -O. All newly deployed scripts are scripts/calibration/kappa_*.
Existing campaign scripts and thm2_c_calib.py fixes remain LOCAL ONLY.

New scripts:
- kappa_pcalib.py: joint P,R core calibration; checks explicit GPU UUID
  and occupancy for new measurements; --reuse-only uses compatible cached
  measurements without GPU. Source is a task directory with checkpoints.json
  or diagnostics/natural.json. Output e.g. source/potential_p6.json.
- kappa_checkpoint_probe.py: round evaluation lifecycle; accepts --source,
  --gpu, --gpu-uuid, --calibration source/potential_p6.json --arm-name p6.
  It freezes 100 eval scenes, runs DP, then the supplied calibrated SCOUT
  pair without redoing eta. Coffee round1 is the first invocation.
- kappa_eval_arm.py: individual arm on a frozen failure set. --guide-off
  supports the paired DP control. --fixed-eta explicitly accepts a fixed
  dose diagnostic. Independent completed arms have exit_code.txt=0.
- kappa_report.py: checks paired metadata and exact per-scene rescue counts.
- kappa_search.py: bounded log-space solver; direction="unknown" explores
  both directions. Seven numerical tests pass.
- kappa_action_sensitivity.py: completed CPU-only base/round1/round2 probes.
- kappa_natural_transfer.py: transfers the successful base kappa's quantile
  from base diagnostics/natural.npz to round diagnostics/natural.npz.
  kappa_diagnostics.py now supports --natural-only and round checkpoints.json.
  kappa_checkpoint_probe.py --kappa-from natural_transfer.json --arm-name
  natural_q recalibrates eta at the predicted kappa before paired evaluation.

Round checkpoint manifests and sensitivity outputs exist at round1/task
and round2/task for can, coffee, threading, tool_hang. Square round1/2 were
not found. Its ORBIT early checkpoints are listed in an old deletion manifest.
An async question asking the user for other retained square paths is pending.

Tool_hang P=2.5 converged to eta=8, kappa=0.3125, R=0.009972655597308719;
its p2.5 rollout is running. Its C_mean is 0.705979 versus 6.37995 at k2.5.

Primary next steps: finish the base P=6 comparison, inspect tool_hang's
low-cap outcome, and continue round1–2 transfer for candidates whose base
performance has passed. Coffee round1 and round2 are running. Can round1
natural_q is running; launch can round2 natural_q on the next suitable free
GPU, using the same BASE q, not round1 as a drifting reference. Threading's
excellent base anchor is eta*kappa=4.692821504030686*2.5≈11.73205; its P=6
base is only minimum quality. Consider validating this task-specific P
on round1–2 after GPUs free. All transfer methods still need rollout proof.
Do not claim a universal rule cannot exist from these finite probes. The
S-ratio alternative is also unvalidated and changes differently by state.

Local README contains definitions, bug fixes, measurements, and hypotheses.
Local snapshot/ is downloaded metadata from an earlier point, not live data.
The functions store key kappaMonitorCommand contains a compact remote monitor.


## Update 2026-09-24 06:36 CST

Five base median arms finished: can18/8, square23/9, coffee26/16, threading17/16, tool33/35. Thus median rule fails universal base criterion. Can median round1=15/13 and round2=3/2; natural-quantile round1=15/13, round2=4/2. Threading median round1=18/17, round2=25/23; all three checkpoints meet minimum only.

New sequential-C mode in kappa_pcalib.py has completed core calibration on coffee/threading rounds1/2. It primes eta at previous kappa to R=.01, holds eta fixed, and roots C against fixed base kappa2.5. R_target=null explicitly; evaluate with --fixed-eta. Coffee round2 kappa1.146255054 eta1.269377534 R.008109848 C4.014093 vs target4.218468; rollout9/DP11 fails, so no round1 environment run. Threading round2 kappa2.973017788 eta1.191945895 R.009867104 C9.435774 vs target9.905537, rollout running. If it beats23, launch the already predetermined round1 sequential_c.json arm. Otherwise no additional round1 rollout needed.

Active jobs only: 1022 GPU0 tool round1 joint_c (session kappa_tool_r1_joint_c), 1022 GPU2 threading round2 sequential_c (kappa_threading_r2_sequential_c), 1024 GPU4 tool round2 joint_c (kappa_tool_r2_joint_c). Others finished; always recheck UUID/memory/controllers before allocation.

Tool joint C uses base kappa1 / eta2.622424856 / R.007156729 / C.866016209. Round1 calibrated kappa.353553391 eta13.778325904; round2 kappa.3692 eta13.83. Both eval running, no final count yet.

Do not finish before remaining authorized evaluations are handled. Then refresh snapshot via tar, rebuild RESULTS.md with --final, update METHODS/README and final LIVE_STATE and server latest code hash manifest. Reports are currently stale: last local metadata snapshot around06:23, 54 completed rows. functions store kappaSnapshotPackCommand and kappaMonitorCommand have reusable commands. Latest server core/eval hashes remained equal to initial audit. No active server campaign script was overwritten.


## Update 2026-09-24 06:55 CST — supersedes earlier job maps

Threading sequential C round2 completed20/DP23; no round1 rollout. Coffee sequential C round2 already9/11. Added an important bounded fixed-base-kappa control using existing core measurements: coffee kappa2.5 / R.01, tool kappa1 / R.007156729. Coffee round2 fixed cap completed9/11; no round1 rollout. Its kappa1? NO: coffee fixed cap is2.5, eta1.1066641191457374, R.010925397647672464.

ONLY active jobs now all tool_hang: (1) 1022 GPU0 round1 joint_c, session kappa_tool_r1_joint_c; (2)1024 GPU4 round2 joint_c, kappa_tool_r2_joint_c; (3)1022 GPU3 round1 fixed_base_kappa, kappa_tool_hang_round1_fixedbase; (4)1022 GPU4 round2 fixed_base_kappa, kappa_tool_hang_round2_fixedbase. New fixed-cap jobs started06:44; joint C jobs earlier. No more candidate families planned. Finish these, audit and deliver. No coffee/thread conditional round1 work remains.

All median runs are complete: base can18/8,square23/9,coffee26/16,thread17/16,tool33/35. Can rounds1/2 median15/13,3/2. Thread rounds1/2 median18/17,25/23. Can natural q15/13,4/2 (base15/8). Thus useful fixed-task candidates forcan andthread; no universal five-task rule found.

Added --calibrate-only to kappa_diversity_eval.py, deployed, syntax checked, cached smoke test oncan round1 passed. It needs only checkpoints.json (task,dp_ckpt,vib_ckpt,core_hdf5,eta_initial), not initial eval. Added cache identity and actualR-band checks. Latest code_manifest_latest.json predates this helper change; refresh hashes at end.

Latest local snapshot around06:52; RESULTS.md builder validates61 completed rows, table still marks tool jobs running and may be stale for coffee fixed cap (completed afterward). build_report.py includes sequential/fixed-base controls and core rejections. METHODS.md updated throughcoffee fixed negative. README preserved as historical notes with links to current results. Final snapshot refresh, final report, code hash manifest, LIVE_STATE completion, WORKING_STATE final summary still needed.


## Update 2026-09-24 07:12 CST

Tool round2 joint C completed12/DP37 (fails). Only THREE tool jobs remain: 1022 GPU0 round1 joint_c; 1022 GPU3 round1 fixed_base_kappa; 1022 GPU4 round2 fixed_base_kappa. All started successfully, no errors; source output paths under E/roundN/tool_hang. Wait for complete summaries and exit0. No further candidate families planned.

Important correction: fixedeta base k1 arms ALL FIVE passed minimum:10/8,19/9,19/16,19/16,41/35. This is a legitimate two-stage common preset: eta calibrated at kappa2.5 toR.01, then hold eta and usekappa1 (finalR varies.0035-.0078). Do not omit this positive base evidence or falsely claim no common BASE rule. To confirm across rounds, we tested that exact preset on coffee round2: kappa1 eta1.1066641191457374, actualR.0070359263663520005, C2.79103207588; final rescue9/DP11 -> fails cross-round rule. Files protocol fixed_k1_after_r25_protocol.json, calibration round2/coffee/diagnostics/k1.json, arm fixed_k1_after_r25. No further round tests needed for this rejected global preset.

Added --eta optional override to kappa_diagnostics.py (backward-compatible defaults; used above), deployed/syntax checked and actual core run passed. Latest code hashes must refresh before finish.

New diagnostic limitation confirmed: rollout mean_inject telemetry in policy.py is BATCH L2 norm, not elementwise MAE used forR. Batchsize varies. Cannot directly infer onlineR or attribute failures to injection amplification from existing telemetry. Documented in METHODS.

Report builder now includes fixed-k1 two-stage base column, matchedRk1 column, P6,median, plus round2 rejection. Also writes validated_candidates.json with full measured pairs forcan naturalq/can median/thread median, all3checkpoints passminimum. File generated and validated. Latest local metadata remains06:52/61completedrows; finalrefreshstillrequired (expected67completedrows once remaining3finish, verify).
