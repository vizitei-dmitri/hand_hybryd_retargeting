# Fingertip-only (precision grasp) in-hand reorientation and finger gaiting

Scope note: palm-supported / table-supported reorientation is deliberately out of scope here except
where a paper compares directly against the fingertip-only case. Everything below concerns an object
held **only** by fingertips, with force closure that must be maintained at all times.

## Q1. What work addresses in-hand reorientation with a FINGERTIP precision grasp and no palm or table support?

### Takeaway
The fingertip-only, no-support case is a distinct and much smaller literature than palm-supported
reorientation, and it is essentially solved only for *continuous spinning about a fixed axis*
(Columbia/Khandate, Bristol/AnyRotate, Berkeley/Hora) plus one goal-conditioned exception on a cube
(DLR). Every successful system needed substantial extra machinery beyond vanilla RL: a
sampling-based planner to seed exploration, a learned state estimator, dense tactile sensing, or
physics-prior reward shaping.

### Cited Findings
- **Khandate et al., ICRA 2022 ("On the Feasibility of Learning Finger-gaiting In-hand Manipulation with Intrinsic Sensing")** is explicitly the precision-grasp-only setting: they "learn finger-gaiting only via precision grasps" using purely on-board proprioceptive and binary tactile feedback, with no external object pose sensing; the demonstrated skill is rotation about a *single* axis, and instability of the task is identified as the core obstacle, addressed by engineering the initial-state distribution — [arXiv:2109.12720](https://arxiv.org/abs/2109.12720); [author page](https://gagkhan.github.io/)
- **Khandate et al., RSS 2023 (SBRL)** maintains a "precision grasp of the manipulated object at all times" with no support surfaces, requiring "at least three fingers in contact with the object" so stability does not rest on friction assumptions. Hand: five identical fingers, each a roll joint plus two flexion joints, 15 fully actuated position-controlled joints; real hardware runs 0.6 rad/s setpoints with 0.5 N·m torque limits — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- Same paper's object taxonomy: easy (sphere, cube, cylinder), moderate (elongated cuboids), hard (concave L-/U-shapes); the authors state in-hand manipulation of the "hard" category "has not been previously demonstrated" — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **DLR (Sievers et al., ICRA 2022)** learned purely tactile in-hand cube manipulation on the torque-controlled DLR-Hand II with the **palm facing downwards**, "demanding permanent force-closure", with continuous regrasping and no external sensors; the real-robot policy reached **more than 46 full 2π rotations of the cube in a single run**, and tolerated disturbances including different cube sizes, changed hand orientation, and a finger being pulled — [arXiv:2204.03698](https://arxiv.org/abs/2204.03698); [DLR-Hand II topic page](https://www.emergentmind.com/topics/dlr-hand-ii)
- **DLR (Pitz et al., Humanoids 2023)** moved from spinning to *goal-conditioned* reorientation: reaching specific goal orientations rather than "infinitely spinning it around an axis", over the 24 orientations of a cube in π/2 increments, upside-down, with permanent force closure and no external sensors. **92% success in simulation across all 24 goals**, and all 24 reachable on the real hand with high success via zero-shot transfer. Architecture is modular: the policy sees only a 0.5 s observation window but receives explicit cube state from a **deep differentiable particle filter** — [arXiv:2303.04705](https://arxiv.org/abs/2303.04705)
- Later DLR line: shape-conditioned purely tactile manipulation of various objects — [arXiv:2407.18834](https://arxiv.org/html/2407.18834v1); time-optimal and speed-adjustable tactile in-hand manipulation, reported as the fastest vision-free tactile manipulation — [arXiv:2411.13148](https://arxiv.org/html/2411.13148v1); composing grasping and in-hand manipulation by scoring with an RL critic — [arXiv:2505.13253](https://arxiv.org/html/2505.13253). Aggregated DLR-Hand II results include up to **99% success in π/2 raster reorientation** — [DLR-Hand II topic page](https://www.emergentmind.com/topics/dlr-hand-ii)
- **Yang et al. (Bristol), AnyRotate, CoRL 2024**: 16-DoF Allegro Hand, four fingers with vision-based tactile fingertips, gravity-invariant multi-axis rotation. It is a precision fingertip grasp: the reward requires "the number of tip contacts is greater or equal to 2" and **penalises non-fingertip contact**. Objects 10–89 g, up to 90×80×76 mm. 1.77 rotations/episode in sim (dense touch), 1.57 real-world palm-up, ~0.9 for thumb-up/thumb-down orientations; 30 s episodes at 20 Hz (600 steps) — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- **Qi et al., Hora (CoRL 2022)** rotates objects about the z-axis "using only fingertips", trained in sim on cylinders only, transferring to 30+ objects spanning **4.5 cm to 7.5 cm** and 5–200 g. Code is public — [Hora paper](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf); [github.com/HaozhiQi/hora](https://github.com/HaozhiQi/hora)
- **Qi et al., CoRL 2023 (vision + touch)** describes the prior fingertip/hand-down work as follows: "Previous work learns a finger-gaiting behavior efficiently and transfers to a real robot when hand facing downwards, but they only use fingertips and the objects considered are all cubes"; and argues that *not* using a supporting surface is the harder setting because it "allows constant tactile feedback on fingertips and enables a natural finger-gaiting to emerge" — [arXiv:2309.09979](https://arxiv.org/html/2309.09979v2)
- **2026 work — "Robust In-Hand Manipulation via Priors in Reinforcement Learning and Mechanical Design"**: 16-DoF Allegro V4, **fingertip contacts only, no palm support**, proprioception only at execution (no tactile, no vision). Objects: cylinder R = 40 mm, a cuboid, a sphere; four palm orientations 0°/45°/90°/180° — [arXiv:2607.12105](https://arxiv.org/html/2607.12105v1)
- Chen et al.'s hand-down configuration is the well-known contrast case: "Visual Dexterity" reorients novel/complex shapes with the hand facing downward, but with vision-based full object pose rather than intrinsic sensing — [arXiv:2211.11744](https://arxiv.org/pdf/2211.11744)

### Inferences
- The only published *goal-conditioned* (reorient-by-a-target-angle) fingertip-only result on a cube is the DLR line, and it needed an explicit object-state estimator (differentiable particle filter) feeding the policy. Continuous-spin formulations (Khandate, AnyRotate, Hora) sidestep the goal-reaching credit-assignment problem entirely by rewarding angular velocity.
- A policy asked for a bounded 20° reorientation is in the harder (goal-conditioned) class; the literature suggests it needs either an object-state estimate as an explicit input or a reset/exploration mechanism that already visits post-gait states.

### Gaps
- I could not confirm the DLR cube edge length from primary sources (the abstracts do not state it; only "different cube sizes" as a robustness axis). Treat any specific number as unverified.
- No paper found that reports a bounded small-angle (e.g. 20°) fingertip-only reorientation benchmark; the field reports either full revolutions or π/2-raster goals.

## Q2. FINGER GAITING specifically: which papers learn or plan it, and by what method?

### Takeaway
Finger gaiting for reorientation has been attacked by five distinguishable methods: (a) RL with a
sampling-based-planner-generated reset distribution, (b) RL guided by sub-optimal sub-skill
controllers, (c) RL with dense tactile sensing, (d) RL with a modular state estimator, and (e) pure
planning/optimisation (smoothed contact models, compliance-enabled multi-modal planning,
contact-implicit MPC). Pure model-free RL from a fixed initial state does **not** learn it.

### Cited Findings
- **(a) RL + planner reset distribution** — Khandate et al. RSS 2023: paths from a non-holonomic RRT are turned into reset distributions for model-free RL under full dynamics. Baseline comparison is decisive: Fixed Initialization (FI) "did not learn finger-gaiting even with zero gravity"; Explored Restarts (ER) "fails to learn a viable policy even for simple objects"; Stable Grasp Sampler (SGS) "fails to learn a viable policy" on hard objects; only G-RRT and M-RRT learn manipulation of the hard (concave) objects — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- Journal/extended version with imitation pre-training: **R×R: Rapid eXploration for RL via Sampling-based Reset Distributions and Imitation Pre-training**, Autonomous Robots (RSS 2023 special issue) — [arXiv:2401.15484](https://arxiv.org/html/2401.15484)
- **(b) RL guided by sub-skill controllers** — Khandate, Mehlman, Wei, Ciocarlie, "Value Guided Exploration with Sub-optimal Controllers for Learning Dexterous Manipulation" (IROS 2024): first to "demonstrate learning hard-to-explore finger-gaiting in-hand manipulation skills" **without** an exploratory reset distribution — [arXiv:2303.03533](https://arxiv.org/pdf/2303.03533); [project page](https://roamlab.github.io/vge)
- **(c) RL + dense tactile** — AnyRotate: no explicit slip detector; "rich multi-fingered tactile sensing" yields emergent reactive behaviour that detects slipping and produces adaptive finger-gaiting; dense tactile feedback (contact pose and force) beat simpler tactile representations — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- **(d) RL + modular estimator** — DLR Pitz et al., as above: short-window policy plus differentiable particle filter, continuous regrasping with permanent force closure — [arXiv:2303.04705](https://arxiv.org/abs/2303.04705)
- **(e1) Planning with smoothed contact models** — Pang, Suh, Yang, Tedrake, "Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasi-dynamic Contact Models", IEEE T-RO 39(6):4691–4711, 2023: classical sampling-based motion planning becomes effective once contact modes are abstracted by smoothing; results "comparable to reinforcement learning with dramatically less computation", including in-hand cube reorientation, with plans generated in under a minute — [ResearchGate record](https://www.researchgate.net/publication/361479950_Global_Planning_for_Contact-Rich_Manipulation_via_Local_Smoothing_of_Quasi-dynamic_Contact_Models)
- **(e2) Compliance-enabled gaiting** — Morgan et al., "Complex In-Hand Manipulation via Compliance-Enabled Finger Gaiting and Multi-Modal Planning": a compliant underactuated hand with **no tactile sensors and no joint encoders**, exploiting "orthogonal safe modes"; claims **complete SO(3) finger gaiting control without a support surface** for convex and non-convex objects, with perturbation-rejection and long-trajectory tests; contacts are "freely broken and remade" — [arXiv:2201.07928](https://arxiv.org/abs/2201.07928)
- **(e3) Geometric regrasp planning** — Sundaralingam & Hermans, "Geometric In-Hand Regrasp Planning: Alternating Optimization of Finger Gaits and In-Grasp Manipulation": an optimisation-based planner that alternates finger gaiting (relocating one finger while the others stabilise) with in-grasp manipulation (repositioning the object with contacts maintained); plans are collision-free with guaranteed kinematic feasibility; evaluated on 5 objects × 5 in-hand regrasp goals — [arXiv:1804.04292](https://arxiv.org/abs/1804.04292)
- **(e4) Contact-implicit MPC** — "Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking" (2025): contact-implicit MPC at the high level plus a hand force-motion model tracker at the low level, jointly optimising motion and contacts where "multiple fingers dynamically make and break contact"; five real-world tasks; claims better accuracy, robustness and real-time performance than approaches "that rely on large-scale training" — [arXiv:2505.04978](https://arxiv.org/abs/2505.04978)

### Inferences
- The methodological consensus is that the bottleneck in fingertip-only gaiting is *exploration*, not
  policy capacity: the same PPO-class learner succeeds or fails purely as a function of where episodes
  start (Khandate's FI/ER/SGS vs G-RRT/M-RRT ablation is the cleanest evidence in the literature).
- A practical corollary for a policy stuck at 8°: the failure is likely that post-gait states (one
  finger lifted, object partially rotated) are never visited from the training reset distribution, so
  the policy never learns the release-and-replace step at all.

### Gaps
- No head-to-head benchmark comparing (a)–(e) on the same hand/object, so relative performance across
  method families cannot be quantified.

## Q3. Khandate et al. in detail: contribution, planner-generated resets, numbers, code

### Takeaway
The contribution is an exploration mechanism, not a new RL algorithm: two non-holonomic RRT variants
find kinematically feasible, stability-verified gaiting states, and the top paths' states become a
uniform reset distribution for ordinary model-free RL. This is what unlocked concave/elongated objects.

### Cited Findings
- **Two RRT variants.** *M-RRT* (manipulation-specific) uses analytical kinematic constraints: it projects the desired direction of motion onto the manifolds defined by contact constraints via Jacobians, with the update Δx_proj = (I − NᵀN)Δx_des, where N collects the contact constraint matrices, enforcing that "the object and the fingers that maintain contact with it must move in unison". Stability is checked by a **quadratic program** testing "whether the hand [has] the ability to create internal object forces by applying normal forces". *G-RRT* (general-purpose) is model-agnostic and uses explicit physics simulation: K_max random actions per iteration, keeping the transition that best approaches the sampled target, and verifying stability by "advancing the simulation for an additional 1 s with no change in the action" — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **Reset-distribution construction.** Trees are grown to 10^5 nodes; they "select the top ten paths from the RRT tree that achieve the largest angular change for the object around the chosen rotation axis", giving roughly **2×10^4 states**, and use a **uniform distribution over these states** as the RL reset distribution — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **Reward.** Object angular velocity about the z-axis, credited only "if the hand [is] re-orienting the object with at least three fingertip contacts", plus penalties on object translational velocity and on deviation from the initial position; **early termination if fewer than two contacts**. No force-closure or grasp-quality term in the reward — stability is enforced by state filtering in the planner, not reward shaping — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **Sim results (learning curves).** Easy objects: G-RRT, M-RRT and SGS comparable. Moderate: all three learn but "the policies learned via G-RRT exploration are more effective". Hard (concave): "Only G-RRT and M-RRT are able to learn manipulation" — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **Real robot (median revolutions / rotation speed).** Cylinder 5 rev at 0.42 rad/s; cube 4.5 rev at 0.44 rad/s; cuboid 1.5 rev at 0.44 rad/s; L-shape 1.5 rev at 0.24 rad/s — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **Sensing and sim-to-real.** Intrinsic sensing only: joint positions and setpoints plus **binary contact/no-contact** touch, no global object pose. Randomisation: 0.05 s simulated latency, joint origins ±0.1 rad, friction 1–40, 1 N perturbation forces; real tactile data thresholded at 1 N below which contact readings were unreliable — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **Code.** The RSS paper does not state a code release; it points to the project site [sbrl.cs.columbia.edu](https://sbrl.cs.columbia.edu) for videos. The earlier ICRA 2022 work lists project page roamlab.github.io/learnfg and the IROS 2024 follow-up roamlab.github.io/vge — [author page](https://gagkhan.github.io/)

### Inferences
- The recipe is directly portable: run a kinematics-level RRT (with a QP force-closure check) from the
  current fingertip grasp, keep the branches that achieve the largest rotation about the goal axis, and
  reset RL episodes uniformly over those states. This converts "20° with a gait" from a
  long-horizon-exploration problem into a local tracking problem.
- Khandate's contact gating (reward requires ≥3 contacts, terminate below 2) is a concrete answer to
  the grasp-degradation trade-off: it tolerates a transient two-finger phase but never rewards it.

### Gaps
- I found no confirmed public code repository for the SBRL/R×R planner or the training code; only project
  pages with videos. Worth checking github.com/roamlab directly.

## Q4. How large a single-step rotation is achievable without regrasping? Is there a kinematic-limit analysis?

### Takeaway
Yes, but the quantification is recent and indirect: the maintained-contact (no-gaiting) rotation
workspace of a fingertip grasp is small, and the best 2026 measurement finds that even the most
dexterous hand tested can roll a pinch grasp into only about a third of orientation directions. I
found **no** published curve of maximum single-step rotation angle versus object size.

### Cited Findings
- **KaRMA (2026), "A Kinematic Metric for Fine Manipulation Ability in Robotic Hands"** measures in-hand manipulation by rolling contact in a thumb–index precision pinch, decomposed into reachable translations (KaRMA-T), rotational coverage (KaRMA-R) and sensitivity to the initial grasp (KaRMA-S), using a standardised 10 mm-radius sphere at a nominal 200 mm reference hand length, with lengths scaled by a characteristic hand dimension. Across **16 hands spanning 3–9 DoF**, "no hand evaluated achieved more than approximately 35.5% coverage of the 228 possible orientation bins" at its best configuration — "even the most dexterous hand at its most capable position rotates the pinch axis into only about a third of the available directions." Allegro scored highest on rotation (KaRMA-R 0.335); LEAP highest on translation (KaRMA-T 0.097); Ability, Inspire and SVH (3–4 DoF) scored lowest — [arXiv:2605.15548](https://arxiv.org/html/2605.15548)
- The standard statement of the limit: constraining contacts to remain fixed on the object during manipulation limits the potential workspace, since motion is subject to the hand's kinematic topology, and **finger gaiting is one way to alleviate that restraint**; besides finger workspace limits "there also exist limits for the maximum rotation of fingertips" — [Geometric In-Hand Regrasp Planning](https://arxiv.org/pdf/1804.04292)
- The dual constraint set for a feasible in-grasp target pose: the pose must lie in the workspace determined by the hand's inverse kinematics **with joint limits**, and the fingertip contact forces must simultaneously satisfy friction-cone limits for grasp stability — [search synthesis over in-hand manipulation literature](https://arxiv.org/pdf/1804.04292)
- **In-grasp manipulation** is formally defined as moving the object from an initial to a goal pose relative to the palm "without breaking or making contacts"; Sundaralingam & Hermans relax the rigid-contact assumption (relaxed-rigidity constraints) with a cost that tries to maintain the initial grasp points, using kinematic trajectory optimisation that needs no dynamic parameters — [RSS 2017 paper](https://www.roboticsproceedings.org/rss13/p15.pdf); [Autonomous Robots version](https://link.springer.com/article/10.1007/s10514-018-9772-z)

### Inferences
- KaRMA's ~1/3-of-directions ceiling is measured for a two-finger pinch on a small sphere; a 60 mm
  cube in a five-finger hand has larger contact-point excursions per degree of object rotation
  (arc length ≈ R·θ with R ≈ 30–52 mm from the cube centre to face/edge contacts), so the
  maintained-contact angular budget shrinks roughly as 1/R for a fixed fingertip workspace. A policy
  stalling at ~8° is consistent with having exhausted the no-gaiting workspace rather than with a
  control-tuning problem.
- Because no source gives an explicit max-angle-vs-size law, the practical way to get the number for a
  specific hand is to compute it: sample fingertip grasps and run the M-RRT-style projected-motion
  integration with the QP stability check until infeasibility, which is exactly Khandate's M-RRT.

### Gaps
- No paper found that reports "maximum in-grasp rotation in degrees" as a function of object size or
  hand kinematics. KaRMA is the closest, and it reports coverage fractions on a fixed 10 mm sphere,
  not angles on a 60 mm cube.
- KaRMA explicitly "does not test larger objects or discuss feasibility boundaries as object size
  increases" — the radius is never varied beyond the nominal 10 mm.

## Q5. Role of object size relative to the hand, and number of fingers in contact. Is a 60 mm cube in a five-finger hand near a feasibility boundary?

### Takeaway
Object size bands are documented empirically (roughly 45–90 mm across Allegro-class hands) and a 60 mm
cube sits **inside** the demonstrated range rather than beyond it — but it sits in the part of the range
where reported success rates fall to 24–83% and where cube/box geometry is called out as the hard case.
Minimum contact counts are a hard design choice: 3 fingers (Khandate) or 2 fingertips (AnyRotate).

### Cited Findings
- Hora transfers to 30+ objects from **4.5 cm to 7.5 cm** and 5–200 g on an Allegro-class hand, fingertips only — [Hora, CoRL 2022](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)
- AnyRotate's object set reaches **90×80×76 mm** and 10–89 g, but the authors identify the failure mode by geometry: difficulty with "box-shaped or larger aspect ratio objects" due to ambiguous tactile signatures, plus hardware actuation weakness with fingers in horizontal positions — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- The 2026 priors paper's main test object is a cylinder of **R = 40 mm** (80 mm diameter) on a 16-DoF Allegro V4 with fingertip contacts only; even so, success rates in its ablation range from **24% to 83%** depending on reward and fingertip morphology — [arXiv:2607.12105](https://arxiv.org/html/2607.12105v1)
- Contact-count requirements: Khandate requires ≥3 fingers in contact so that stability does not depend on friction properties, rewards rotation only with ≥3 fingertip contacts, and terminates the episode below 2 — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486). AnyRotate requires ≥2 tip contacts and penalises non-fingertip contact — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- KaRMA's DoF ranking shows that fine in-hand rotation capability is dominated by finger DoF and kinematic topology: 8-DoF LEAP/Allegro at the top, 3–4 DoF Ability/Inspire/SVH at the bottom — [arXiv:2605.15548](https://arxiv.org/html/2605.15548)
- Qi et al. note the object-class narrowness of the earlier hand-down fingertip work: "the objects considered are all cubes" — [arXiv:2309.09979](https://arxiv.org/html/2309.09979v2)

### Inferences
- A 60 mm cube is not past a published feasibility boundary for a five-finger hand — it is between
  Hora's 45–75 mm band and the 80 mm cylinder of the 2026 priors paper — but it is squarely in the
  regime where the literature reports partial success rather than reliability, and cube/box geometry is
  the specifically-flagged hard shape (flat faces and edges give discontinuous contact normals and
  ambiguous tactile signatures).
- With a 60 mm cube, holding ≥3 fingertips on *distinct faces* while one finger relocates is the crux;
  the literature's two answers are to demand 3 contacts and tolerate a transient 2 (Khandate) or to
  design the reward to maximise tip contacts with a stability curriculum (AnyRotate).

### Gaps
- I found no study that sweeps cube edge length for a fixed hand and reports where fingertip-only
  reorientation breaks down. The claim "60 mm is near the feasibility boundary" is not directly
  supported or refuted by any source I located; it is plausible but unverified.

## Q6. Grasp-quality metrics (force closure, smallest eigenvalue of the grasp matrix/Gramian, grasp wrench space) as an RL reward term — did it help?

### Takeaway
Yes, and the strongest evidence is from 2026: using λ_min of the grasp Gramian as a dense reward term
for fingertip-only in-hand rotation more than doubled success rate (24% → 56%) with stock fingertips
and raised it from 64% to 83% with task-aligned fingertips. Older fingertip-gaiting work (Khandate,
DLR) treats force closure as a *constraint* or a *state filter* rather than a reward.

### Cited Findings
- **Definition used as a reward.** "The eigenvalues and eigenvectors of **M** determine the principal axes of the corresponding wrench ellipsoid"; the prior is q_grasp = λ_min(**M**) with **M** = **G G**ᵀ the **grasp Gramian**, measuring "wrench generation in the weakest direction", i.e. worst-case disturbance resistance. Reward term: r_gq(t) = β_gq · ψ(λ_min(**M**_h(t))) — dense shaping that "encourages well-distributed contacts during training without requiring force measurements" — [arXiv:2607.12105](https://arxiv.org/html/2607.12105v1)
- **Ablation (grasp-quality reward on/off), Allegro V4, fingertip-only:** default fingertips 24% → **56%** success; task-aligned cylindrical fingertips 64% → **83%**; average cumulative rotation 0.78 → 1.25 turns (default tips) and 1.06 → 1.27 turns (aligned tips). Largest gains at 90° palm tilt, where off-axis gravity loading dominates — [arXiv:2607.12105](https://arxiv.org/html/2607.12105v1)
- **Fingertip morphology ablation with a fixed policy:** aligned cylindrical 83% success / 1.27 turns, spherical 53%, flat 49%, orthogonal cylindrical 43%. The aligned cylindrical tip has curvature κ_f along the rolling direction and zero curvature orthogonal to it, creating anisotropic rolling mobility that "passively suppress[es] off-axis disturbances while preserving task-aligned rolling" — [arXiv:2607.12105](https://arxiv.org/html/2607.12105v1)
- **Contrast — force closure as a constraint, not a reward:** DLR's task is defined by "permanent force-closure" that must hold throughout, enforced by the task/termination structure rather than by a shaped reward — [arXiv:2204.03698](https://arxiv.org/abs/2204.03698)
- **Contrast — stability by state filtering:** Khandate's planner verifies stability with a QP checking the hand's ability to create internal object forces, and the RL reward carries no force-closure term at all — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- Background on the metric family: the minimum eigenvalue of the wrench matrix "measures the strength of wrench generation in the weakest direction, providing a proxy for grasp quality that captures similar intuition to force closure analysis"; force-closure-derived "Grasp Quality Score" indices have been used as RL rewards in grasping, and a force-closure-based reward has been proposed to explicitly encourage grasp stability in in-hand manipulation RL — [survey/synthesis across grasping literature](https://h2t.iar.kit.edu/pdf/Newbury2023.pdf); [In-Hand Manipulation with Enforced Grasp Stability](https://contact-rich.github.io/assets/pdf/papers/32_In_Hand_Manipulation_with_Eenforced.pdf)

### Inferences
- λ_min(GGᵀ) is cheap to compute in simulation from contact points and normals and needs no force
  sensing, which is why it works as dense shaping. It is the single most directly transferable idea in
  this literature for a policy that stalls mid-rotation: it penalises the degenerate near-collinear /
  near-coplanar contact layouts that a rotating grasp drifts into.
- Because the 2026 paper couples the reward prior with a fingertip *shape* prior and gets its best
  numbers only from the combination, reward shaping alone should be expected to give the 24→56 style
  gain, not the 83% figure, on unchanged hardware.

### Gaps
- No ablation found that isolates alternative metrics (grasp wrench space volume, Ferrari–Canny ε,
  task-wrench-space based measures) against λ_min for in-hand reorientation.

## Q7. The trade-off where rotating the object necessarily degrades the grasp — how do successful systems resolve it?

### Takeaway
Four distinct resolutions appear in the literature: contact-count gating with a tolerated transient
(Khandate), contact-maximisation reward plus an adaptive stability curriculum (AnyRotate), mechanical
compliance that makes break-and-remake safe (Morgan et al.), and dense grasp-quality shaping that
steers the policy away from degenerate layouts before they become failures (2026 priors paper). The
2026 paper argues the trade-off is not fundamental once shaping and fingertip geometry are right.

### Cited Findings
- **Gating with tolerated transient:** reward only credits rotation with ≥3 fingertip contacts; episode terminates below 2 contacts — i.e. a brief two-finger phase during a gait is allowed but never rewarded — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- **Contact maximisation + curriculum:** AnyRotate's reward combines a rotation objective (keypoint distance to goal), a contact-maximisation term favouring tip contacts, stability terms penalising excessive angular velocity and joint deviation, and early-termination penalties; the stability terms are **scaled up through an adaptive curriculum** as the agent achieves successive rotation goals — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- **Reactive recovery instead of avoidance:** AnyRotate has no explicit slip detector; dense tactile sensing lets the policy detect unstable grasps and react, which the authors credit for robustness — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2); [Bristol press coverage](https://www.techexplorist.com/breakthrough-development-tactile-robotic-hands/85650/)
- **Compliance:** Morgan et al. use a compliant underactuated hand whose "orthogonal safe modes" allow contacts to be "freely broken and remade", achieving complete SO(3) gaiting without a support surface and without tactile sensors or joint encoders — [arXiv:2201.07928](https://arxiv.org/abs/2201.07928)
- **Dense grasp-quality shaping, and the claim of complementarity:** combining the grasp-quality reward with the aligned fingertip gave "best overall SR and T by a large margin", which the authors read as the two priors being complementary rather than competing — maintaining grasp stability *improves* rotation efficiency — [arXiv:2607.12105](https://arxiv.org/html/2607.12105v1)
- **Restricting the task:** the DLR line handles the trade-off partly by restricting goals to the 24 π/2-raster cube orientations, i.e. to symmetric targets where a gait can restore an equivalent grasp — [arXiv:2303.04705](https://arxiv.org/abs/2303.04705)

### Inferences
- Note the structural asymmetry: the systems that report the highest robustness (DLR, AnyRotate) either
  restrict goals to symmetry-equivalent orientations or reward unbounded spinning. A hard 20°
  intermediate target has neither property — the grasp must end in a *worse* configuration than a
  symmetry-restoring gait would give, which is a genuine reason this specific setting is harder than
  what the literature reports as solved.
- Practical ranking of interventions by expected payoff for a stalled policy: (1) planner-seeded reset
  distribution over post-gait states, (2) λ_min(GGᵀ) dense reward, (3) contact gating at 3 with
  termination at 1–2, (4) stability-term curriculum, (5) fingertip geometry change.

### Gaps
- No source quantifies the trade-off directly (e.g. grasp-quality metric plotted against rotation
  angle for a fingertip grasp). This appears to be an unpublished measurement.

## Q8. Non-RL solutions for this exact subproblem

### Takeaway
There is a credible non-RL stack for fingertip-only reorientation: smoothed-contact sampling-based
global planning (Pang/Tedrake), contact-implicit trajectory optimisation and MPC, geometric regrasp
planning that alternates gaits with in-grasp motion, and compliance-based multi-modal planning. For a
bounded 20° reorientation these are arguably a better fit than RL, because the horizon is short and a
planner can be asked directly whether the rotation is feasible without a gait.

### Cited Findings
- **Smoothed quasi-dynamic contact models + sampling-based planning** — Pang, Suh, Yang, Tedrake, T-RO 2023: contact-mode abstraction via local smoothing makes classical sampling-based planners work on contact-rich tasks including in-hand cube reorientation, with performance "comparable to reinforcement learning with dramatically less computation" and plans "in under a minute" — [ResearchGate record](https://www.researchgate.net/publication/361479950_Global_Planning_for_Contact-Rich_Manipulation_via_Local_Smoothing_of_Quasi-dynamic_Contact_Models)
- **Contact-implicit MPC with explicit finger make/break** — [arXiv:2505.04978](https://arxiv.org/abs/2505.04978)
- **Contact Trust Region** for dexterous contact-rich manipulation — [arXiv:2505.02291](https://arxiv.org/pdf/2505.02291)
- **Multi-finger manipulation via trajectory optimization with contact models** — [arXiv:2408.13229](https://arxiv.org/pdf/2408.13229)
- **Approximating global contact-implicit MPC via sampling and local complementarity** — [arXiv:2505.13350](https://arxiv.org/html/2505.13350.pdf)
- **Convex relaxations for contact-rich manipulation** — [arXiv:2402.10312](https://arxiv.org/pdf/2402.10312)
- **Graph-of-reachable-sets global planning for contact-rich SE(2) manipulation (2026)** — [arXiv:2601.10827](https://arxiv.org/pdf/2601.10827)
- **Geometric in-hand regrasp planning (alternating gaits and in-grasp manipulation)**, kinematically feasible and collision-free by construction — [arXiv:1804.04292](https://arxiv.org/abs/1804.04292)
- **In-grasp manipulation via relaxed-rigidity kinematic trajectory optimisation**, requiring no dynamic model of object or robot — [RSS 2017](https://www.roboticsproceedings.org/rss13/p15.pdf); [Autonomous Robots](https://link.springer.com/article/10.1007/s10514-018-9772-z)
- **Compliance-enabled finger gaiting with multi-modal planning**, complete SO(3) without support surface — [arXiv:2201.07928](https://arxiv.org/abs/2201.07928)
- **Hybrid RL + model-based regrasping (2025):** "Robust In-Hand Reorientation With Hierarchical RL-Based Motion Primitives and Model-Based Regrasping" combines learned primitives with a model-based regrasp layer — [ResearchGate record](https://www.researchgate.net/publication/397694451_Robust_In-Hand_Reorientation_With_Hierarchical_RL-Based_Motion_Primitives_and_Model-Based_Regrasping)
- Classical background on rolling plus gaiting as the two primitives of dexterous manipulation — [Dextrous Manipulation by Rolling and Finger Gaiting](https://www.researchgate.net/publication/3749255_Dextrous_Manipulation_by_Rolling_and_Finger_Gaiting); survey of learning-based in-hand manipulation — [PMC11573780](https://pmc.ncbi.nlm.nih.gov/articles/PMC11573780/)

### Inferences
- The hybrid pattern (planner for the gait schedule / reset states, RL or MPC for execution) is what
  all the strongest results share, under different names: Khandate's RRT-seeded resets, DLR's
  estimator-coupled modular policy, and the 2025 hierarchical-primitives-plus-model-based-regrasp work.
- Given a bounded 20° goal, an M-RRT-style projected-motion integration with a QP force-closure check is
  the cheapest available *diagnostic*: it answers whether 20° is reachable without a gait at all for the
  reader's hand and 60 mm cube, which determines whether the fix is reward shaping or gait learning.

### Gaps
- None of the non-RL papers found reports a fingertip-only 60 mm cube reorientation benchmark with
  success rates, so their reliability in this exact setting is not directly documented.

## Honest bottom line (explicit status of this setting)

### Takeaway
Fingertip-only reorientation of a cube to a **specific, non-symmetry-equivalent target angle** with no
palm support is not a solved, off-the-shelf capability. It has been solved only (a) as unbounded
spinning about an axis, or (b) as goal-reaching over the 24 π/2-symmetric cube orientations with a
learned state estimator in the loop — and in both cases with substantial extra machinery.

### Cited Findings
- Plain model-free RL from a fixed initial state does not learn fingertip-only gaiting at all, even with gravity removed — [ar5iv:2303.03486](https://ar5iv.labs.arxiv.org/html/2303.03486)
- The best-reported fingertip-only, proprioception-only rotation success rate in the 2026 literature on an Allegro-class hand with an 80 mm-diameter object is 83%, and that required both a grasp-quality reward and a custom fingertip geometry — [arXiv:2607.12105](https://arxiv.org/html/2607.12105v1)
- AnyRotate, with dense tactile sensing, averages 1.57 rotations per 30 s episode palm-up and ~0.9 when the thumb is up or down — i.e. failures are common within a single minute of operation — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- Maintained-contact rolling in a precision pinch covers at most ~35.5% of orientation directions on the best of 16 hands tested — so gaiting is not optional for general reorientation — [arXiv:2605.15548](https://arxiv.org/html/2605.15548)

### Inferences
- For the reader's 8°-then-stall symptom, the literature points to three concrete, independently-evidenced
  changes, in order of expected effect: (1) replace the reset distribution with planner-generated
  post-gait states (Khandate's decisive ablation), (2) add λ_min(GGᵀ) dense reward (24→56% effect),
  (3) gate reward on ≥3 fingertip contacts with termination below 2 while explicitly *permitting* the
  transient, so a gait is representable at all.

### Gaps
- No published result reports a bounded small-angle fingertip-only reorientation success rate, so there
  is no benchmark number the reader's ~8° can be compared against.
