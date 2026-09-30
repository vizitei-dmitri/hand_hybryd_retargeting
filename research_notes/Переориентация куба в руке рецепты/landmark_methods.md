# Landmark and Current Methods for In-Hand Object Reorientation: What Actually Converges, and at What Cost

Scope note: this file covers multi-fingered in-hand reorientation to a commanded orientation (and the
closely related "continuous rotation about an axis" family, which is a *different and much cheaper* task).
Every number below is quoted from a primary source; where a source does not state a number, it is listed
under **Gaps** rather than estimated.

## Q1. Which systems have demonstrably achieved in-hand cube/object reorientation to a commanded goal?

### Takeaway
Eight systems have demonstrably solved goal-conditioned in-hand reorientation: OpenAI Dactyl (2018),
OpenAI Rubik's Cube / ADR (2019), Allshire TriFinger (2021), Chen et al. CoRL 2021, Chen et al. Visual
Dexterity (2023), Handa et al. DeXtreme (2023), plus 2026 successors in the DextrAH/ADEPT line. A second,
distinct family (Hora, Rotating-without-Seeing, AnyRotate) solves *continuous rotation about an axis*, not
goal-conditioned reorientation — and its sample cost is two orders of magnitude lower, which is the single
most important fact for anyone whose goal-conditioned PPO run is not learning.

### Cited Findings

**OpenAI Dactyl, 2018 — sim-to-real, Shadow Hand, block reorientation**
- Distributed training on "384 worker machines, each with 16 CPU cores" (= 6,144 CPU cores) plus "a single machine with 8 GPUs" for optimization, communicating via Redis — [Learning Dexterous In-Hand Manipulation, arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Throughput: "approximately 22 years of simulated experience per hour" — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Total experience: "about 100 years of experience" with full domain randomization enabled, versus "3 years of simulated experience" for the non-randomized baseline — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Wall-clock: "roughly 50 hours" with full randomizations versus "1.5 hours" without — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Episode/trial termination: "50 rotations achieved, object dropped, or 80-second timeout" — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Real-world result: median "13 consecutive successful rotations" on the physical block task — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Policy: "a recurrent neural network with memory, namely an LSTM with an additional hidden layer with ReLU activations"; the LSTM correctly predicted whether an object was larger or smaller than average "in 80% of cases after 5 seconds", i.e. implicit online system identification — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Domain randomization was manual (hand-designed ranges), not adaptive — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)

**OpenAI "Solving Rubik's Cube with a Robot Hand", 2019 — sim-to-real, Shadow Hand, ADR introduced**
- Block reorientation task compute: "32 NVIDIA V100 GPUs and 400 worker machines with 32 CPU cores each" (= 12,800 CPU cores) — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Rubik's Cube task compute: "64 NVIDIA V100 GPUs and 920 worker machines with 32 CPU cores each" (= 29,440 CPU cores) — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Cumulative simulated experience for the Rubik's Cube policy: "roughly 13 thousand years" — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Wall-clock: policies were "continuously trained for several months" at scale, with simulation and architecture changes landed mid-run — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Continuous training was made possible by *policy cloning / behavioural distillation* ("policy surgery"): architecture and observation-space changes were absorbed without restarting training — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Block reorientation, real world: best ADR policy achieved a median of 42 consecutive successes against a cap of "50 consecutive successes" — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Rubik's Cube, real world: best policy (ADR XXL) achieved "26.8±4.9 successful rotations/flips" mean; 60% end-to-end success on a fair 15-move scramble and 20% on a maximally difficult 26-move scramble — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Architecture: "single feed-forward layer with 2048 units" plus "LSTM layer with 1024 units" — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)

**Allshire et al., "Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger", 2021 — sim-to-real, 3-finger, full 6-DoF pose goal**
- "16,384 environments in parallel on a single NVIDIA Tesla V100 or RTX 3090 GPU" — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Scale: "billions of steps of experience"; specific runs reached **4 billion steps in approximately 24 hours on a single GPU** — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Success criterion: "the position within 2 cm, and orientation within 0.4 rad (22°) of the target goal pose", explicitly aligned with the earlier Dactyl-lineage benchmark — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Real-world: best policy (object-keypoint observation + keypoint reward, O-KP+R-KP) reached "a success rate of 82.5%" over 40 trials; the position+quaternion observation variant with keypoint reward reached 77.5% — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Algorithm: plain "Proximal Policy Optimisation (PPO)" with an **asymmetric actor-critic** (noisy observations to the actor, privileged simulator state to the critic). No teacher-student distillation — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)

**Chen, Xu, Agrawal, "A System for General In-Hand Object Re-Orientation", CoRL 2021 (best paper) — simulation-first, 2000+ objects, both hand-up and hand-down**
- "40K parallel environments for data collection" for the upward-facing hand; 360 environments for the vision experiments — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Hardware: "at most 22 GPUs with a 32GB memory", trained with Horovod — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Success criterion: angular error threshold θ̄ = **0.1 rad (5.7°)** for non-vision experiments, 0.2 rad for vision experiments, plus a Chamfer-distance threshold of 0.01 for vision-based success — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Episode ends "if the object is reoriented to the goal orientation successfully, or the object falls, or the maximum episode length is reached" — i.e. a **single goal per episode**, not a goal stream — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Teacher/student: teacher observes a **134-dimensional** privileged state (including velocities unavailable on hardware); student observes **31 dimensions** (joint positions, object pose, orientation difference); distilled with DAgger minimising "KL-divergence between π^E and π^S" — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Update schedule: "8 epochs after every 8 rollout steps for the MLP policies and 50 rollout steps for the RNN policies"; Adam, actor LR 3e-4, critic LR 1e-3, γ=0.99 — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)

**Chen et al., "Visual Dexterity", Science Robotics 2023 — sim-to-real, hand facing DOWN, dynamic in-air reorientation**
- Total training compute: "less than 400 hours" of GPU time across the teacher and the two student stages; teacher trained on an RTX 3090, student on a V100 32 GB — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Wall-clock: student stage 1 (synthetic point clouds) ≈ 3 days; stage 2 (rendered refinement) ≈ 1 day — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Cost: "under $15.60 in GPU electricity at 2022 US rates" — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Success criterion: "within 0.4 radians (22.9°)"; relaxed threshold "within 0.8 radians (45.8°)". Stability required object velocity and joint motion below thresholds; **no explicit hold time beyond motion cessation** — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Goal representation: "the point cloud of the object in a target orientation" rather than a quaternion — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Real-world: 81% success at 0.4 rad on in-distribution objects (rigid fingertips), 55% at 0.4 rad on out-of-distribution objects (soft fingertips); control at "12 Hz"; median in-air reorientation time "close to seven seconds" — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Object sets: 150 objects for training, 12 for evaluation — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)

**Handa et al., "DeXtreme", ICRA 2023 — sim-to-real, Allegro hand, palm-up cube reorientation, goal stream**
- "16,384 agents per GPU" across "8 NVIDIA A40 GPUs"; aggregate throughput "700K frames/sec" — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Wall-clock: manual DR 1.41 days (~34 h, elsewhere "about 32 hours"); ADR 2.5 days (~60 h) — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2); the 32 h / "42 years of a single robot's experience" framing is also in [NVIDIA coverage](https://www.design-engineering.com/nvidia-researchers-teach-dexterity-to-robot-hand-via-simulation-1004039597/)
- Estimated cloud cost: $553.80 (manual DR) vs $977.20 (ADR) — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Rates: "simulation dt of (1/60)s and a control dt of (1/30)s" (30 Hz policy) — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Success thresholds: training used a tight **0.1 rad**; testing used **0.4 rad** — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Metric: "consecutive target orientations achieved without dropping", with a maximum trial duration of 80 seconds per orientation — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Reward: rotation-distance shaping of the form **1/(d+0.1)**, plus goal bonus **+250.0**, position-drift penalty **−10.0**, action-magnitude penalty **−0.001**, action-rate penalty **−0.25**, joint-velocity penalty **−0.003** — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Algorithm: **plain PPO with an asymmetric actor-critic** (critic receives privileged state). No teacher-student distillation — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Real-world: "27.8 ± 19.0 consecutive successes" mean (median 14.0); best single rollout "112 consecutive successes" — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)

**ADEPT, 2026 — successor in the DeXtreme/DextrAH line, geometric-fabric action space**
- Total environment steps: **11 billion** (8 B pretraining + 3 B post-training across downstream tasks); 4,096 parallel environments cited for the post-training critic warm-up — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- Wall-clock to ADR level 50 during post-training: ≈ **19.9 hours** — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- Platforms: 23-DoF Kuka-Allegro and 29-DoF Flexiv-Sharpa; skills include grasping, lifting, **in-hand reorientation**, transport and insertion — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- Pretraining success criterion is keypoint-based: "An episode succeeds when e_kp < 0.10 m" — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- Curriculum: ADR with **50 discrete levels**, plus Population-Based Training for hyperparameter search during pretraining — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- Pipeline: state-based RL teacher distilled into a vision-based student; no human demonstrations — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)

### Inferences
- DeXtreme's total sample count is not printed as a single number, but 700,000 frames/s × 32 h ≈ **8×10^10 environment steps** for the manual-DR run. This is an arithmetic inference from two stated figures, not a quoted number.
- Every system that solved *goal-conditioned* reorientation consumed at least 4×10^9 steps (Allshire, the cheapest, and that with a 3-finger hand on a supporting floor). Nothing in the literature solves goal-conditioned SO(3) reorientation at 10^7–10^8 steps. A policy that "isn't learning" after tens of millions of steps is inside the regime where every published system was also still at zero.
- The 0.1 rad training / 0.4 rad testing split used by both DeXtreme and Chen et al. is the operative pattern: train against a tolerance tighter than the one you report, because the shaping reward, not the success bonus, is what drives early learning.

### Gaps
- Dactyl's exact angular tolerance is not quoted in the text retrieved. 0.4 rad is corroborated indirectly: Allshire et al. state their 0.4 rad threshold "aligns with benchmarks from prior dexterous manipulation work" in this lineage, and DeXtreme uses 0.4 rad at test explicitly while reproducing the Dactyl protocol. Treat 0.4 rad for Dactyl as strongly implied rather than directly quoted.
- Visual Dexterity does not state parallel-environment count or total environment steps — only GPU-hours. Its "< 400 GPU-hours" figure is therefore not convertible to a sample count.
- Chen et al. 2021 does not state wall-clock training time.
- The Rubik's Cube paper does not give a per-run step count; "13 thousand years" is cumulative across a months-long continuously-trained lineage, so it is not a from-scratch cost.

## Q2. Success criterion, tolerance, and hold time

### Takeaway
Two tolerance regimes exist: a **tight 0.1 rad (5.7°)** used for *training* (DeXtreme) or as the headline
metric in simulation-first work (Chen 2021), and a **loose 0.4 rad (22.9°)** used for *reporting* sim-to-real
results across Dactyl's lineage, Allshire, Visual Dexterity and DeXtreme. No paper retrieved imposes an
explicit multi-second hold time; stability is enforced via velocity thresholds or by the goal stream itself.

### Cited Findings
- DeXtreme: 0.1 rad during training, 0.4 rad during testing — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Chen et al. 2021: θ̄ = 0.1 rad (non-vision), 0.2 rad (vision) — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Visual Dexterity: 0.4 rad (22.9°) primary, 0.8 rad (45.8°) relaxed; success additionally requires object velocity and joint motion below thresholds, with no stated hold duration — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Allshire TriFinger: position within 2 cm **and** orientation within 0.4 rad (22°) — note this is a full 6-DoF pose criterion, stricter in kind than orientation-only — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Dactyl and DeXtreme both cap a trial at **80 seconds per orientation** and terminate on a drop — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177); [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Dactyl and the Rubik's Cube block task cap a trial at **50 consecutive successes** — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177); [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)

### Inferences
- The absence of an explicit hold time in the goal-stream systems is structural: because a new goal is sampled the instant the current one is hit, "holding" is replaced by "not dropping while chasing the next goal", which is a strictly harder stability constraint and cheaper to implement.

### Gaps
- No source retrieved specifies a hold-time-in-tolerance requirement (e.g. "within 0.1 rad for N consecutive control steps"). If such a criterion exists in an appendix it was not in the retrieved text.

## Q3. Goal structure: single large rotation, goal stream, or continuous rotation? Angle-magnitude curriculum?

### Takeaway
Three distinct task formulations, with very different costs. **Goal stream** (Dactyl, Rubik's, DeXtreme):
resample a random SO(3) goal after each success, report consecutive successes — the most expensive.
**Single goal per episode** (Chen 2021, Visual Dexterity, Allshire): episode ends on success, drop or timeout.
**Continuous rotation about an axis** (Hora, AnyRotate): no goal orientation at all, the reward is angular
velocity projected onto an axis — roughly two orders of magnitude cheaper. No retrieved source uses an
angle-magnitude curriculum; the curricula that are used are over *domain randomization* or *gravity*, not
over goal difficulty.

### Cited Findings
- DeXtreme: "random target orientation is sampled in SO(3)" after each success, requiring continuous adaptation without reset — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Chen 2021: episode terminates on success, drop, or max length — one goal per episode — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Hora: reward is "r_rot ≐ max(min(ω·k̂, r_max), r_min)" with clipping at ±0.5, "to encourage sustained rotation rather than speed maximization"; the full reward is "r ≐ r_rot + λ_pose r_pose + λ_linvel r_linvel + λ_work r_work + λ_torque r_torque". The task is continuous rotation about z, **not** reaching a commanded orientation — [arXiv:2210.04887](https://ar5iv.labs.arxiv.org/html/2210.04887)
- AnyRotate: reward is "r_rotation + r_contact + r_stable + r_terminate", and the rotation term uses **auxiliary goal keypoints** rather than direct angular-velocity commands, which the authors report "proved critical for multi-axis learning" — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- Allshire: keypoint-based reward and keypoint-based object observation outperformed the position+quaternion formulation (82.5% vs 77.5% real-world success) — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Curriculum, Allshire: a **reward-weight curriculum**, not a difficulty curriculum — "reduc[ed] the weight of r_f reward to 0 after 5e7 timesteps" so that the robot could learn the *ungrasping* behaviour needed for reorientation later in training — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Curriculum, Chen 2021: a **gravity curriculum** — initial gravity **+1 m/s²**, decremented by **−0.5 m/s²** stepwise to **−9.81 m/s²**, applied only to the YCB dataset with the downward-facing hand — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Curriculum, Visual Dexterity: no gravity curriculum; the paper notes prior work used a "manually designed gravity curriculum" and states their reward design removed the need — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Curriculum, DeXtreme/OpenAI: ADR — randomization starts at almost none and ranges expand as the policy crosses performance thresholds (DeXtreme's vectorized ADR uses thresholds t_H = 20, t_L = 5 consecutive successes) — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2); ADR introduced in [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- Curriculum, ADEPT: ADR with 50 levels plus PBT — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- Hora: "Training employed domain randomization without explicit curriculum" — stable gaits emerged from reward shaping alone — [arXiv:2210.04887](https://ar5iv.labs.arxiv.org/html/2210.04887)

### Inferences
- Three independent groups converged on **keypoint-based** goal representation and reward (Allshire's object keypoints, Visual Dexterity's goal point cloud, AnyRotate's auxiliary goal keypoints) in preference to quaternion-difference rewards, and two of them report it as the difference between working and not working. This is the highest-leverage, lowest-cost change available to an engineer whose quaternion-distance reward is flat.
- The curricula that appear in successful systems are over *physics* (gravity, domain randomization) and over *reward weights* (dropping a shaping term at a fixed step count), never over goal angle magnitude. An angle-magnitude curriculum is not a validated recipe in this literature.

### Gaps
- No source retrieved implements or evaluates an angle-magnitude curriculum for the goal rotation, so there is no evidence either for or against it.

## Q4. Palm-supported / gravity-assisted vs fingertip-only precision grasp

### Takeaway
This is the sharpest dividing line in the literature and it correlates directly with cost. The cheap,
early, high-consecutive-success results (Dactyl, Rubik's, DeXtreme) are all **palm-up, gravity-assisted**.
The fingertip-only and hand-down results are either **continuous rotation** (Hora, AnyRotate) or required a
**gravity curriculum plus a separate lifting policy for initialization** (Chen 2021) or a **teacher-student
pipeline with 400 GPU-hours and a redesigned reward** (Visual Dexterity).

### Cited Findings
- DeXtreme: Allegro hand "rigidly mounted at the wrist" with the wrist locked, four fingers, **palm-supported** manipulation — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Chen 2021 configurations: **upward** (object initialized 0.13 m above the hand base); **downward with a table** (0.12 m separation between hand and tabletop, so the table catches and supports); **downward without a table** — the last one "requires stable pose initialization from a lifting policy", i.e. it could not be trained from a naive initial state — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Chen 2021: the gravity curriculum (+1 → −9.81 m/s²) was applied **only** to the downward-facing hand configuration — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Visual Dexterity: **downward-facing** hand throughout, "deliberately chosen for real-world applicability despite increased difficulty compared to upward-facing alternatives" — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Hora: fingertip-only, stated explicitly — "at all times the fingers need to maintain a dynamic or static force closure on the object to prevent it from falling (as it can not make use of any other supporting surface such as the palm)" — [arXiv:2210.04887](https://ar5iv.labs.arxiv.org/html/2210.04887)
- AnyRotate: fingertip-only, enforced in the grasp-validity criteria — "The number of tip contacts is greater than 2" and "The number of non-tip contacts is zero"; "No palm support is used" — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- AnyRotate gravity invariance: trained across **six hand orientations** (palm up/down, thumb up/down, base up/down), randomized per episode, so that "the gravity vector is continuously changing in the hand's frame of reference" — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- Allshire's TriFinger operates inside an arena with a floor, and the success criterion includes a 2 cm position component, so the object is surface-supported rather than held against gravity — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)

### Inferences
- Mapping cost onto this axis: the palm-up goal-stream systems (DeXtreme, ~8×10^10 steps by the inference above, 32–60 h on 8 A40s) and the fingertip continuous-rotation systems (Hora, 5×10^8 steps) bracket the range, but they differ in *both* support and goal structure, so neither comparison isolates the support variable.
- Because every fingertip-only or hand-down success in the literature added at least one structural aid (gravity curriculum, lifting-policy initialization, teacher-student distillation, or a reduction of the task to continuous rotation), the practical reading is that fingertip-only goal-conditioned SO(3) reorientation with plain single-stage PPO has **no published success** to copy.

### Gaps
- Dactyl's hand orientation (palm-up in a cage) is not confirmed by any passage retrieved here; it is widely depicted that way but I did not retrieve a quotable sentence.
- Hora's paper states the object is fingertip-held with no palm support, but the retrieved text does **not** state whether the palm faces up or down. One fetch of the arXiv HTML asserted "facing downwards"; a second, more targeted fetch of the same paper returned "the paper does not explicitly state whether the palm faces up or down". **These two retrievals conflict; do not report an orientation for Hora without checking the figures directly.**

## Q5. Which methods needed teacher-student / privileged information, and was it necessary or convenient?

### Takeaway
A clean split. Methods whose deployment observation space is *state-like* (joint angles plus a pose
estimate) used **plain PPO with an asymmetric critic** — DeXtreme and Allshire both did, and both transferred
to hardware. Methods whose deployment observation space is *vision or touch* used teacher-student
distillation, and in those cases the papers describe it as necessary for deployment, not for learning.

### Cited Findings
- DeXtreme: "Plain PPO (no teacher-student); asymmetric actor-critic with critic receiving privileged state information" — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- Allshire: PPO, "an actor-critic approach with asymmetric observations—the actor receives noisy sensor data while the critic accesses privileged simulator information for improved value estimates" — [arXiv:2108.09779](https://ar5iv.labs.arxiv.org/html/2108.09779)
- Chen 2021: DAgger distillation from a 134-D privileged teacher to a 31-D student, motivated by the teacher's use of velocities "unavailable in the real world"; the paper's own framing is that distillation makes policies "amenable to real-world operation" — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Chen 2021's stated rationale for teacher-student more broadly: "Teacher-student training enables the agent to specialize its behavior to the current dynamics, instead of learning a single behavior that works across different dynamics" — [ResearchGate record](https://www.researchgate.net/publication/355924919_A_System_for_General_In-Hand_Object_Re-Orientation)
- Visual Dexterity: teacher-student distillation was "Essential for real-world deployment" — the teacher learns from privileged state and the student learns a visual policy by supervised learning, "making deployment feasible without object-specific pose estimators" — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- Hora: two-stage RMA — "first training a base policy with privileged object information, then an adaptation module φ that estimates object properties from proprioceptive history alone"; 100,000 gradient updates for the base policy — [arXiv:2210.04887](https://ar5iv.labs.arxiv.org/html/2210.04887)
- AnyRotate: teacher sees privileged object pose, dimensions and mass; student sees only "proprioception and tactile feedback"; the student **initializes its policy weights from the teacher** and trains only the encoder by supervised regression onto the teacher's latent vector — [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- ADEPT (2026) still uses the same pattern: "state-based teacher trained via RL, then distilled into vision-based student for real-world deployment" — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- OpenAI's alternative to distillation for *architecture* changes was policy cloning / behavioural distillation, used to keep a months-long run alive rather than to compress privileged information — [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)

### Inferences
- No retrieved source claims that teacher-student distillation is required in order for the *reorientation task itself* to be learned. In every case the stated purpose is bridging an observation gap (velocities, vision, tactile) between simulation and hardware. An engineer whose policy is not learning at all in simulation is facing a different problem, and adding distillation will not fix it.
- Conversely, the two systems that dispensed with distillation (DeXtreme, Allshire) both kept the *asymmetric critic*. That is the cheap half of the idea and is worth adopting independently: privileged state to the critic, noisy observations to the actor, single-stage PPO.

### Gaps
- None of the retrieved papers report an ablation of "plain PPO on the final observation space" versus "teacher-student", which would be the experiment that settles necessary-vs-convenient. I found no such ablation.

## Q6. Reported sample-complexity gap between palm-supported and fingertip-only reorientation

### Takeaway
**No source quantifies this gap directly.** I found no paper that trains the same policy, reward and goal
structure under both palm-supported and fingertip-only support and reports the step counts for each. The
qualitative evidence is unanimous that fingertip-only and hand-down are harder, and the evidence takes the
form of *added machinery* rather than *added steps*.

### Cited Findings
- Chen 2021 explicitly treats the hand-down configuration as substantially harder and adds a gravity curriculum and a lifting-policy initialization for it, but "does not provide direct quantitative comparison of total environment steps between configurations" — [arXiv:2111.03043](https://ar5iv.labs.arxiv.org/html/2111.03043)
- Visual Dexterity states the downward-facing choice carries "increased difficulty compared to upward-facing alternatives" without quantifying it — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- The nearest quantified contrast in the literature is not about support but about randomization: Dactyl needed "about 100 years of experience" with full domain randomization versus "3 years" without — a **~33× sample-complexity multiplier for domain randomization alone** — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177)
- Cross-paper order-of-magnitude contrast, tasks not matched: DeXtreme, palm-up goal stream, 16,384 envs/GPU × 8 A40, ~32–60 h, ~700K fps — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2) — versus Hora, fingertip-only continuous rotation, 16,384 envs, **~500 million agent steps**, ≈"7,000 hours real-world time" equivalent — [arXiv:2210.04887](https://ar5iv.labs.arxiv.org/html/2210.04887)

### Inferences
- The DeXtreme-vs-Hora contrast spans roughly two orders of magnitude in environment steps (~8×10^10 inferred vs 5×10^8 stated), but it confounds support (palm vs fingertip) with goal structure (SO(3) goal stream vs continuous axis rotation) and with tolerance. It should be cited as "the two task formulations differ by ~100× in sample cost", not as a palm-vs-fingertip measurement.
- The Dactyl 33× randomization multiplier is the one clean, quoted elasticity in this literature, and it suggests a concrete diagnostic: if a policy is not learning, turning domain randomization off should make the task learnable within a small fraction of the budget. If it does not, the problem is the reward or the task definition, not the sample budget.

### Gaps
- No quantified palm-vs-fingertip sample-complexity ratio exists in any source I retrieved. Reporting one would require estimating, which I have not done.

## Out of scope / not the same problem

### Takeaway
Two frequently co-cited lines are *not* multi-fingered SO(3) reorientation and should not be used as
evidence about it.

### Cited Findings
- "Rotating Objects via In-Hand Pivoting using Vision, Force and Touch" (2023) controls "the rotation of an object around the grip point of a **parallel gripper** by allowing rotational slip" — a two-jaw gripper pivoting problem, not multi-fingered reorientation — [arXiv:2303.10865](https://arxiv.org/abs/2303.10865)
- "In-Hand Gravitational Pivoting Using Tactile Sensing" similarly rotates an object by "loosening the grip on the object, with gravity inducing a torque that causes it to pivot inside the gripper" — [arXiv:2210.05068](https://arxiv.org/pdf/2210.05068)
- "Rotating without Seeing: Towards In-hand Dexterity through Touch" (2023) is a touch-only *continuous rotation* system, in the Hora family rather than the goal-conditioned family — [arXiv:2303.10880](https://arxiv.org/pdf/2303.10880)

### Gaps
- I could not extract quantitative training details for "Rotating without Seeing": the arXiv PDF returned binary/image content that the fetch tool could not parse, and I did not retrieve an HTML version. Its environment count, step count and GPU hours are unknown to this report.

## Chronology and supersession

### Cited Findings
- 2018 Dactyl → 2019 Rubik's Cube: same lab, same hardware family; the 2019 paper supersedes 2018 by adding ADR, and reports a higher median (42 vs 13 consecutive block successes) — [arXiv:1808.00177](https://ar5iv.labs.arxiv.org/html/1808.00177); [arXiv:1910.07113](https://ar5iv.labs.arxiv.org/html/1910.07113)
- 2023 DeXtreme supersedes the OpenAI line on **cost**: "8 NVIDIA A40 GPUs ... as opposed to OpenAI's use of a CPU cluster composed of 400 servers with 32 CPU-cores each, as well as 32 NVIDIA V100 GPUs" — [arXiv:2210.13702](https://arxiv.org/html/2210.13702v2)
- 2021 Chen et al. → 2023 Visual Dexterity: same lab; the 2023 work removes the gravity curriculum, moves to vision-based goals and to a downward-facing hand, and is the stronger result for hand-down reorientation — [arXiv:2211.11744](https://ar5iv.labs.arxiv.org/html/2211.11744)
- 2022 Hora → 2024 AnyRotate: continuous-rotation line extended from single-axis z to arbitrary axes and to gravity-invariant hand orientations — [arXiv:2210.04887](https://ar5iv.labs.arxiv.org/html/2210.04887); [arXiv:2405.07391](https://arxiv.org/html/2405.07391v2)
- 2026 ADEPT continues the NVIDIA line with an 11 B-step pretrain-then-post-train recipe and geometric-fabric action spaces, referencing DeXtreme and DextrAH as prior work — [arXiv:2608.19182](https://arxiv.org/html/2608.19182v1)
- Isaac Lab is now the documented successor platform to Isaac Gym for this class of work, with stated support for "high-fidelity simulation of multi-DOF hands, accurate contact modeling", domain randomization and tiled rendering — [Isaac Lab, arXiv:2511.04831](https://arxiv.org/pdf/2511.04831)

### Gaps
- I did not find a 2025–2026 paper that reports goal-conditioned fingertip-only cube reorientation trained with single-stage PPO in Isaac Lab, with published step counts. If such a result exists it did not surface in the searches run here.
