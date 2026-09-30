# Compute budget and the exploration problem for in-hand dexterous manipulation on a single 6 GB GPU

Scope note: this file covers (a) honest feasibility on one consumer 6 GB GPU, (b) throughput and
env-count evidence, (c) exploration / entropy / std-collapse evidence with actual config values,
(d) sample-cost-reduction techniques with the evidence for each. Numbers marked **[arithmetic]**
are my own computation from cited figures, not quoted from a source.

---

## Q1. Throughput and environment counts actually achieved for dexterous-hand tasks on consumer GPUs

### Takeaway
NVIDIA's own benchmark for the reference cube-reorientation task reports **6.4 GB VRAM at 8192
environments** — i.e. the canonical configuration does not fit in 6 GB at all, and the smallest GPU
NVIDIA benchmarks is an RTX 4090 (170k FPS full train loop). I found **no documented report of
anyone training a hand reorientation policy to success on a 6–8 GB consumer GPU**, successfully or
not; the only consumer-GPU Isaac Lab write-ups I could find are locomotion toy tasks.

### Cited Findings
- Isaac Lab official performance benchmarks, task `Isaac-Repose-Cube-Shadow-Direct-v0` (Shadow Hand
  cube reorientation), at **8192 envs: 6.7 GB RAM, 6.4 GB VRAM** — [Isaac Lab performance benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- Same task, **single RTX 4090, 8192 envs: 200,000 env-step FPS; 190,000 step+inference;
  170,000 step+inference+training** — [Isaac Lab performance benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- Same task, **single L40, 8192 envs: 170,000 / 140,000 / 120,000 FPS**; 4x L40 → 390,000 FPS full
  loop; 16x L40 (4 nodes) → 1,800,000 FPS full loop — [Isaac Lab performance benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- The RTX 4090 is **the smallest single GPU listed** in that benchmark table; no 60-class or
  8 GB-class card appears — [Isaac Lab performance benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- Isaac Gym paper: Shadow Hand, **NVIDIA A100**, maximum effective frame-rate **150K parallel
  environment steps per second at 16384 agents** — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)
- Search-surfaced claim that dexterous hand tasks reach **40,000+ mean FPS with 2,048 envs on one
  RTX 3090** (24 GB) — surfaced via search summary of NVIDIA/Isaac material; I could not open the
  primary table, treat as indicative only — [search result set](https://developer.nvidia.com/blog/r2d2-scaling-multimodal-robot-learning-with-nvidia-isaac-lab/)
- The only consumer-GPU Isaac Lab report I found is an **RTX 4060, 8 GB**, running `Isaac-Ant-v0`
  and `Isaac-Velocity-Rough-Anymal-C-v0`; the author began at `--num_envs=20`, ran 1,000 ants, and
  reported 10,000 ants as viewport-breaking. **No FPS numbers, no hand task, and explicit VRAM-exhaustion
  crashes** when pushing env counts, plus driver-version GPU crashes — [tech-multiverse: Yes, You Can Train Robots in Isaac Lab on an RTX 4060](https://tech-multiverse.com/projects/yes-you-can-train-robots-in-isaac-lab-on-an-rtx-4060/)
- DeXtreme (the sim-to-real cube reorientation result): **8x NVIDIA A40**, 8,192–16,384 envs per GPU,
  **~700K frames/second aggregate**, **2.5 days** wall-clock for the best ADR policies and
  **1.41 days** for manual domain randomization, equivalent to **~42 years of real-world
  interaction** — [DeXtreme (ar5iv 2210.13702)](https://ar5iv.labs.arxiv.org/html/2210.13702)

### Inferences
- **[arithmetic]** The reader's stated operating point — 1024 envs, ~4 s per PPO iteration — implies
  roughly **4,100 env steps/s** if using the Isaac Lab Shadow Hand default `num_steps_per_env: 16`
  (1024 x 16 = 16,384 steps per iteration / 4 s). That is **~40x below a single RTX 4090** on the
  same task and **~37x below the A100 Isaac Gym peak**.
- **[arithmetic]** DeXtreme's manual-DR run consumed ~700,000 x 86,400 x 1.41 ≈ **8.5e10 env steps**.
  At 4,100 steps/s that is **~2.1e7 seconds ≈ 240 days**. The full DeXtreme sim-to-real recipe is
  therefore roughly **eight months of continuous compute** on this hardware. This is the honest gap:
  not a tuning gap, a two-orders-of-magnitude compute gap.
- The 6.4 GB VRAM figure at 8192 envs, on a card with 6 GB total (minus display/driver overhead),
  means the reference config is unrunnable. If VRAM scaled linearly, 1024 envs would be ~0.8 GB of
  sim state, which is consistent with 1024 envs being a *forced* choice rather than a tuning choice —
  but linearity is an assumption, since a fixed base cost (PhysX scene, USD stage, kernels) dominates
  at low env counts.

### Gaps
- No primary source gives FPS for a hand-reorientation task on an RTX 3060/4060-class card. NVIDIA's
  table stops at the 4090; community reports stop at locomotion.
- No published report — success or failure — of a hand reorientation policy trained on 6–8 GB.
- No per-environment VRAM breakdown that would let one predict the max safe `num_envs` on 6 GB.

---

## Q2. Relationship between number of parallel environments and PPO sample efficiency / final performance

### Takeaway
The documented effect of env count on Shadow Hand is **wall-clock, not asymptote**. The Isaac Gym
scaling study shows an order-of-magnitude wall-clock reduction going from 256 to 16384 envs, and
notably does **not** report a materially different final reward for small env counts. So **1024 envs
is not documented as "too few to learn"** — it is documented as "roughly 10x slower to get there."
I found no study establishing a minimum viable env count for this task.

### Cited Findings
- Isaac Gym Shadow Hand scaling: "As the number of agents is increased, in this case, from 256 to
  16384, the **training time is reduced by an order of magnitude from 5x10^4 seconds (~14 hours) to
  3x10^3 seconds (~1 hour)**" — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)
- Best training time achieved at **both 8192 and 16384 envs**, with **horizon lengths 16 and 8**
  respectively — i.e. the batch is held roughly constant by shortening the rollout as envs grow — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)
- Shadow Hand "reaches performant dexterity of **10 consecutive successes at reward of 3000 in just
  5 minutes**" (A100, large env count) — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)
- The Isaac Gym paper's scaling figure plots "rewards and effective FPS with respect to number of
  parallel environments"; the fetched analysis found **no significant difference in final achieved
  reward across env counts**, with the emphasis entirely on wall-clock — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)
- IsaacGymEnvs ships `ShadowHand.yaml` with **numEnvs: 16384**, `episodeLength: 600` — [IsaacGymEnvs cfg/task/ShadowHand.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/task/ShadowHand.yaml)
- Corresponding `ShadowHandPPO.yaml` uses **minibatch_size: 32768, horizon_length: 8, mini_epochs: 5**
  — i.e. a full batch of 16384 x 8 = 131,072 steps split into 4 minibatches — [IsaacGymEnvs cfg/train/ShadowHandPPO.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/ShadowHandPPO.yaml)
- Isaac Lab `ShadowHandPPORunnerCfg` (rsl_rl): **num_steps_per_env: 16, max_iterations: 10000,
  num_mini_batches: 4** — [IsaacLab shadow_hand rsl_rl_ppo_cfg.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py)

### Inferences
- **[arithmetic]** At 1024 envs x 16 steps the reader's batch is **16,384 steps/iteration**, i.e.
  **1/8 of the IsaacGymEnvs batch (131,072)** and 1/8 of Isaac Lab's Shadow Hand batch at 8192 envs.
  With `num_mini_batches: 4` each minibatch is 4,096 samples. For a 20-DoF action space under domain
  randomization this is a small, high-variance gradient estimate — the most likely mechanism by which
  low env count hurts *quality* rather than only speed: noisier advantage estimates plus a
  KL-adaptive LR that reacts to that noise.
- The cheapest structural fix that preserves batch size on fixed VRAM is to **trade envs for horizon**:
  raise `num_steps_per_env` from 16 to 64–128 at 1024 envs to recover a 65k–131k batch. This mirrors
  the Isaac Gym finding in reverse (they shortened horizon as envs grew). Cost: more wall-clock per
  iteration, more off-policyness within the 5 epochs, and longer credit-assignment windows.
- **[arithmetic]** The reader's 10,000-iteration default at 4 s/iter is **~11 hours**, but delivers
  only 1/8 the samples of the reference run. Sample-matching the Isaac Lab default requires
  ~80,000 iterations ≈ **3.7 days**. That is the realistic bound for an in-sim-only result — not
  months — *provided* exploration works.

### Gaps
- No ablation isolating final performance at 256/512/1024 envs with a *fixed sample budget* (as
  opposed to fixed wall-clock). The Isaac Gym figure conflates the two.
- No source states a minimum viable env count for hand reorientation. The claim "1024 is too few" is
  **not supported by any source I found** — the supported claim is "1024 is ~8–16x slower."

---

## Q3. Entropy coefficient, initial and learned action-noise std, and std collapse — actual config values

### Takeaway
This is the highest-value finding in the file. **Every reference hand config I could open starts the
action std at exactly 1.0 (log_std = 0) and uses a state-independent (single global) sigma
parameter, never a state-dependent network head.** Entropy coefficients are small: **0.0 (IsaacGymEnvs
ShadowHand), 0.002 (DeXtreme / AllegroHandDextremeADR), 0.005 (Isaac Lab rsl_rl Shadow Hand)**. A
std that *shrinks while performance degrades* is the opposite of these configs' behaviour and points
at std parameterization + KL-adaptive LR, not at the entropy coefficient alone.

### Cited Findings — IsaacGymEnvs / rl_games, ShadowHand
- `ShadowHandPPO.yaml`: **`entropy_coef: 0.0`**, **`sigma_init: const_initializer, val: 0`**
  (log-std 0 → std 1.0), **`fixed_sigma: True`**, `learning_rate: 5e-4`, `kl_threshold: 0.016`,
  `horizon_length: 8`, `minibatch_size: 32768`, `mini_epochs: 5`, `bounds_loss_coef: 0.0001`,
  network ELU MLP [512, 512, 256, 128], **no `central_value_config`** (symmetric critic) — [IsaacGymEnvs cfg/train/ShadowHandPPO.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/ShadowHandPPO.yaml)

### Cited Findings — DeXtreme / AllegroHandDextremeADR (the sim-to-real config)
- `AllegroHandDextremeADRPPO.yaml`: **`entropy_coef: 0.002`**, **`sigma_init: const_initializer,
  val: 0`**, **`fixed_sigma: True`**, `learning_rate: 1e-4` with **`lr_schedule: linear`**,
  `kl_threshold: 0.01`, `horizon_length: 16`, `minibatch_size: 16384`, `mini_epochs: 4`,
  **`bounds_loss_coef: 0.005`** (50x the plain ShadowHand value) — [IsaacGymEnvs cfg/train/AllegroHandDextremeADRPPO.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/AllegroHandDextremeADRPPO.yaml)
- Same file, `central_value_config` (asymmetric critic): separate value network fed
  `dof_pos, dof_vel, dof_force, object_pose, object_pose_cam_randomized, object_vels, goal_pose,
  goal_relative_rot, last_actions` **plus randomization and sensor parameters**, with
  `learning_rate: 5e-5`, `minibatch_size: 16384`, `mini_epochs: 4`, `kl_threshold: 0.016`, and an
  **LSTM with 2048 units** — [IsaacGymEnvs cfg/train/AllegroHandDextremeADRPPO.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/AllegroHandDextremeADRPPO.yaml)
- DeXtreme paper: **entropy regularization coefficient 0.002**, discount **γ = 0.998** (not 0.99),
  clipping ε = 0.2, **policy lr 1e-4, value lr 5e-5**, GAE λ = 0.95, minibatch 16384 — [DeXtreme (ar5iv 2210.13702)](https://ar5iv.labs.arxiv.org/html/2210.13702)

### Cited Findings — Isaac Lab / rsl_rl, Shadow Hand
- `ShadowHandPPORunnerCfg`: **`init_noise_std: 1.0`**, **`entropy_coef: 0.005`**, `desired_kl: 0.016`,
  `learning_rate: 5.0e-4`, `num_steps_per_env: 16`, `max_iterations: 10000`,
  `num_learning_epochs: 5`, `num_mini_batches: 4`, `clip_param: 0.2`, `gamma: 0.99`, `lam: 0.95`,
  actor and critic both [512, 512, 256, 128] — [IsaacLab shadow_hand rsl_rl_ppo_cfg.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py)
- `ShadowHandAsymFFPPORunnerCfg` (asymmetric variant, shipped in-tree): **`init_noise_std: 1.0`**,
  **`entropy_coef: 0.005`**, `desired_kl: 0.01`, **actor [400, 400, 200, 100] vs critic
  [512, 512, 256, 128]** — [IsaacLab shadow_hand rsl_rl_ppo_cfg.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py)
- `ShadowHandVisionFFPPORunnerCfg`: **`num_steps_per_env: 64`, `max_iterations: 50000`**,
  `init_noise_std: 1.0`, `entropy_coef: 0.005`, actor/critic [1024, 512, 512, 256, 128] — [IsaacLab shadow_hand rsl_rl_ppo_cfg.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py)
- Isaac Lab documentation describes the Shadow Hand MLP setup as "Gaussian action distribution
  **initialized with unit standard deviation**", empirical observation normalization, entropy
  coefficient 0.005, desired KL 0.016, gradient clipping 1.0 — [Isaac Lab configuring an RL agent](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/configuring_rl_training.html)
- rsl_rl ≥ 5.0.0 **deprecates `init_noise_std`** in favour of `distribution_cfg` with an `init_std`
  field — relevant if the reader is on a recent rsl_rl and the old field is being silently ignored — [isaaclab_rl API docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)

### Cited Findings — reference values elsewhere
- Recent dexterous-manipulation recipe (Regrind): **initial action noise std 0.5** (scalar, diagonal
  Gaussian), **entropy coefficient 0.002**, and explicitly "when using residual RL with retargeted
  motion as the base action, a **smaller** initial action noise standard deviation is used" — [Regrind (ar5iv 2607.11874)](https://ar5iv.labs.arxiv.org/html/2607.11874)
- Entropy coefficients across dexterous-manipulation PPO work range **0 to 0.01**, tuned per task — [search synthesis over arXiv dexterous-manipulation PPO configs](https://arxiv.org/pdf/2607.11874)

### Inferences
- **The single most actionable diagnosis**: the reference configs make std collapse *structurally
  harder* in two ways at once — (1) sigma is **state-independent** (`fixed_sigma: True` in rl_games,
  a single `nn.Parameter` in rsl_rl), so it cannot be driven down by per-state overfitting; and
  (2) it **starts at 1.0**, which for a normalized action space in [-1, 1] means near-uniform
  saturating exploration for the first thousands of iterations. If the reader's policy uses a
  state-dependent sigma head, or an `init_noise_std` well below 1.0, they are outside the regime every
  working config occupies.
- **Caveat I want to flag honestly**: in rl_games, `fixed_sigma: True` selects a *state-independent*
  sigma; whether that parameter remains gradient-trainable depends on the rl_games version and
  `learn_sigma`-style flags, which the YAML fetch reported as "learn_sigma disabled" for ShadowHand.
  I could not verify the runtime semantics from source. The safe reading — state-independent sigma,
  initialized to 1.0 — is well supported; "sigma is frozen entirely" is not something I verified.
- **KL-adaptive LR interaction (mechanism, not a cited result)**: both stacks use adaptive LR against
  a KL target (`kl_threshold: 0.016` / `desired_kl: 0.016`). KL between two Gaussians scales as
  Δμ²/σ². As σ shrinks, an unchanged mean shift produces a *larger* measured KL, so the adaptive
  controller cuts the LR, which freezes the policy at exactly the moment it has stopped exploring.
  This is a self-reinforcing loop and it matches the reader's symptom — shrinking σ with degrading
  performance. Worth logging σ and LR on the same axis to confirm.
- `bounds_loss_coef` is the under-appreciated knob: DeXtreme raises it to **0.005** vs ShadowHand's
  **0.0001**. It penalizes pre-tanh action means leaving [-1, 1]. Without it, the mean saturates at
  the action bound, gradients vanish, and the only way the policy can reduce the action penalty is to
  shrink σ — a plausible route to the reader's collapse.

### Gaps
- I could not open `hora`'s config (the guessed raw path 404s), so I have no primary entropy/sigma
  values for the in-hand-rotation RMA recipe.
- No source documents a *numerical* relationship between entropy coefficient and final hand-task
  success rate (no entropy-coef sweep for hand reorientation was found).
- No source states an explicit sigma floor value for hand tasks.

---

## Q4. Is premature std collapse a recognized failure mode, and what are the remedies?

### Takeaway
Entropy collapse in PPO is a recognized and studied failure mode, and there is a specific published
result that PPO's repeated epochs **amplify** it while the clipping mechanism, correctly orchestrated,
can **protect against** it. However, evidence for the specific remedy list (std floors, resetting std)
is thin in the literature; the strongest *evidence-backed* remedy is the one the reference configs
already encode — state-independent sigma initialized at 1.0 plus a small entropy bonus.

### Cited Findings
- "The **repeated policy updates in PPO amplify entropy collapse** empirically. However, the clipping
  in PPO, when appropriately orchestrated, **can protect against entropy collapse** as well." — [No Representation, No Trust: Connecting Representation, Collapse, and Trust Issues in PPO](https://arxiv.org/html/2405.00662v1)
- Distinct, related failure mode — **rank collapse**: "Rank collapse of the policy network gives a
  policy with **high entropy but zero variance across states**, where the network outputs the same
  high-entropy action distribution in all states, as all the neurons in the feature layer are dead."
  This means an apparently healthy entropy number can coexist with a dead policy — entropy alone is
  not a sufficient diagnostic — [No Representation, No Trust (arXiv 2405.00662)](https://arxiv.org/html/2405.00662v1)
- Entropy regularization's stated purpose is to "prevent **premature convergence** of one action
  probability dominating the policy and preventing exploration"; the entropy coefficient is
  multiplied by maximum possible entropy and added to the loss — [PPO hyperparameters and ranges](https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe) (secondary source, practitioner guide)
- In PPO implementations the **gradient of the objective and of the entropy loss with respect to the
  log-std vector is computed explicitly**, confirming log-std is a first-class trainable parameter
  separate from the mean head — [PPO in PyTorch](https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-rl-in-pytorch-75067bb571b5) (secondary)
- Entropy regularization in policy optimization has a dedicated literature treating it as the
  principal mechanism against premature determinism — [Policy Optimization Reinforcement Learning with Entropy Regularization (arXiv 1912.01557)](https://arxiv.org/pdf/1912.01557)
- Recent work frames entropy preservation as an explicit algorithmic objective rather than a
  hyperparameter — [Entropy-Preserving Reinforcement Learning (arXiv 2603.11682)](https://arxiv.org/pdf/2603.11682)
- Practitioner consensus that hyperparameters "must be carefully hand-tuned to **avoid policy
  collapse**" — [search synthesis, PPO fine-tuning literature](https://openreview.net/pdf?id=rxEmiOEIFL)

### Inferences (ranked remedy list, with the strength of evidence for each)
1. **Use state-independent sigma, initialized to 1.0** — *strongest evidence*: this is what all three
   reference hand configs do ([ShadowHandPPO.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/ShadowHandPPO.yaml), [AllegroHandDextremeADRPPO.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/AllegroHandDextremeADRPPO.yaml), [IsaacLab rsl_rl cfg](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py)). Zero cost to try.
2. **Set entropy_coef to 0.005** (Isaac Lab's value for this exact task) rather than 0 or a
   hand-picked small number — direct config evidence.
3. **Raise `bounds_loss_coef` toward 0.005** (DeXtreme's value) to stop mean saturation at the action
   bound — direct config evidence, mechanism inferred.
4. **Decouple the LR schedule from measured KL** — use DeXtreme's `lr_schedule: linear` with
   `learning_rate: 1e-4` instead of KL-adaptive, if σ and LR are observed collapsing together. Config
   evidence that a linear schedule is what the sim-to-real result used; the collapse-loop mechanism is
   my inference, not cited.
5. **Clamp log-std to a floor** — plausible and widely implemented, but I found **no primary source
   quantifying its effect on a dexterous hand task**. Treat as unvalidated engineering.
6. **Resetting std mid-training** — I found **no source** documenting this for manipulation. Do not
   present it as evidence-backed.
7. **Monitor variance-across-states, not just entropy** — motivated by the rank-collapse result:
   log the std of the action *mean* across the batch alongside σ — [No Representation, No Trust](https://arxiv.org/html/2405.00662v1)

### Gaps
- No paper I found isolates std collapse specifically in a dexterous-manipulation PPO run with
  before/after curves. The entropy-collapse literature I could reach is either general policy-gradient
  theory or LLM-RL (long-CoT) rather than robot control.
- No quantified comparison of state-dependent vs state-independent sigma for hand tasks. The evidence
  is "every working config chose state-independent," which is suggestive, not an ablation.

---

## Q5. Techniques documented to reduce sample cost for dexterous manipulation

### Takeaway
Asymmetric actor-critic with a privileged critic is **used by every serious hand result** and is
essentially free on a small budget (the critic can be large even when the actor cannot).
Reference-motion / residual-policy guidance has the strongest *quantified* evidence for turning a
0–22% task into a 99% task. Teacher-student distillation and ADR/PBT are documented as *necessary for
sim-to-real*, not as sample-cost reducers, and PBT in particular is the single worst fit for a 6 GB
budget.

### Cited Findings — asymmetric actor-critic / privileged critic
- DeXtreme: asymmetric actor-critic where the **critic receives 265-dimensional privileged
  observations** (fingertip positions, velocities, forces, and the domain-randomization parameters
  themselves) while the **actor receives only 50 dimensions** — [DeXtreme (ar5iv 2210.13702)](https://ar5iv.labs.arxiv.org/html/2210.13702)
- The `central_value_config` in `AllegroHandDextremeADRPPO.yaml` implements this with a separate
  value net at `learning_rate: 5e-5` and a **2048-unit LSTM**, fed `dof_force`, `object_vels`,
  `object_pose_cam_randomized` and randomization/sensor parameters that the actor never sees — [IsaacGymEnvs AllegroHandDextremeADRPPO.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/AllegroHandDextremeADRPPO.yaml)
- Isaac Lab ships an in-tree asymmetric config, `ShadowHandAsymFFPPORunnerCfg`, with a **smaller actor
  (400,400,200,100) than critic (512,512,256,128)** — direct evidence that shrinking the actor while
  keeping the critic large is the sanctioned trade — [IsaacLab shadow_hand rsl_rl_ppo_cfg.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py)

### Cited Findings — reference motion, residual policy, retargeting
- Regrind: the policy learns "a **residual action on top of the reference motion**"; the control
  target is retargeted trajectory + learned residual, and the residual head is **initialized to
  produce zero-mean residual actions** — [Regrind (ar5iv 2607.11874)](https://ar5iv.labs.arxiv.org/html/2607.11874)
- Regrind reward: a **keypoint-based distance metric** against the retargeted reference, as a dense
  multi-term reward, rather than explicit contact priors — [Regrind (ar5iv 2607.11874)](https://ar5iv.labs.arxiv.org/html/2607.11874)
- Quantified necessity: **Regrind ~99% sim success vs 0–22% for naive IK-based retargeting on the
  same tasks**; the authors state that without interaction-aware retargeting the references are
  "physically implausible," **undermining exploration quality rather than guiding it** — [Regrind (ar5iv 2607.11874)](https://ar5iv.labs.arxiv.org/html/2607.11874)
- Regrind's budget for context: **4,096 envs, 24 steps, 98,304 transitions per iteration**, initial
  action noise std 0.5, entropy 0.002 — [Regrind (ar5iv 2607.11874)](https://ar5iv.labs.arxiv.org/html/2607.11874)

### Cited Findings — teacher-student distillation
- hora (In-Hand Object Rotation via Rapid Motor Adaptation): the controller "encodes the object's
  intrinsic properties (such as mass and size) to an **extrinsics vector**", and an **adaptation
  module estimates the extrinsics vector from the discrepancy between observed proprioception history
  and commanded actions** — the teacher-student structure — [Qi et al., CoRL 2022 (PMLR v205)](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)
- hora trains "entirely in simulation on **only cylindrical objects**", then deploys **without
  fine-tuning** to rotate dozens of objects of diverse size/shape/weight about z — evidence that
  **narrowing the object and goal distribution is a legitimate scope reduction, not a compromise** — [Qi et al., CoRL 2022](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)
- "Natural and stable **finger gaits automatically emerge**" from RL training on this reduced task — [Qi et al., CoRL 2022](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)
- Isaac Lab has first-class **RSL-RL distillation** configs merged in-tree, so student training is
  supported without custom code — [IsaacLab PR #2182: Add configs and adapt exporter for RSL-RL distillation](https://github.com/isaac-sim/IsaacLab/pull/2182), [isaaclab_rl distillation_cfg docs](https://docs.robotsfan.com/isaaclab_official/main/_modules/isaaclab_rl/rsl_rl/distillation_cfg.html)

### Cited Findings — curriculum, ADR, PBT
- DeXtreme's **Vectorized ADR (VADR)** automatically adjusts randomization ranges during training with
  **40% of vectorized environments dedicated to evaluation** — i.e. ADR *costs* 40% of throughput — [DeXtreme (ar5iv 2210.13702)](https://ar5iv.labs.arxiv.org/html/2210.13702)
- ADR vs manual DR outcome: **27.8 vs 14.8 mean consecutive successes** on the real robot; ADR also
  cost **2.5 days vs 1.41 days** — the gain is real-world robustness, not sim sample efficiency — [DeXtreme (ar5iv 2210.13702)](https://ar5iv.labs.arxiv.org/html/2210.13702)
- Isaac Lab 2.3 ships **ADR and Population Based Training** as supported techniques "to enable better
  scaling for RL training" for dexterous manipulation — [NVIDIA: Streamline Robot Learning ... Isaac Lab 2.3](https://developer.nvidia.com/blog/streamline-robot-learning-with-whole-body-control-and-enhanced-teleoperation-in-nvidia-isaac-lab-2-3)
- Curriculum evidence for manipulation against gravity: a dedicated study finds **curriculum is more
  influential than haptic information** during RL of object manipulation against gravity — [Curriculum Is More Influential Than Haptic Information (arXiv 2407.09986)](https://arxiv.org/pdf/2407.09986)

### Inferences — ranked for a 6 GB single-GPU budget
1. **Asymmetric actor-critic (privileged critic)** — highest value per unit of effort. VRAM cost is in
   the critic's forward pass on already-simulated state, not in more environments; DeXtreme's critic
   sees the DR parameters themselves, which directly fixes the "critic cannot explain the variance"
   problem that a small batch aggravates. Supported by three independent configs.
2. **Reference-motion / residual policy** — the only technique with a quantified order-of-magnitude
   task-success delta (0–22% → 99%). It converts an exploration problem into a tracking problem, which
   is exactly what a small batch can solve. Note the paired hyperparameter change: **reduce init noise
   std when acting residually** (Regrind uses 0.5 and says smaller still for residual).
3. **Curriculum over physics / goal difficulty** — cited as more influential than sensing for
   gravity-loaded manipulation; cheap to implement.
4. **Teacher-student distillation** — use it the hora way: reduce the *object and goal distribution*
   (one object class, one rotation axis) rather than trying to distil a policy you cannot afford to
   train first. Distillation itself does not reduce the cost of getting the teacher.
5. **ADR** — actively harmful on this budget: 40% of envs go to evaluation, and the payoff is
   real-robot robustness the reader may not be chasing yet.
6. **PBT** — worst fit. It multiplies runs; a single 6 GB card has no population headroom.

### Gaps
- **No source isolates a speedup number for asymmetric actor-critic on a hand task.** Every paper uses
  it; none ablates it with a sample-count ratio. I found no "Nx fewer steps" figure — do not let the
  report invent one.
- No sample-count or wall-clock comparison of residual-vs-scratch PPO in Regrind (the authors report
  success rates, explicitly not step counts).
- No hora hyperparameter table obtained (config path 404'd).

---

## Q6. Warm-starting from a policy trained on a DIFFERENT task (e.g. a grasp-holding policy)

### Takeaway
I found **no source addressing cross-task warm-starting for dexterous manipulation specifically**, and
no paper naming "the warm-started actor degrades once training resumes" as a phenomenon in robot RL.
The closest documented and directly relevant cause is **critic/value mismatch at resume**: the
literature consistently identifies the un-warm-started value function as the destabilizing component,
and reports that value pretraining helps while naive critic-first warm-up can itself destabilize.

### Cited Findings
- Value warm-start helps: "**Value-pretraining injects knowledge into the value model**, which is a
  superior form of value warm-up"; and "while the critic value is typically negative at the start of
  PPO training, **warm starting the critic helps improve the initial stability of gradients**" — [Delve into PPO: Implementation Matters for Stable RLHF](https://openreview.net/pdf?id=rxEmiOEIFL)
- But naive critic-first warm-up hurts: "training the critic before activating actor training ... **can
  lead to unstable training dynamics**" — [search synthesis over PPO fine-tuning literature](https://openreview.net/pdf?id=rxEmiOEIFL)
- Direct-copy warm start is a documented practice: "Parameters from a trained policy can be **directly
  copied to a new network structure**, enabling the learner to start with nearly as good performance,
  and the learner will further improve the policy using a gradient-based approach" — [search synthesis, warm-start PPO literature](https://www.sandia.gov/app/uploads/sites/86/2023/03/Effectiveness_of_Warm_Start_PPO_for_Guidance_with_Highly_Constrained_Nonlinear_Fixed_Wing_Dynamics.pdf)
- Warm-start PPO has a dedicated evaluation in a constrained-dynamics control domain (fixed-wing
  guidance), i.e. the technique is taken seriously outside manipulation — [Effectiveness of Warm-Start PPO for Guidance with Highly Constrained Nonlinear Fixed-Wing Dynamics (Sandia)](https://www.sandia.gov/app/uploads/sites/86/2023/03/Effectiveness_of_Warm_Start_PPO_for_Guidance_with_Highly_Constrained_Nonlinear_Fixed_Wing_Dynamics.pdf)
- PPO collapse is specifically traced to value optimization: "**value optimization holds the secret**"
  to PPO's collapse, with the finding that "value optimization dynamics are more tolerant to variance,
  with the value model favoring higher variance but lower bias, while the **policy model prefers lower
  variance**" — [What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret (arXiv 2503.01491)](https://arxiv.org/pdf/2503.01491)
- Online RL fine-tuning of a pretrained policy is an active area with the finding that retaining the
  original offline data is **not** required — [Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data (arXiv 2412.07762)](https://arxiv.org/pdf/2412.07762)
- An alternative to fine-tuning the pretrained weights at all: keep the base policy frozen and learn a
  bounded residual around it — [Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model (arXiv 2412.13630)](https://arxiv.org/pdf/2412.13630)
- Behaviour-cloning-then-RL at scale, with the pretrained policy kept as a KL anchor — [Video PreTraining (VPT) (arXiv 2206.11795)](https://arxiv.org/pdf/2206.11795)

### Inferences
- The mechanism that best explains "warm-started actor degrades once training resumes", assembled from
  the cited pieces rather than quoted from one: the **actor is competent and the critic is random**, so
  the first advantage estimates are pure noise with large magnitude; PPO's clip does not protect
  against a systematically wrong advantage sign, and the first few updates destroy the transferred
  behaviour before the critic has converged. This is consistent with both "warm-starting the critic
  improves initial gradient stability" and "value optimization holds the secret" to PPO collapse.
- Remedies that follow from the cited material, ranked:
  1. **Warm-start or pretrain the critic too** (roll out the frozen warm-started actor for some
     iterations updating only the value head, then enable the actor) — supported by the value-warmup
     finding, but note the cited caveat that a *long* critic-only phase can itself destabilize; keep it
     short.
  2. **Freeze the pretrained policy and learn a bounded residual** instead of fine-tuning its weights —
     Policy Decorator; this also composes with Regrind's residual formulation.
  3. **KL-anchor to the warm-started policy** for the early iterations — VPT's approach.
  4. **Reset σ to 1.0 at resume, and expect it to be wrong**: a grasp-holding policy's σ will have
     annealed to something small and task-specific. Carrying that σ into a reorientation task gives a
     policy that is both confident and wrong — precisely the reader's symptom. This is my inference
     from the config evidence in Q3, not a cited result.
- Cross-task specifically (grasp-hold → reorient): a grasp-holding policy is trained to *not move*.
  Its optimum is a fixed point that reorientation must escape. There is a real risk the warm start is a
  worse initialization than random for this task pair. **I found no evidence either way** — flag this as
  an open empirical question, cheap to settle with one A/B run.

### Gaps
- **No source on cross-task policy transfer within dexterous manipulation.** All warm-start evidence I
  could reach is either LLM-RLHF or a non-manipulation control domain.
- "Warm-started actor degrades at resume" is **not** a named, documented phenomenon in any source I
  found. The report should present the mechanism as an inference with named contributing causes, not as
  a citable finding.

---

## Q7. When the success bonus almost never fires and the critic has no positive examples

### Takeaway
The reference configs do not rely on the success bonus at all — they carry learning on a **dense
rotation-distance reward**, with the bonus as a late-stage amplifier. HER and goal relabelling are
well documented for exactly this sparsity problem in goal-conditioned manipulation, but they are
**off-policy methods and are not directly usable with PPO** — this is the critical caveat for this
reader.

### Cited Findings — the reference reward is dense, and the bonus is large
- IsaacGymEnvs `ShadowHand.yaml` reward scales: **`distRewardScale: -10.0`, `rotRewardScale: 1.0`,
  `actionPenaltyScale: -0.0002`, `reachGoalBonus: 250`, `fallPenalty: 0.0`,
  `successTolerance: 0.1`** — [IsaacGymEnvs cfg/task/ShadowHand.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/task/ShadowHand.yaml)
- Same file: `episodeLength: 600`, `actionsMovingAverage: 1.0` (no action smoothing in the base task) — [IsaacGymEnvs cfg/task/ShadowHand.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/task/ShadowHand.yaml)
- The Isaac Gym success criterion used for reporting is **"10 consecutive successes at reward of
  3000"** — note 10 x 250 = 2500 of that 3000 is bonus, so the bonus *does* dominate the final
  return, but only once the dense term has already got the policy to tolerance — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)

### Cited Findings — HER and goal relabelling
- "**HER addresses sparse reward challenges** in goal-conditioned reinforcement learning by
  **relabeling failed trajectories with achieved goals**, transforming them into successful
  experiences" — [Hindsight Experience Replay](https://www.researchgate.net/publication/318224066_Hindsight_Experience_Replay)
- HER has been applied to in-hand manipulation: it "is used in learning dexterous in-hand manipulation
  policies that can perform **vision-based object reorientation on a physical Shadow Dexterous Hand**" — [Visual Hindsight Experience Replay](https://www.researchgate.net/publication/330775768_Visual_Hindsight_Experience_Replay)
- Hindsight *goal selection* (choosing which achieved goals to relabel with) is treated as the key
  design choice for long-horizon dexterous tasks — [Wish you were here: Hindsight Goal Selection for long-horizon dexterous manipulation](https://www.researchgate.net/publication/356710935_Wish_you_were_here_Hindsight_Goal_Selection_for_long-horizon_dexterous_manipulation)
- Relay HER for sequential object manipulation with sparse rewards, framed as self-guided continual RL — [Relay Hindsight Experience Replay (arXiv 2208.00843)](https://arxiv.org/pdf/2208.00843)
- Hindsight regularization as a sample-efficiency mechanism for goal-conditioned RL — [GCHR: Goal-Conditioned Hindsight Regularization (arXiv 2508.06108)](https://arxiv.org/pdf/2508.06108)
- Risk-aware replay prioritization specifically for stable in-hand manipulation, built on HER — [Risk-Prioritized Experience Replay for Stable In-Hand Manipulation (Sensors 2026)](https://doi.org/10.3390/s26123633)
- The general in-hand reorientation system line of work reorients "complex and new object shapes by
  **any rotation**" from a single commodity depth camera — the maximal-difficulty goal distribution, for
  contrast with reduced-scope targets — [A System for General In-Hand Object Re-Orientation](https://www.researchgate.net/publication/355924919_A_System_for_General_In-Hand_Object_Re-Orientation)
- hora's reduced goal distribution — **cylinders only, z-axis rotation only** — was sufficient for
  real-robot generalization to dozens of objects — [Qi et al., CoRL 2022](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)
- Curriculum over task difficulty outweighed sensory information for manipulation against gravity — [Curriculum Is More Influential Than Haptic Information (arXiv 2407.09986)](https://arxiv.org/pdf/2407.09986)

### Inferences — ranked, and what is actually usable with PPO
1. **Stop relying on the bonus; verify the dense term.** The reference reward is
   `rotRewardScale: 1.0` on a rotation-distance term plus `distRewardScale: -10.0`, with
   `reachGoalBonus: 250`. If the reader's dense rotation term is weak, mis-signed, or swamped by
   penalties, the bonus firing rate is a symptom and not the disease. This is the cheapest check.
2. **Loosen `successTolerance` and anneal it.** The shipped value is **0.1 rad**, which is generous.
   Starting looser and tightening is the standard curriculum for this task family; the *annealing*
   part I believe is in the DeXtreme Allegro configs but **I did not verify it in a fetched file** —
   flag as unconfirmed.
3. **Easier goal distribution — restrict the axis and the angle.** hora's cylinders-and-z-axis result
   is the strongest cited precedent that a drastically reduced goal set still yields a useful,
   transferable policy.
4. **Seed initial states near success.** Mechanically sound (it converts the sparse-reward problem into
   a short-horizon one) and it is what the reader's existing `grasp_cache` infrastructure is for, but
   **I found no citation for initial-state-near-goal seeding in dexterous RL** — present as engineering,
   not evidence.
5. **HER / goal relabelling: do not plan on it with PPO.** Every HER variant cited above is an
   off-policy replay method. Relabelling rewrites the reward of trajectories collected under a different
   behaviour policy, which breaks PPO's on-policy importance-ratio assumption. Using HER means switching
   to SAC/TD3-family off-policy learning — a defensible choice on a small GPU (replay reuses samples,
   which is exactly what a low-throughput setup needs) but a full rewrite, not a patch.

### Gaps
- No source quantifies a minimum success-firing rate below which PPO stalls on this task.
- I could not confirm from a primary config file that DeXtreme/Allegro anneals `successTolerance`
  toward a `targetSuccessTolerance`. Treat the tolerance-curriculum claim as unverified.
- No citation found for initial-state seeding near success in dexterous in-hand RL.

---

## Q8. Reducing the action space or the control frequency

### Takeaway
Control-frequency reduction is documented in the strongest hand result: **DeXtreme runs sim at 1/60 s
but control at 1/30 s**, a decimation of 2 that halves policy steps per second of simulated time.
Action-space reduction is documented indirectly, via hora's single-axis task and via action smoothing
parameters, but I found no paper that ablates action dimensionality against sample cost.

### Cited Findings
- DeXtreme: **simulation dt = 1/60 s, control dt = 1/30 s** — control runs at half the physics rate — [DeXtreme (ar5iv 2210.13702)](https://ar5iv.labs.arxiv.org/html/2210.13702)
- IsaacGymEnvs base ShadowHand: **`dt = 0.01667` (60 Hz) with `controlFrequencyInv: 1`** — i.e. control
  at the full 60 Hz, no decimation, and **`actionsMovingAverage: 1.0`** (smoothing disabled) — [IsaacGymEnvs cfg/task/ShadowHand.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/task/ShadowHand.yaml)
- Isaac Lab's Shadow Hand vision config raises `num_steps_per_env` to **64** and `max_iterations` to
  **50000** — evidence that longer rollouts per iteration are an accepted configuration, not an
  anti-pattern — [IsaacLab shadow_hand rsl_rl_ppo_cfg.py](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rsl_rl_ppo_cfg.py)
- Reduced task scope as action-space reduction in practice: hora rotates **about the z-axis only**, on
  cylinders only, and still transfers — [Qi et al., CoRL 2022](https://proceedings.mlr.press/v205/qi23a/qi23a.pdf)
- Regrind uses **24 steps per env per iteration** at 4096 envs, i.e. shorter rollouts than Isaac Lab's
  vision config and longer than its MLP config — the horizon is a free parameter across working recipes — [Regrind (ar5iv 2607.11874)](https://ar5iv.labs.arxiv.org/html/2607.11874)
- Tactile-sensing-distribution work studies how much sensing is actually needed for in-hand dexterity,
  the observation-side analogue of action-space reduction — [The Role of Touch: Towards Optimal Tactile Sensing Distribution (arXiv 2509.14984)](https://arxiv.org/pdf/2509.14984)

### Inferences
- **[arithmetic]** Halving control frequency (decimation 2, as DeXtreme does) halves the number of
  policy steps needed per second of simulated behaviour. On a fixed step-per-second budget of ~4,100
  that is an effective **2x** in simulated experience, at the cost of coarser control authority. It is
  the single cheapest throughput win available and it is what the sim-to-real result actually used.
- Reducing the goal distribution (one rotation axis) does not reduce the *action* dimension but does
  shrink the effective state-goal space the policy must cover, which is what a small batch needs. This
  is the hora-shaped scope reduction and it has the best precedent.
- Trading env count for horizon (1024 envs x 64 steps = 65,536 per batch) recovers most of the
  reference batch size within 6 GB, at ~16 s/iteration. Supported by the existence of the
  `num_steps_per_env: 64` vision config and by the Isaac Gym finding that horizon and env count are
  traded against each other.

### Gaps
- No ablation of action dimensionality (e.g. coupled finger joints, reduced DoF) against sample cost
  for in-hand reorientation.
- No source states how low control frequency can go before in-hand reorientation fails.

---

## Bottom line on feasibility (synthesis, clearly marked as such)

### Cited anchors
- Full sim-to-real DeXtreme recipe: **8x A40, 1.41–2.5 days, ~700K FPS, ~42 years of experience** — [DeXtreme (ar5iv 2210.13702)](https://ar5iv.labs.arxiv.org/html/2210.13702)
- Reference task at 8192 envs needs **6.4 GB VRAM**; smallest benchmarked GPU is a **4090 at 170K FPS** — [Isaac Lab performance benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html)
- In-sim milestone for plain ShadowHand: **10 consecutive successes at reward 3000 in 5 minutes** at
  ~150K FPS on an A100 — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)
- Env count buys **wall-clock, ~10x from 256→16384 envs**, with no reported asymptote penalty — [Isaac Gym (ar5iv 2108.10470)](https://ar5iv.labs.arxiv.org/html/2108.10470)

### Inferences — the honest verdict
- **Not achievable on 6 GB**: the DeXtreme-class result (ADR, sim-to-real, 20+ consecutive real-robot
  successes). **[arithmetic]** ~8.5e10 steps at ~4,100 steps/s ≈ **240 days**. The gap is ~40x in
  throughput against one 4090 and ~170x against the 8-GPU DeXtreme cluster. No amount of
  hyperparameter tuning closes two orders of magnitude.
- **Achievable on 6 GB**: an in-simulation cube-reorientation policy reaching the Isaac Gym
  "10 consecutive successes" bar. **[arithmetic]** 5 min x 150K FPS ≈ 4.5e7 steps; at 4,100 steps/s
  that is **~3 hours** of compute. Even with a 10–20x penalty for the smaller batch and worse
  gradient quality, this lands in the **1–4 day** range — the same order as the reader's existing
  10,000-iteration x 4 s ≈ 11 h budget scaled by the 8x sample deficit (**~3.7 days**).
- **Best reduced-scope target to commit to**: hora-shaped. One object class, **one rotation axis**,
  loosened-then-annealed success tolerance, **control decimation 2** (30 Hz control on 60 Hz physics),
  **asymmetric privileged critic**, batch recovered by raising `num_steps_per_env` to 64, and
  **std fixed state-independent at init 1.0 with entropy_coef 0.005**. Sim-only, no ADR, no PBT.
- **The reader's reported symptom is most likely not a compute problem.** Shrinking σ with degrading
  return, against configs that all pin σ state-independent at 1.0 with entropy 0.005 and a raised
  bounds loss, points at std parameterization, `bounds_loss_coef`, and the KL-adaptive LR feedback loop.
  Those are free to fix and should be ruled out before any conclusion is drawn about the 6 GB budget.
