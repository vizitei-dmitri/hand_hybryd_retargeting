"""Build a warm-start checkpoint that carries the ACTOR ONLY across a reward change.

model_4000 of the reward-v3 night run is the best deterministic grasp stabiliser we have
(drop 0.078 against 0.172 for zero actions), so it is worth keeping as the starting manipulation
controller. Its critic, however, estimates the value of the OLD reward and would inject a large,
systematically wrong advantage into the first updates under reward v4; the same goes for the
optimizer moments, the adaptive learning rate it had drifted to (4.5e-5 by iteration 4000) and the
action noise it had inflated to.

So this rewrites a checkpoint in place of a plain --resume:
  kept       actor.*                    the mean network, i.e. the policy itself
  kept       actor_obs_normalizer.*     the input scaling those weights were trained against
  kept       critic_obs_normalizer.*    observation statistics, not value estimates (see below)
  re-init    critic.*                   fresh nn.Linear defaults, same shapes
  reset      std                        explicitly, to --init_noise_std
  reset      optimizer                  param groups kept, moments dropped, lr back to cfg
  reset      iter                        0

Deviation worth stating: the spec says to reinitialize the critic, and the critic MLP is
reinitialized. The critic's observation NORMALIZER is kept, because it holds running observation
statistics rather than anything about the old reward, and discarding it would hand the fresh critic
mis-scaled inputs for its first thousands of steps. The actor normalizer must be kept for the same
reason, only more so: without it the transferred actor weights see a different input scale and the
warm start is worthless.

No Isaac Sim needed: the architecture is unchanged, so this is pure tensor surgery.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

KEEP_PREFIXES = ("actor.", "actor_obs_normalizer.", "critic_obs_normalizer.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init_noise_std", type=float, default=0.2,
                        help="Explicit action std for the new run (spec: 0.15-0.20)")
    parser.add_argument("--learning_rate", type=float, default=1.0e-3,
                        help="Must match the new agent config; the adaptive schedule restarts from it")
    parser.add_argument("--seed", type=int, default=42, help="Seeds the critic re-initialization")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.source, weights_only=False, map_location="cpu")
    old_model = checkpoint["model_state_dict"]

    model = {}
    kept, reinitialized = [], []
    for key, value in old_model.items():
        if key.startswith(KEEP_PREFIXES):
            model[key] = value.clone()
            kept.append(key)
        elif key.startswith("critic."):
            if key.endswith(".weight"):
                # nn.Linear's own default init (kaiming_uniform + fan-in uniform bias), which is
                # what MLP uses: ActorCritic never calls MLP.init_weights.
                out_features, in_features = value.shape
                layer = nn.Linear(in_features, out_features)
                model[key] = layer.weight.detach().clone()
                model[key.replace(".weight", ".bias")] = layer.bias.detach().clone()
                reinitialized.append(key.rsplit(".", 1)[0])
            elif not key.endswith(".bias"):
                raise RuntimeError(f"Unexpected critic parameter {key}")
        elif key in ("std", "log_std"):
            continue  # set explicitly below
        else:
            raise RuntimeError(f"Unexpected checkpoint parameter {key}; refusing to guess")

    if "std" in old_model:
        model["std"] = torch.full_like(old_model["std"], args.init_noise_std)
        std_note = f"std := {args.init_noise_std} (was {float(old_model['std'].mean()):.4f} mean)"
    elif "log_std" in old_model:
        model["log_std"] = torch.full_like(old_model["log_std"], float(torch.log(torch.tensor(args.init_noise_std))))
        std_note = f"log_std := log({args.init_noise_std})"
    else:
        raise RuntimeError("Checkpoint has neither std nor log_std")

    assert set(model) == set(old_model), (set(model) ^ set(old_model))

    # A fresh optimizer: the param-group structure has to survive load_state_dict, the moments
    # must not. RSL-RL's runner.load always loads the optimizer, so an empty state is how the
    # reset is expressed rather than a flag.
    optimizer = {"state": {}, "param_groups": []}
    for group in checkpoint["optimizer_state_dict"]["param_groups"]:
        fresh = dict(group)
        fresh["lr"] = args.learning_rate
        optimizer["param_groups"].append(fresh)

    info = {
        "warm_start_source": str(args.source.resolve()),
        "source_iter": int(checkpoint.get("iter", -1)),
        "kept_parameters": kept,
        "reinitialized_layers": sorted(set(reinitialized)),
        "std": std_note,
        "learning_rate": args.learning_rate,
        "optimizer_state": "dropped (moments and step count)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model, "optimizer_state_dict": optimizer,
                "iter": 0, "infos": {"warm_start": info}}, args.out)

    print("[WARM] " + json.dumps(info, indent=1))
    print(f"[WARM] kept {len(kept)} tensors, re-initialized {len(set(reinitialized))} critic layers")
    print(f"[WARM] wrote {args.out}")


if __name__ == "__main__":
    main()
