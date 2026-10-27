"""Differentiable MANO blend shapes + linear blend skinning in metres.

No model weights are bundled. Official MANO arrays are supplied locally by the
user. The independent numerical fixture is explicitly gated and is NOT MANO.
Pose is 15 local axis-angle rotations (flat hand); root rotation is separate.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

JOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues with analytic limits at zero, including finite gradients."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*vector.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(angle / torch.pi)[..., None, None]
    b = (0.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    return torch.eye(3, dtype=vector.dtype, device=vector.device) + a * skew + b * (skew @ skew)


def _array(value):
    if hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "r"):
        value = value.r
    return np.asarray(value)


def convert_official_pickle(source, destination, handedness="right", unit_scale=1.0):
    """Convert a TRUSTED official MANO pickle to non-executable NPZ arrays.

    Old Chumpy pickles require Chumpy in a separate conversion environment.
    Never use this function on untrusted pickle files.
    """
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Official MANO model not found: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite model: {destination}")
    if handedness not in ("right", "left") or not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("handedness must be right/left and unit_scale positive")
    try:
        with source.open("rb") as stream:
            data = pickle.load(stream, encoding="latin1")
    except ModuleNotFoundError as error:
        raise RuntimeError("Legacy MANO pickle needs its original dependencies (often chumpy). Convert in an isolated environment; do not downgrade ROS/PyTorch.") from error
    parents = _array(data["kintree_table"])[0].astype(np.int64)
    parents[0] = -1
    arrays = {key: _array(data[key]) for key in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights")}
    for key in ("v_template", "shapedirs", "posedirs"):
        arrays[key] = arrays[key] * unit_scale
    arrays.update(parents=parents, faces=_array(data["f"]).astype(np.int64),
                  hands_mean=_array(data.get("hands_mean", np.zeros(45))),
                  tip_vertex_ids=np.array([745, 317, 444 if handedness == "right" else 445, 556, 673]),
                  metadata_json=np.array(json.dumps(dict(model_type="MANO", handedness=handedness, units="m", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), tips_convention="manopth"))))
    # Validate before saving; no synthetic fallback on corrupt/missing models.
    ManoModel(arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


class ManoModel(nn.Module):
    def __init__(self, source, *, allow_test_model=False, dtype=torch.float32, device="cpu"):
        super().__init__()
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"MANO model missing: {path}. Supply your licensed MANO model; use convert-model for an official .pkl.")
            if path.suffix != ".npz":
                raise ValueError("Use convert-model first: runtime accepts validated non-pickle .npz models")
            with np.load(path, allow_pickle=False) as archive:
                data = dict(archive)
            self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            data = source
            self.sha256 = "in-memory"
        self.metadata = json.loads(str(data.get("metadata_json", "{}")))
        self.model_type = self.metadata.get("model_type")
        if self.model_type != "MANO" and not (allow_test_model and self.model_type == "numerical_fixture"):
            raise ValueError("Not a declared MANO model. Numerical fixtures require explicit allow_test_model=True")
        if self.metadata.get("units") != "m" or self.metadata.get("handedness") not in ("right", "left"):
            raise ValueError("Model metadata must declare metres and handedness")
        expected = dict(v_template=(778, 3), shapedirs=(778, 3, 10), posedirs=(778, 3, 135), J_regressor=(16, 778), weights=(778, 16), parents=(16,), tip_vertex_ids=(5,))
        for name, shape in expected.items():
            array = np.asarray(data[name])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name}: expected finite {shape}, got {array.shape}")
            integer = name in ("parents", "tip_vertex_ids")
            self.register_buffer(name, torch.as_tensor(array.copy(), dtype=torch.long if integer else dtype, device=device))
        faces = np.asarray(data["faces"], dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0 or faces.max() >= 778:
            raise ValueError("faces must be nonempty triangle indices in [0, 778)")
        self.register_buffer("faces", torch.as_tensor(faces, dtype=torch.long, device=device))
        if int(self.parents[0]) != -1 or any(not 0 <= int(self.parents[j]) < j for j in range(1, 16)):
            raise ValueError("Invalid topologically ordered MANO kinematic tree")
        # Build hierarchy once, rather than synchronizing 16 GPU scalars on
        # every optimizer step. Each depth level is one batched matrix product.
        parent_list = self.parents.cpu().tolist()
        depths = [0]
        for joint in range(1,16):
            depths.append(depths[parent_list[joint]]+1)
        self._level_count = max(depths)+1
        flattened, previous = [], []
        for depth in range(self._level_count):
            nodes = [j for j,d in enumerate(depths) if d == depth]
            self.register_buffer(f"_nodes_{depth}", torch.tensor(nodes,device=device,dtype=torch.long))
            if depth:
                parents_in_previous = [previous.index(parent_list[j]) for j in nodes]
                self.register_buffer(f"_parents_{depth}",torch.tensor(parents_in_previous,device=device,dtype=torch.long))
            flattened.extend(nodes)
            previous = nodes
        self.register_buffer("_restore_joint_order",torch.tensor([flattened.index(j) for j in range(16)],device=device,dtype=torch.long))
        self.register_buffer("_safe_parents",torch.tensor([0]+parent_list[1:],device=device,dtype=torch.long))
        if not torch.allclose(self.weights.sum(1), torch.ones(778, dtype=dtype, device=device), atol=1e-4) or (self.weights < 0).any():
            raise ValueError("Skinning weights must be nonnegative and sum to one")
        if not torch.allclose(self.J_regressor.sum(1), torch.ones(16, dtype=dtype, device=device), atol=1e-3):
            raise ValueError("Joint regressor must sum to one per joint")
        if int(self.tip_vertex_ids.min()) < 0 or int(self.tip_vertex_ids.max()) >= 778:
            raise ValueError("Invalid fingertip vertex index")
        # Catch accidental mm input; this is a sanity check, not a scale estimator.
        extent = float(torch.linalg.vector_norm(self.v_template.max(0).values - self.v_template.min(0).values))
        if not 0.03 < extent < 0.6:
            raise ValueError(f"Implausible hand extent {extent:g} m; check model units")

    @property
    def device(self):
        return self.v_template.device

    @property
    def dtype(self):
        return self.v_template.dtype

    def forward(self, theta, beta, global_orient, translation_m):
        """Return (vertices[B,778,3], joints[B,21,3]); t is wrist position."""
        theta = theta.reshape(-1, 15, 3)
        batch = theta.shape[0]
        beta = beta.reshape(-1, 10).expand(batch, -1)
        rotation = axis_angle_to_matrix(global_orient.reshape(-1, 3)).expand(batch, -1, -1)
        translation_m = translation_m.reshape(-1, 3).expand(batch, -1)
        shaped = self.v_template + torch.einsum("vck,bk->bvc", self.shapedirs, beta)
        rest_joints = torch.einsum("jv,bvc->bjc", self.J_regressor, shaped)
        local_rotations = axis_angle_to_matrix(theta)
        feature = (local_rotations - torch.eye(3, dtype=self.dtype, device=self.device)).reshape(batch, 135)
        posed = shaped + torch.einsum("vcp,bp->bvc", self.posedirs, feature)
        root = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch, 1, 3, 3)
        all_rotations = torch.cat([root,local_rotations],1)
        relative = torch.cat([rest_joints[:,:1],rest_joints[:,1:]-rest_joints[:,self._safe_parents[1:]]],1)
        top = torch.cat([all_rotations,relative[...,None]],-1)
        bottom = top.new_tensor([0.,0.,0.,1.]).expand(batch,16,1,4)
        local = torch.cat([top,bottom],-2)
        levels = [local[:,getattr(self,"_nodes_0")]]
        for depth in range(1,self._level_count):
            levels.append(levels[-1][:,getattr(self,f"_parents_{depth}")] @ local[:,getattr(self,f"_nodes_{depth}")])
        transforms = torch.cat(levels,1)[:,self._restore_joint_order]
        joints = transforms[:, :, :3, 3]
        bind_offset = torch.einsum("bjcd,bjd->bjc", transforms[:, :, :3, :3], rest_joints)
        corrected = torch.cat([transforms[:, :, :3, :3], (joints - bind_offset)[..., None]], dim=-1)
        blended = torch.einsum("vj,bjcd->bvcd", self.weights, corrected)
        homogeneous = torch.cat([posed, torch.ones_like(posed[..., :1])], dim=-1)
        vertices = torch.einsum("bvcd,bvd->bvc", blended, homogeneous)
        vertices = vertices - rest_joints[:, :1]
        joints = joints - rest_joints[:, :1]
        vertices = vertices @ rotation.transpose(-1, -2) + translation_m[:, None]
        joints = joints @ rotation.transpose(-1, -2) + translation_m[:, None]
        joints = torch.cat([joints, vertices[:, self.tip_vertex_ids]], dim=1)[:, JOINT_ORDER]
        return vertices, joints
