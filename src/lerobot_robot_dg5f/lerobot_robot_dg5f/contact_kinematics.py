"""Small URDF FK/Jacobian evaluator used only for closing/relief classification.

No SDK, dynamics, optimization or MuJoCo. Jacobians are metres per degree.
Fingertip distances are NOT full mesh/self-collision detection.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np

from dg5f_teleop.contact_signals import PAIRS
from .constants import JOINT_NAMES


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


class ContactKinematics:
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.records = {}
        seen = set()
        for joint in root.findall("joint"):
            kind, name = joint.attrib["type"], joint.attrib["name"]
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"Unsupported safety FK joint type {kind}")
            origin = joint.find("origin")
            attributes = {} if origin is None else origin.attrib
            xyz = np.fromstring(attributes.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(attributes.get("rpy", "0 0 0"), sep=" ")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
            index = None
            axis = np.array([1., 0., 0.])
            if kind != "fixed":
                if name not in JOINT_NAMES:
                    raise ValueError(f"Unknown actuated safety FK joint {name}")
                index = JOINT_NAMES.index(name)
                seen.add(name)
                element = joint.find("axis")
                if element is not None:
                    axis = np.fromstring(element.attrib["xyz"], sep=" ")
                if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 0:
                    raise ValueError(f"Invalid axis for {name}")
                axis /= np.linalg.norm(axis)
            child = joint.find("child").attrib["link"]
            self.records[child] = (joint.find("parent").attrib["link"], transform, index, axis)
        if seen != set(JOINT_NAMES):
            raise ValueError("Safety URDF does not contain all DG5F joints")
        self.chains = []
        for finger in range(1, 6):
            chain, link, visited = [], f"rl_dg_{finger}_tip", set()
            if link not in self.records:
                raise ValueError(f"Missing fingertip {link}")
            while link in self.records:
                if link in visited:
                    raise ValueError("Cycle in URDF")
                visited.add(link)
                record = self.records[link]
                chain.append(record)
                link = record[0]
            self.chains.append(list(reversed(chain)))

    def tips_and_jacobians(self, positions_deg, *, with_jacobians=True):
        q = np.asarray(positions_deg, dtype=np.float64)
        if q.shape != (20,) or not np.all(np.isfinite(q)):
            raise ValueError("Safety FK requires 20 finite degree positions")
        tips = np.zeros((5, 3))
        jacobians = np.zeros((5, 3, 20))
        for finger, chain in enumerate(self.chains):
            transform, axes = np.eye(4), []
            for _, origin, index, axis in chain:
                transform = transform @ origin
                if index is not None:
                    if with_jacobians:
                        axes.append((index, transform[:3, :3] @ axis, transform[:3, 3].copy()))
                    motion = np.eye(4)
                    motion[:3, :3] = rotation(axis, np.deg2rad(q[index]))
                    transform = transform @ motion
            tips[finger] = transform[:3, 3]
            for index, axis, origin in axes:
                jacobians[finger, :, index] = np.cross(axis, tips[finger] - origin) * np.pi / 180
        return tips, jacobians

    def pair_gradients(self, positions_deg):
        return self.pair_geometry(positions_deg)[1]

    def pair_distances(self, positions_deg):
        tips, _ = self.tips_and_jacobians(positions_deg, with_jacobians=False)
        return np.array([np.linalg.norm(tips[b] - tips[a]) for a, b in PAIRS])

    def pair_geometry(self, positions_deg):
        """Distances and their gradients from the SAME existing URDF evaluator."""
        tips, jacobians = self.tips_and_jacobians(positions_deg)
        result = np.zeros((len(PAIRS), 20))
        distances = np.zeros(len(PAIRS))
        for index, (a, b) in enumerate(PAIRS):
            difference = tips[b] - tips[a]
            distance = np.linalg.norm(difference)
            distances[index] = distance
            if distance > 1e-8:
                result[index] = difference / distance @ (jacobians[b] - jacobians[a])
        return distances, result
