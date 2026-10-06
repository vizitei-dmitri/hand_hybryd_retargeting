"""LeRobot plugin for a UR10e arm carrying a Tesollo DG5F hand."""

from .config_ur10_dg5f import Ur10Dg5fConfig
from .ur10_dg5f import Ur10Dg5f

__all__ = ["Ur10Dg5f", "Ur10Dg5fConfig"]
