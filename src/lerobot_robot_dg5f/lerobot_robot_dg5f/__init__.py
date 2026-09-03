"""LeRobot plugin for the Tesollo DG5F hand."""

from .config_dg5f import Dg5fConfig
from .dg5f import Dg5f

__all__ = ["Dg5f", "Dg5fConfig"]
