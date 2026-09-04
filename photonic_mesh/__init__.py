"""Programmable photonic mesh: Clements decomposition + fab-tolerance model."""
from .components import (
    coupler_matrix, phase_matrix, mzi_matrix, embed_2x2,
    dc_cross_coupling, coupler_angle_from_splitting,
)
from .clements import (
    MZISetting, MeshProgram, decompose,
    build_mesh, build_ideal, fidelity, transmission_db,
)
from .coupler_fem import FEMCouplerModel, get_model, dc_cross_coupling_fem

__all__ = [
    "coupler_matrix", "phase_matrix", "mzi_matrix", "embed_2x2",
    "dc_cross_coupling", "coupler_angle_from_splitting",
    "MZISetting", "MeshProgram", "decompose",
    "build_mesh", "build_ideal", "fidelity", "transmission_db",
    "FEMCouplerModel", "get_model", "dc_cross_coupling_fem",
]
__version__ = "1.0.0"
