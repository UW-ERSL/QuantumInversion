"""quantuminversion -- fast-inversion block encoding of preconditioned periodic elasticity.

The preconditioned operator M = (K^+)^{1/2} K^chi (K^+)^{1/2} / E_max of a periodic
two-phase Q4 elasticity cell factors as V^T Etilde V with V a partial isometry, and
V = (F^dag x I_12) blockdiag_k Vhat(k) (F x I_2). The block encoding of M has
subnormalization 1 and condition number at most rho for every mesh size N.

    from quantuminversion import IsometryCircuit
    circ = IsometryCircuit(n=4, nu=0.35, F=24)   # N = 16, 24 fixed-point bits
    Q, checks = circ.simulate()                  # gate-level, every wavenumber
    print(circ.resources())                      # Toffoli, wires, rotations
"""
from .material import plane_strain
from .fourier_symbol import symbol, symbol_inverse, symbol_inv_sqrt
from .isometry_circuit import (IsometryCircuit, vhat_exact, V_assembled, T_matrix,
                               F0_matrix, exact_angles, canonical, end_to_end_error)
from .revcirc import Circuit, BitSim

__all__ = ["plane_strain", "symbol", "symbol_inverse", "symbol_inv_sqrt",
           "IsometryCircuit", "vhat_exact", "V_assembled", "T_matrix", "F0_matrix",
           "exact_angles", "canonical", "end_to_end_error", "Circuit", "BitSim"]
