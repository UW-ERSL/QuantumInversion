# QuantumInversion

Fast-inversion block encoding of preconditioned periodic elasticity with
mesh-independent condition number.

For a periodic two-phase Q4 elasticity cell with contrast `rho`, the operator
preconditioned by the homogeneous operator `K`,

    M = (K^+)^{1/2} K^chi (K^+)^{1/2} / E_max,

factors as `V^T Etilde V`, where `V = W^{1/2} G (K^+)^{1/2}` is a partial
isometry and `Etilde` is diagonal. Because `K` is block-circulant, `V` is a
quantum Fourier transform pair around one 12 x 2 isometry per wavenumber. The
resulting block encoding of `M` has subnormalization 1 and condition number at
most `rho` for every mesh size `N`. This package builds the wavenumber-controlled
stage of that circuit at gate level and verifies it.

The two-phase operator itself, and the O(1) oracle for a square inclusion, come
from the companion package [`pyblockencode`](https://github.com/UW-ERSL/PyBlockEncode).

## Install

    pip install git+https://github.com/UW-ERSL/QuantumInversion

or, from a clone, `pip install -r requirements.txt`. Only numpy and scipy are
required; Qiskit and Qiskit Aer are needed only for the optional cross-check.

## Quick start

```python
from quantuminversion import IsometryCircuit, vhat_exact
import numpy as np

circ = IsometryCircuit(n=4, nu=0.35, F=24)        # N = 16, 24 fixed-point bits
Q, checks = circ.simulate()                       # every wavenumber, gate by gate
N = circ.N
err = max(np.linalg.norm(Q[k1 * N + k2, :12, :2] - vhat_exact(N, 0.35, k1, k2), 2)
          for k1 in range(N) for k2 in range(N))
print(checks, err, circ.resources()["toffoli_total"])
```

    python verify_isometry.py          # circuit checks, about 1 minute
    python verify_isometry.py --aer    # adds the Qiskit Aer cross-check, about 5 minutes
    python verify_square.py            # Sec. 4 numerics, about 10 minutes

## Layout

| File | Contents |
|---|---|
| `quantuminversion/fourier_symbol.py` | Closed-form 2 x 2 symbol of `K`, its inverse and inverse square root |
| `quantuminversion/material.py` | Plane-strain `D` |
| `quantuminversion/revcirc.py` | X/CX/CCX reversible fixed-point arithmetic (Cuccaro adders, CORDIC), bit-packed simulator |
| `quantuminversion/isometry_circuit.py` | Closed-form factorization of `Vhat(k)` into three angles; the gate-level circuit `U_V`; dense reference `V` |
| `verify_isometry.py` | Arithmetic, factorization, circuit, and scaling checks |
| `verify_isometry_aer.py` | The exported circuit in Qiskit Aer (MPS) for every wavenumber at N = 4 |
| `verify_square.py` | Condition numbers of `K^chi` and of the preconditioned operator, and the encoding of `M`, for a square inclusion |

## Factorization

With `phi_j = pi k_j / N`, `a = sin phi1 cos phi2`, `b = cos phi1 sin phi2`,
`zeta = sin phi1 sin phi2` and `r = (a^2 + b^2)^{1/2}`,

    Vhat(k) = e^{i(phi1 + phi2)} T . Rpair(2 omega, omega) . F0 . Rtau(tau1, tau2) . R0(-omega),

    omega = atan2(b, a),   tau1 = atan2(gamma zeta, sqrt(lam + 2 mu) r),
    tau2 = atan2(gamma zeta, sqrt(mu) r),   gamma^2 = (lam + 3 mu) / 3,

with `T` and `F0` fixed 16 x 16 unitaries given in closed form. The three
angles come from two CORDIC rotations of `pi (k1 +- k2) / N` and three CORDIC
vectorings; the decision bits control the rotations directly, and the
arithmetic is then reversed so that every work wire returns to zero.

## What is verified

- Adders exhaustively; CORDIC to its truncation error.
- Closed-form factorization to 3e-15 for every `k`, `nu` in {-0.3, 0, 0.35, 0.45}.
- Gate-level simulation for every `k`, `N` = 4 ... 256, `F` = 12 ... 32
  (39 cases): work register clean in every case, block error `delta <= 7 N 2^-F`.
- Dense `U_V`, with the QFT built from gates, against `V` assembled element by
  element (N = 8, 16).
- The exported 727-qubit circuit in Qiskit Aer (MPS) for every `k` at N = 4,
  agreeing to 9e-10.
- Square inclusion, plane strain, `nu = 0.35`, `v_f = 1/4`, `N` = 8 ... 64,
  `rho` = 1.5 ... 1e4: `kappa(M) = rho` exactly (the bound is attained), while
  the condition number of `K^chi` seen by QSVT grows by 4.03 to 4.26 per
  doubling of `N`; the block of `U_V^dag U_E U_V` from the gate-level circuit
  matches `M + cbar Pi_0` to below `2 delta`.

## Cost per application of U_V

Toffoli = 51 F^2 + 196 F + 38 + 4 log2 N (exact in all 39 cases), with
`F = ceil(log2(7 N / delta))` for block error `delta`; 5F + 3 controlled
single-qubit rotations; about 4 F^2 qubits. For N = 1024 and delta = 1e-6:
F = 33, 62,270 Toffolis, 4,170 qubits.

## What is NOT established

1. **Rotation synthesis:** the rotations are simulated exactly; Clifford+T
   synthesis is not included.
2. **Qubit count:** intermediate CORDIC values are kept until the final
   uncomputation; pebbling is not applied.
3. **Error bound:** `delta <= 7 N 2^-F` is measured in every case tested, not
   proved for all `N`.
4. **Readout:** how the preconditioned inverse is read out is outside this
   package.

## Conventions

Plane strain, engineering shear. `N_dof = 2 N^2`. The DFT has entries
`N^-1 e^{-2 pi i k.j/N}`. Local index `3 g + c` for Gauss point `g` and strain
component `c` (xx, yy, gamma); element `(i, j)` has index `i N + j`, with `i`
along `x`.
