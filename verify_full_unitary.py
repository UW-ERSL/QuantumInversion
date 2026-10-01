"""verify_full_unitary -- full 16 x 16 error of the wavenumber-controlled stage (App. B).

    python verify_full_unitary.py      # about 10 minutes

delta = max_{k != 0} ||Q(k) - Q_ex(k)||_2, with Q_ex(k) the unitary of the
three-angle factorization at exact angles. At k = (N/2, N/2), where r = 0,
omega is arbitrary and Q_ex uses the circuit's value. Because the work
register returns to zero and the Fourier stages are exact, delta is the
distance between U_V and an exact unitary whose block is V.
"""
import numpy as np

from quantuminversion.isometry_circuit import (IsometryCircuit, T_matrix, F0_matrix, Rpair,
                                                Rpsi, Rq0, exact_angles, vhat_exact)

nu = 0.35
T, F0 = T_matrix(nu), F0_matrix(nu)
print(f"{'N':>4} {'F':>3} {'block err':>10} {'full err':>10} {'ratio':>6} {'7N2^-F':>8} {'ok':>3}")
for n in range(2, 9):
    for F in (12, 16, 20, 24, 28, 32):
        c = IsometryCircuit(n=n, nu=nu, F=F); N = c.N
        Q, chk = c.simulate()
        blk = full = 0.0
        for kk in range(1, N * N):
            k1, k2 = divmod(kk, N)
            b, p1, p2 = exact_angles(N, nu, k1, k2)
            if k1 == k2 == N // 2:
                b = c.decoded['beta'][kk]
            Qex = np.exp(1j * np.pi * (k1 + k2) / N) * T @ Rpair(2 * b, b) @ F0 @ Rpsi(p1, p2) @ Rq0(-b)
            full = max(full, np.linalg.norm(Q[kk] - Qex, 2))
            blk = max(blk, np.linalg.norm(Q[kk][:12, :2] - vhat_exact(N, nu, k1, k2), 2))
        bound = 7 * N * 2.0 ** -F
        ok = full <= bound and chk['work_bits_left_set'] == 0 and chk['inputs_restored']
        print(f"{N:4d} {F:3d} {blk:10.2e} {full:10.2e} {full / blk:6.2f} {bound:8.1e} "
              f"{'yes' if ok else 'NO':>3}", flush=True)
