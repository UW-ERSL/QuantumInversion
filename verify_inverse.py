"""verify_inverse -- the preconditioned inverse applied to specific loads (Sec. 4).

    python verify_inverse.py           # about 5 minutes

u_Q = E_max^{-1} (K^+)^{1/2} q_t(Mtilde) (K^+)^{1/2} f, with q_t the odd QSVT
polynomial of Sec. 3.5, is compared with the finite element solution
u = (K^chi)^+ f in the energy norm of K^chi. Prop. 5 bounds the relative
error by eps_p for every load orthogonal to the rigid translations. FFT-PCG
(preconditioner K^+) is run to the same energy-norm error for comparison.
Square inclusion, v_f = 1/4, plane strain, nu = 0.35, E_max = 1.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from verify_square import assemble, square_chi, symbol, circuit_block, NU
from quantuminversion.isometry_circuit import D0


# ---- block-circulant functions of K by FFT ----------------------------------
def k_function(f, N):
    S = symbol(N).reshape(-1, 2, 2)
    e, V = np.linalg.eigh(S)
    fe = np.where(e > 1e-12, f(np.maximum(e, 1e-300)), 0.0)
    T = np.einsum("kab,kb,kcb->kac", V, fe, V).reshape(N, N, 2, 2)

    def apply(x):
        X = np.fft.fft2(x.reshape(N, N, 2, -1), axes=(0, 1))
        return np.fft.ifft2(np.einsum("ijab,ijbm->ijam", T, X), axes=(0, 1)).real.reshape(2 * N * N, -1)

    def apply_cols(x):                                          # in column blocks, to bound memory
        if x.ndim == 1:
            return apply(x).ravel()
        return np.hstack([apply(x[:, c:c + 512]) for c in range(0, x.shape[1], 512)])
    return apply_cols


# ---- odd QSVT polynomial q_t at kappa = rho ---------------------------------
def q_poly(rho, eps):
    a = 1 / rho ** 2
    s = (1 + a) / (1 - a)
    t = int(np.ceil(0.5 * rho * np.log(2 / eps)))
    Ts = np.cosh(t * np.arccosh(s))

    def q(x):
        z = (1 + a - 2 * x ** 2) / (1 - a)
        Tz = np.where(np.abs(z) <= 1, np.cos(t * np.arccos(np.clip(z, -1, 1))),
                      np.cosh(t * np.arccosh(np.maximum(np.abs(z), 1))))
        return (1 - Tz / Ts) / x
    return q, 2 * t - 1


# ---- loads, solutions, errors -----------------------------------------------
def no_translation(f):
    f = f.copy()
    f[0::2] -= f[0::2].mean()
    f[1::2] -= f[1::2].mean()
    return f


def loads(N, A, Et, Kh, lam, U):
    n = N * N
    i, j = np.divmod(np.arange(n), N)
    x, y = i / N, j / N
    s = np.zeros(2 * n)
    s[0::2] = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    s[1::2] = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    d, Qm = np.linalg.eigh(D0(NU))
    wd = np.tile(((Qm * np.sqrt(d)) @ Qm.T) @ np.array([1.0, 0, 0]) / (2 * N), 4 * n)
    nz = lam > 1e-9
    f = {"smooth": s,
         "random": np.random.default_rng(0).standard_normal(2 * n),
         "cell": -A.T @ (Et * wd),                              # macroscopic strain e_xx
         "worst": Kh(U[:, nz][:, np.argmin(lam[nz])])}          # eigenvector at 1/rho
    return {k: no_translation(v) for k, v in f.items()}


def fe_solve(Kc, f):
    keep = np.arange(2, Kc.shape[0])                            # pin node 0, then remove translation
    u = np.zeros_like(f)
    u[keep] = spla.spsolve(Kc[keep][:, keep].tocsc(), f[keep])
    return no_translation(u)


def energy_error(Kc, u, x):
    e = x - u
    return np.sqrt(e @ (Kc @ e) / (u @ (Kc @ u)))


def pcg_iterations(Kc, Kinv, f, u, eps):
    x = np.zeros_like(f); r = f.copy(); z = Kinv(r); p = z.copy(); rz = r @ z
    for it in range(1, 10 ** 5):
        Ap = Kc @ p; al = rz / (p @ Ap)
        x += al * p; r -= al * Ap
        if energy_error(Kc, u, x) <= eps:
            return it
        z = Kinv(r); rz, rz0 = r @ z, rz; p = z + (rz / rz0) * p


def apply_inverse(q, Kih, lam, U, cbar, f):
    lt = np.where(np.abs(lam) > 1e-9, lam, cbar)                # spectrum of Mtilde
    return Kih(U @ (q(lt) * (U.conj().T @ Kih(f))))


# ---- main ------------------------------------------------------------------
if __name__ == "__main__":
    eps = 1e-6
    names = ("smooth", "random", "cell", "worst")
    print(f"{'N':>3} {'rho':>6} {'F':>4} {'d':>7} {'PCG':>4}  " + "  ".join(f"{k:>9}" for k in names))
    for rho in (10.0, 1e4):
        q, d = q_poly(rho, eps)
        for N in (8, 16, 32, 64):
            G, Wh = assemble(N)
            A = (Wh @ G).tocsr()
            chi = square_chi(N)
            Et = np.repeat(np.where(chi == 1, 1.0, 1 / rho), 12)
            Kc = (A.T @ sp.diags(Et) @ A).tocsr()               # K^chi / E_max
            Kih, Kh, Kinv = (k_function(g, N) for g in (lambda e: e ** -0.5, np.sqrt, lambda e: 1 / e))
            M = Kih(Kc @ Kih(np.eye(2 * N * N)))
            lam, U = np.linalg.eigh((M + M.T) / 2)
            del M
            cbar = chi.mean() + (1 - chi.mean()) / rho
            F_ = loads(N, A, Et, Kh, lam, U)
            sols = {k: fe_solve(Kc, f) for k, f in F_.items()}
            errs = [energy_error(Kc, sols[k], apply_inverse(q, Kih, lam, U, cbar, F_[k])) for k in names]
            its = max(pcg_iterations(Kc, Kinv, F_[k], sols[k], eps) for k in names)
            print(f"{N:3d} {rho:6g} {'-':>4} {d:7d} {its:4d}  " + "  ".join(f"{e:9.2e}" for e in errs), flush=True)
            if rho == 10.0 and N <= 16:                          # Mtilde from the gate-level circuit
                for Fb in (24, 32):
                    Mc, chk = circuit_block(N, rho, Fb)
                    lc, Uc = np.linalg.eigh((Mc + Mc.conj().T) / 2)
                    errs = [energy_error(Kc, sols[k], apply_inverse(q, Kih, lc, Uc, cbar, F_[k].astype(complex)).real)
                            for k in names]
                    print(f"{N:3d} {rho:6g} {Fb:4d} {d:7d} {'':>4}  " + "  ".join(f"{e:9.2e}" for e in errs), flush=True)
            del lam, U
