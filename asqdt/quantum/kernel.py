"""
Statevector quantum simulator for parameterized quantum kernels.

Implements the exact same mathematics as PennyLane/Qiskit but in pure NumPy,
which is standard practice for quantum ML papers targeting non-quantum venues
(Havlíček et al. Nature 2019, Cultice et al. ISVLSI 2024, Terada et al. 2025).

Key objects:
  - Parameterized circuit U_θ(ξ) = Π_l [U_ent^(l) · U_enc^(l)(ξ;θ)] |0⟩
  - Fidelity kernel k_θ(ξ,ξ') = |⟨0|U_θ(ξ)† U_θ(ξ')|0⟩|²
  - Parameter-shift gradient ∂k/∂θ_j
"""
import numpy as np
from .. import config as cfg


# ── Single-qubit gates ─────────────────────────────────────────────

def ry(angle):
    """Rotation about Y axis."""
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(angle):
    """Rotation about Z axis."""
    return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])


def _kron_at(gate, qubit, n_qubits):
    """Embed a single-qubit gate at position `qubit` in n-qubit space."""
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def cnot(control, target, n_qubits):
    """CNOT gate between control and target qubits."""
    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(format(i, f'0{n_qubits}b'))
        if bits[control] == '0':
            U[i, i] = 1.0
        else:
            bits[target] = '1' if bits[target] == '0' else '0'
            j = int(''.join(bits), 2)
            U[j, i] = 1.0
    return U


# ── Precompute CNOT matrices ──────────────────────────────────────

_cnot_cache = {}

def _get_cnot(c, t, nq):
    key = (c, t, nq)
    if key not in _cnot_cache:
        _cnot_cache[key] = cnot(c, t, nq)
    return _cnot_cache[key]


# ── Parameterized Quantum Circuit ─────────────────────────────────

class QuantumCircuit:
    """
    Parameterized feature-map circuit:
      U_θ(ξ) = Π_{l=1}^{D} [U_ent^(l) · U_enc^(l)(ξ; θ)]
    
    Encoding layer l:
      For each qubit i: RY(θ_{l,i,0} · ξ_{j}) · RZ(θ_{l,i,1} · ξ_{j})
      where j = ((l-1)*nq + i) mod d_features
    
    Entangling layer l:
      CNOT(i, i+1) for i = 0..nq-2, plus CNOT(nq-1, 0) if nq > 2
    
    Parameters θ: shape (depth, nq, 2) — scaling for RY and RZ per qubit per layer
    """

    def __init__(self, n_qubits=cfg.N_QUBITS, depth=cfg.CIRCUIT_DEPTH):
        self.nq = n_qubits
        self.depth = depth
        self.dim = 2 ** n_qubits
        self.n_params = depth * n_qubits * 2
        # Precompute entangling layer
        self._ent = np.eye(self.dim, dtype=complex)
        for i in range(n_qubits - 1):
            self._ent = _get_cnot(i, i + 1, n_qubits) @ self._ent
        if n_qubits > 2:
            self._ent = _get_cnot(n_qubits - 1, 0, n_qubits) @ self._ent

    def initial_params(self, rng=None):
        """Random initialization of θ."""
        if rng is None:
            rng = np.random.RandomState()
        return rng.uniform(0.5, 1.5, size=(self.depth, self.nq, 2))

    def statevector(self, xi, theta):
        """Compute U_θ(ξ)|0⟩."""
        d = len(xi)
        psi = np.zeros(self.dim, dtype=complex)
        psi[0] = 1.0  # |0...0⟩

        for l in range(self.depth):
            # Encoding layer
            for i in range(self.nq):
                j = ((l * self.nq) + i) % d
                angle_y = theta[l, i, 0] * xi[j]
                angle_z = theta[l, i, 1] * xi[j]
                Ry = _kron_at(ry(angle_y), i, self.nq)
                Rz = _kron_at(rz(angle_z), i, self.nq)
                psi = Rz @ Ry @ psi
            # Entangling layer
            psi = self._ent @ psi
        return psi

    def fidelity_kernel(self, xi, xi_prime, theta, n_shots=0):
        """
        k_θ(ξ, ξ') = |⟨0|U_θ(ξ)† U_θ(ξ')|0⟩|²
        
        If n_shots > 0, simulate shot noise via binomial sampling.
        """
        psi = self.statevector(xi, theta)
        psi_p = self.statevector(xi_prime, theta)
        overlap = np.abs(np.vdot(psi, psi_p)) ** 2

        if n_shots > 0:
            # Swap-test simulation: P(0) = (1 + |⟨ψ|ψ'⟩|²) / 2
            p0 = (1 + overlap) / 2
            count_0 = np.random.binomial(n_shots, p0)
            overlap_est = 2 * (count_0 / n_shots) - 1
            return np.clip(overlap_est, 0, 1)
        return float(overlap)

    def gram_matrix(self, X, theta, n_shots=0):
        """Compute kernel Gram matrix K[i,j] = k_θ(X[i], X[j])."""
        N = len(X)
        K = np.zeros((N, N))
        for i in range(N):
            K[i, i] = 1.0  # k(x,x) = 1 for fidelity kernel
            for j in range(i + 1, N):
                K[i, j] = self.fidelity_kernel(X[i], X[j], theta, n_shots)
                K[j, i] = K[i, j]
        return K

    def cross_gram(self, X, Y, theta, n_shots=0):
        """Cross kernel matrix K[i,j] = k_θ(X[i], Y[j])."""
        K = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                K[i, j] = self.fidelity_kernel(X[i], Y[j], theta, n_shots)
        return K

    def spsa_gradient(self, X_nom, X_atk, theta, mmd_fn, rng=None, eps=0.1):
        """
        SPSA gradient: 2 function evaluations regardless of parameter count.
        Much faster than parameter-shift for large θ.
        """
        if rng is None:
            rng = np.random.RandomState()
        delta = rng.choice([-1, 1], size=theta.shape)
        mmd_plus = mmd_fn(X_nom, X_atk, theta + eps * delta)
        mmd_minus = mmd_fn(X_nom, X_atk, theta - eps * delta)
        grad = (mmd_plus - mmd_minus) / (2 * eps) * delta
        return grad

    def parameter_shift_gradient(self, X_nom, X_atk, theta, mmd_fn):
        """
        Exact parameter-shift gradient: 2×n_params function evaluations.
        More accurate than SPSA but O(n_params) times slower.
        """
        shift = np.pi / 2
        grad = np.zeros_like(theta)
        for l in range(self.depth):
            for i in range(self.nq):
                for k in range(2):
                    tp = theta.copy(); tp[l, i, k] += shift
                    tm = theta.copy(); tm[l, i, k] -= shift
                    grad[l, i, k] = (mmd_fn(X_nom, X_atk, tp) - mmd_fn(X_nom, X_atk, tm)) / 2
        return grad

    def parameter_shift_gradient(self, X_nom, X_atk, theta, mmd_fn):
        """
        Exact parameter-shift gradient: 2 × n_params function evaluations.
        Use for final verification or small parameter counts.
        """
        shift = np.pi / 2
        grad = np.zeros_like(theta)
        for l in range(self.depth):
            for i in range(self.nq):
                for k in range(2):
                    tp = theta.copy(); tp[l, i, k] += shift
                    tm = theta.copy(); tm[l, i, k] -= shift
                    grad[l, i, k] = (mmd_fn(X_nom, X_atk, tp) - mmd_fn(X_nom, X_atk, tm)) / 2
        return grad


# ── Classical kernels for baselines ────────────────────────────────

def rbf_kernel(X, Y, gamma=None):
    """RBF (Gaussian) kernel matrix."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    return sk_rbf(X, Y, gamma=gamma)


def laplacian_kernel(X, Y, gamma=None):
    """Laplacian kernel matrix."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    from sklearn.metrics.pairwise import laplacian_kernel as sk_lap
    return sk_lap(X, Y, gamma=gamma)
