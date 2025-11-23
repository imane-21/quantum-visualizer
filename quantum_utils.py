"""Quantum utilities: state parsing, normalization, gates using NumPy."""

from typing import List, Tuple
import numpy as np

# Basic single-qubit gates
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)


def parse_state_vector(s: str) -> np.ndarray:
    """Parse a comma-separated state vector string into a numpy array of complex numbers.

    Accepts entries like '1', '0', '1/\u221a2', '0.707+0.707j', '1j', etc.
    """
    s = s.strip()
    if s == '':
        raise ValueError("Empty state string")

    # Support Dirac notation like |0>, |1>, |+>, |->, and multi-qubit like |00>, |+0>
    if s.startswith('|') and s.endswith('>'):
        ket = s[1:-1].strip()
        # single-ket like |+>
        if all(ch in '01+-' for ch in ket):
            # build tensor product of basis or plus/minus states
            qubits = []
            for ch in ket:
                if ch == '0':
                    qubits.append(np.array([1, 0], dtype=complex))
                elif ch == '1':
                    qubits.append(np.array([0, 1], dtype=complex))
                elif ch == '+':
                    qubits.append(1/np.sqrt(2) * np.array([1, 1], dtype=complex))
                elif ch == '-':
                    qubits.append(1/np.sqrt(2) * np.array([1, -1], dtype=complex))
            # tensor product: leftmost char is qubit 0 (MSB)
            state = qubits[0]
            for q in qubits[1:]:
                state = np.kron(state, q)
            return state
        else:
            raise ValueError(f"Unsupported ket notation: {s}")

    # Otherwise treat as comma-separated list of numeric expressions
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if len(parts) == 0:
        raise ValueError("Empty state string")

    vals = []
    for p in parts:
        # try eval with safe replacements
        try:
            # replace unicode sqrt symbol if present
            p = p.replace('\u221a', 'np.sqrt')
            # allow j for imaginary
            val = complex(eval(p, {"np": np, "sqrt": np.sqrt}))
        except Exception as e:
            raise ValueError(f"Can't parse element '{p}': {e}")
        vals.append(val)
    return np.array(vals, dtype=complex)


def norm(state: np.ndarray) -> float:
    return float(np.vdot(state, state).real)


def is_normalized(state: np.ndarray, tol: float = 1e-9) -> bool:
    return abs(norm(state) - 1.0) < tol


def normalize(state: np.ndarray) -> np.ndarray:
    n = np.sqrt(norm(state))
    if n == 0:
        raise ValueError("Zero vector can't be normalized")
    return state / n


def tensor_product(gates: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of a list of gates (left-most is first qubit)."""
    result = gates[0]
    for g in gates[1:]:
        result = np.kron(result, g)
    return result


def apply_gate(state: np.ndarray, gate: np.ndarray, target_qubits: List[int], num_qubits: int) -> np.ndarray:
    """Apply a (possibly multi-qubit) gate to state vector.

    gate: matrix for len(target_qubits) qubits.
    target_qubits: list of qubit indices (0 is most significant / left)
    num_qubits: total number of qubits
    """
    # Build full operator as tensor of either gate or identity
    # Qubit ordering: we assume leftmost tensor factor is qubit 0. For kron order, we need to build [q0, q1,...]
    ops = []
    tset = set(target_qubits)
    tq_sorted = sorted(target_qubits)
    # If gate acts on contiguous block with correct order, place it
    # We'll build by placing identity or small gate blocks. Simpler approach: build by tensoring identities and when encountering first target, put the gate and skip the others.
    k = 0
    i = 0
    while i < num_qubits:
        if i in tset:
            # find length of contiguous block starting at i
            j = i
            while j < num_qubits and j in tset:
                j += 1
            # place gate (must match size)
            ops.append(gate)
            i = j
        else:
            ops.append(np.eye(2, dtype=complex))
            i += 1
    full_op = ops[0]
    for op in ops[1:]:
        full_op = np.kron(full_op, op)
    return full_op @ state


def single_qubit_gate_by_name(name: str) -> np.ndarray:
    name = name.upper()
    if name == 'X':
        return X
    if name == 'Y':
        return Y
    if name == 'Z':
        return Z
    if name == 'H':
        return H
    if name == 'S':
        return S
    if name == 'T':
        return T
    raise ValueError(f'Unknown gate {name}')


def computational_probabilities(state: np.ndarray) -> List[float]:
    probs = np.abs(state)**2
    return [float(p) for p in probs]


def apply_cnot(state: np.ndarray, control: int, target: int, num_qubits: int) -> np.ndarray:
    """Apply a CNOT with given control and target qubit indices to the state vector.

    Qubit indexing: 0 is leftmost (MSB) tensor factor.
    """
    if control == target:
        raise ValueError("Control and target must be different")
    # projectors
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    # build first term: control in |0> -> identity on target
    ops0 = []
    for i in range(num_qubits):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(I)
    term0 = ops0[0]
    for op in ops0[1:]:
        term0 = np.kron(term0, op)

    # build second term: control in |1> -> apply X on target
    ops1 = []
    for i in range(num_qubits):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(X)
        else:
            ops1.append(I)
    term1 = ops1[0]
    for op in ops1[1:]:
        term1 = np.kron(term1, op)

    full_op = term0 + term1
    return full_op @ state
