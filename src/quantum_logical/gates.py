import numpy as np
import qutip as qt

def sqrtISWAP_operator(dim: int, level:int) -> qt.Qobj:
    """Return a logical sqrt iSWAP operator for the given dimension and swap level.

    Supports `dim==2` (standard iSWAP) and `dim==3` (qutrit analogue used in the
    notebook). 
    For `dim==3`, the `level` argument specifies which pair of levels to swap:
      - level=1: swap |0> and |1>
      - level=2: swap |1> and |2>
    The returned object is a `qutip.Qobj` usable in tensor constructions.

    Args:
        dim: The Hilbert-space dimension (2 or 3).
        level: The level of the swap (1 or 2).

    Returns:
        A `qutip.Qobj` representing the iSWAP-like gate.

    Raises:
        ValueError: if `dim` is not 2 or 3.
    """
    if dim == 2:
        return qt.Qobj([[1, 0, 0, 0],
                        [0, 1/np.sqrt(2), 1j/np.sqrt(2), 0],
                        [0, 1j/np.sqrt(2), 1/np.sqrt(2), 0],
                        [0, 0, 0, 1]], dims=[[2, 2], [2, 2]])
    if dim == 3:
        if level == 1:
            # Turn |01> into (|01> + i|10>)/sqrt(2) and |10> into (|10> + i|01>)/sqrt(2)
            return qt.Qobj([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1/np.sqrt(2), 0, 1j/np.sqrt(2), 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1j/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ], dims=[[3, 3], [3, 3]])
        elif level == 2:
            # Swap |12> and |21>
            return qt.Qobj([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1/np.sqrt(2), 0, 1j/np.sqrt(2), 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1j/np.sqrt(2), 0, 1/np.sqrt(2), 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dims=[[3, 3], [3, 3]])
        elif level == 3:
            # Turn |02> into (|02> + i|20>)/sqrt(2) and |20> into (|20> + i|02>)/sqrt(2)
            return qt.Qobj([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1/np.sqrt(2), 0, 0, 0, 1j/np.sqrt(2), 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1j/np.sqrt(2), 0, 0, 0, 1/np.sqrt(2), 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ], dims=[[3, 3], [3, 3]])
    raise ValueError("Dimension must be 2 or 3.")


def embed_two_qutrit_gate(gate: qt.Qobj, n_qutrits: int, qutrit_a: int, qutrit_b: int) -> qt.Qobj:
    """
    Embed a two-qutrit gate into an n-qutrit system using QuTiP.
    
    The gate acts on qutrits at positions `qutrit_a` and `qutrit_b`,
    which can be non-adjacent. All other qutrits are unaffected.
    
    Parameters
    ----------
    gate : Qobj
        The two-qutrit gate with dims [[3,3], [3,3]].
    n_qutrits : int
        Total number of qutrits in the system.
    qutrit_a, qutrit_b : int
        Indices of the qutrits the gate acts on (0-indexed).
        `qutrit_a` is the "first" qutrit of the gate.
    
    Returns
    -------
    Qobj
        The gate acting on the full n-qutrit Hilbert space.
    """
    if qutrit_a == qutrit_b:
        raise ValueError("qutrit_a and qutrit_b must be different")
    if not (0 <= qutrit_a < n_qutrits and 0 <= qutrit_b < n_qutrits):
        raise ValueError(f"Qutrit indices must be in [0, {n_qutrits})")
    
    # Build the operator by summing over projector contributions
    # Gate = Σ_{ijkl} G_{kl,ij} |k⟩⟨i| ⊗ |l⟩⟨j|
    # We embed this into the full space with identities on other qutrits
    
    d = 3
    dims = [[d] * n_qutrits, [d] * n_qutrits]
    total_dim = d ** n_qutrits
    result_matrix = np.zeros((total_dim, total_dim), dtype=complex)
    gate_matrix = gate.full()
    
    for input_index in range(total_dim):
        input_states = _index_to_states(input_index, n_qutrits, d)
        
        in_a = input_states[qutrit_a]
        in_b = input_states[qutrit_b]
        gate_input = in_a * d + in_b
        
        for gate_output in range(d * d):
            amplitude = gate_matrix[gate_output, gate_input]
            if amplitude == 0:
                continue
            
            out_a = gate_output // d
            out_b = gate_output % d
            
            output_states = input_states.copy()
            output_states[qutrit_a] = out_a
            output_states[qutrit_b] = out_b
            
            output_index = _states_to_index(output_states, d)
            result_matrix[output_index, input_index] = amplitude
    
    return qt.Qobj(result_matrix, dims=dims)


def _index_to_states(index: int, n: int, d: int) -> list[int]:
    """Convert basis index to list of subsystem states."""
    states = []
    for _ in range(n):
        states.append(index % d)
        index //= d
    return states[::-1]


def _states_to_index(states: list[int], d: int) -> int:
    """Convert list of subsystem states to basis index."""
    index = 0
    for s in states:
        index = index * d + s
    return index

def sqrt_iswap_qutrit(n_qutrits: int, qutrit_a: int, qutrit_b: int) -> qt.Qobj:
    """
    Sqrt-iSWAP gate on qutrits a and b in an n-qutrit register.
    Qutrits can be non-adjacent.
    """
    return embed_two_qutrit_gate(sqrtISWAP_operator(3, 1), n_qutrits, qutrit_a, qutrit_b)

def Rx_gate(theta:float):
    """Return a logical Rx rotation operator for a single qutrit.

    The returned object is a `qutip.Qobj` usable in tensor constructions.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A `qutip.Qobj` representing the Rx rotation.
    """
    return qt.Qobj([[np.cos(theta/2), -1j*np.sin(theta/2), 0],
                    [-1j*np.sin(theta/2), np.cos(theta/2), 0],
                    [0, 0, 1]])


def Ry_gate(theta:float):
    """Return a logical Ry rotation operator for a single qutrit.

    The returned object is a `qutip.Qobj` usable in tensor constructions.

    Args:
        theta: Rotation angle in radians.

    Returns:
        A `qutip.Qobj` representing the Ry rotation.
    """
    return qt.Qobj([[np.cos(theta/2), -np.sin(theta/2), 0],
                    [np.sin(theta/2), np.cos(theta/2), 0],
                    [0, 0, 1]])

def Rz_gate(phi:float):
    """Return a logical Rz rotation operator for a single qutrit.

    The returned object is a `qutip.Qobj` usable in tensor constructions.

    Args:
        phi: Rotation angle in radians.

    Returns:
        A `qutip.Qobj` representing the Rz rotation.
    """
    return qt.Qobj([[np.exp(-1j*phi/2), 0, 0],
                    [0, np.exp(1j*phi/2), 0],
                    [0, 0, 1]])

def hadamard_operator(dim: int) -> qt.Qobj:
    """Return a logical Hadamard operator for the given dimension.

    Supports `dim==2` (standard Hadamard) and `dim==3` (qutrit analogue used in the
    notebook). The returned object is a `qutip.Qobj` usable in tensor constructions.

    Args:
        dim: The Hilbert-space dimension (2 or 3).

    Returns:
        A `qutip.Qobj` representing the Hadamard-like gate.

    Raises:
        ValueError: if `dim` is not 2 or 3.
    """
    if dim == 2:
        return 1 / np.sqrt(2) * qt.Qobj([[1, 1], [1, -1]])
    if dim == 3:
        return qt.Qobj([
            [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [0, 1, 0],
            [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
        ])
    raise ValueError("Dimension must be 2 or 3.")


def state_swap(dim: int, n: int) -> qt.Qobj:
    """Construct a logical lowering/swap operator for a single d-dimensional qudit.

    The operator swaps |n> <-> |n-1 (mod d)> on the specified index and leaves
    other basis states unchanged (constructed as a dense matrix wrapped in `Qobj`).

    Args:
        dim: Hilbert-space dimension (1 <= dim <= 4 supported by original code).
        n: Index of the state to lower (0 <= n < dim).

    Returns:
        A `qutip.Qobj` representing the lowering operator.

    Raises:
        ValueError: for out-of-range `n` or invalid `dim`.
    """
    if n < 0 or n >= dim:
        raise ValueError("State n must be between 0 and dim-1.")
    if dim < 1 or dim > 4:
        raise ValueError("Dimension must be between 1 and 4.")
    gate = np.eye(dim, dtype=complex)
    target_index = (n - 1) % dim
    gate[target_index, n] = 1
    gate[n, n] = 0
    gate[n, target_index] = 1
    gate[target_index, target_index] = 0
    return qt.Qobj(gate)


def cnot_operator(num_qudits: int, dim: int, control_idx: int, target_idx: int, trigger: int) -> qt.Qobj:
    """Construct a conditional unitary that performs a swap on the target subspace
    when the control qudit is in a given `trigger` state.

    Behaviour summary:
      - If control is in `|trigger>` the target performs a swap between `|0>` and
        `|trigger>` (subspace X); otherwise the target is identity.
      - All non-target, non-control subsystems are identity.

    Args:
        num_qudits: Total number of qudits in the composed system.
        dim: Dimension of each qudit Hilbert space.
        control_idx: Index of the control qudit (0-based).
        target_idx: Index of the target qudit (0-based).
        trigger: The control basis state that triggers the target action.

    Returns:
        A `qutip.Qobj` representing the total controlled operation on the full tensor
        product Hilbert space (dims = [dim]*num_qudits).

    Raises:
        ValueError: for invalid dimensions or indices.
    """
    if dim < 2:
        raise ValueError("Dimension must be at least 2 to have a 0-1 subspace.")
    if control_idx < 0 or control_idx >= num_qudits:
        raise ValueError("Control index out of range.")
    if target_idx < 0 or target_idx >= num_qudits:
        raise ValueError("Target index out of range.")
    if control_idx == target_idx:
        raise ValueError("Control and target indices must be different.")

    X_sub = qt.Qobj(np.zeros((dim, dim), dtype=complex))
    X_sub += qt.basis(dim, 0) * qt.basis(dim, trigger).dag()
    X_sub += qt.basis(dim, trigger) * qt.basis(dim, 0).dag()
    for k in range(dim):
        if k != 0 and k != trigger:
            X_sub += qt.basis(dim, k) * qt.basis(dim, k).dag()

    sys_dims = [dim] * num_qudits
    total_dim = dim ** num_qudits
    U = qt.Qobj(np.zeros((total_dim, total_dim), dtype=complex), dims=[sys_dims, sys_dims])

    for c_state in range(dim):
        if c_state == trigger:
            target_op = X_sub
        else:
            target_op = qt.identity(dim)
        P_c = qt.basis(dim, c_state) * qt.basis(dim, c_state).dag()
        op_list = []
        for i in range(num_qudits):
            if i == control_idx:
                op_list.append(P_c)
            elif i == target_idx:
                op_list.append(target_op)
            else:
                op_list.append(qt.identity(dim))
        U += qt.tensor(op_list)
       
    return U

def cnot_sqrt_iswap_decomposition(control_idx: int, target_idx: int, num_qudits: int, dim: int) -> list[qt.Qobj]:
    """Helper function to return the sequence of gates for sqrtISWAP decomposition of CNOT."""
    gate_list = []
    operations = []
    for i in range(num_qudits):
        if i == control_idx:
            operations.append(Ry_gate(np.pi/2))
        elif i == target_idx:
            operations.append(Rz_gate(np.pi/2))
        else:
            operations.append(qt.qeye(dim))
    gate_list.append(qt.tensor(operations))
    operations.clear()
    for i in range(num_qudits):
        if i == control_idx:
            operations.append(Rz_gate(-np.pi/2))
        elif i == target_idx:
            operations.append(Ry_gate(-np.pi/2))
        else:
            operations.append(qt.qeye(dim))

    gate_list.append(qt.tensor(operations))
    operations.clear() 

    gate_list.append(sqrt_iswap_qutrit(num_qudits, control_idx, target_idx))

    for i in range(num_qudits):
        if i == control_idx:
            operations.append(Rx_gate(np.pi))
        elif i == target_idx:
            operations.append(Rz_gate(-np.pi))
        else:
            operations.append(qt.qeye(dim))

    gate_list.append(qt.tensor(operations))
    operations.clear()
        
    gate_list.append(sqrt_iswap_qutrit(num_qudits, control_idx, target_idx))

    for i in range(num_qudits):
        if i == control_idx:
            operations.append(Rx_gate(np.pi/2))
        elif i == target_idx:
            operations.append(Rz_gate(np.pi/2))
        else:
            operations.append(qt.qeye(dim))

    gate_list.append(qt.tensor(operations))

    return gate_list