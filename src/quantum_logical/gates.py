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
                        [0, 1j/np.sqrt(2), 0, 1/np.sqrt(2)],
                        [0, 0, 0, 1]])
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
            ])
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
        ])
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
            ])
    raise ValueError("Dimension must be 2 or 3.")

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