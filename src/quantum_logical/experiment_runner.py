"""Experiment runner helper class extracted from notebook.

Contains the main helper functions and a class-based API so notebooks can remain concise.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from quantum_logical.channel import AmplitudeDamping, PhaseDamping
from quantum_logical.trotter import TrotterGroup


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


class ExperimentRunner:
    @staticmethod
    def generate_trotterizer(trotter_dt: float, T1: float, T2: float, dim: int, num_qubits: int) -> TrotterGroup:
        """Create a configured `TrotterGroup` with amplitude and phase damping channels.

        Args:
            trotter_dt: Time-step for Trotterization.
            T1: Relaxation time constant for amplitude damping.
            T2: Dephasing time constant for phase damping.
            dim: Hilbert-space dimension for each qudit.
            num_qubits: Number of qudits in the system.

        Returns:
            A `TrotterGroup` instance configured with amplitude and phase damping channels.

        Raises:
            Any exception raised constructing the underlying channel or TrotterGroup.
        """
        amp_damp = AmplitudeDamping(T1=T1, hilbert_space_dim=dim, num_qubits=num_qubits)
        phase_damp = PhaseDamping(T1=T1, T2=T2, hilbert_space_dim=dim, num_qubits=num_qubits)
        trotterizer = TrotterGroup([amp_damp, phase_damp], trotter_dt)
        return trotterizer

    @staticmethod
    def run_unitary_circuit(trotterer: TrotterGroup, unitary_circuit: List[qt.Qobj], initial_state: qt.Qobj, gate_times: List[float]) -> List[qt.Qobj]:
        """Trotterize and apply a sequence of unitaries to an initial state.

        Each gate in `unitary_circuit` is applied with the associated duration in
        `gate_times` using the provided `trotterer.apply(...)` method. The function
        returns the list of states produced at each sub-step of the Trotter application
        (first element is the `initial_state`).

        Args:
            trotterer: A `TrotterGroup` instance exposing `apply(state, discrete_unitary, duration)`.
            unitary_circuit: Ordered list of `qutip.Qobj` unitaries to apply.
            initial_state: The density matrix / state to begin from (as `qutip.Qobj`).
            gate_times: Matching list of durations for each gate.

        Returns:
            A list of `qutip.Qobj` states containing the initial and intermediate states.
        """
        state_list = [initial_state]
        for gate, gate_time in zip(unitary_circuit, gate_times):
            state_list.extend(trotterer.apply(state=state_list[-1], discrete_unitary=gate, duration=gate_time))
        return state_list



    @staticmethod
    def experiment_run(trotterer: TrotterGroup, initial_state: qt.Qobj, setup_circuits: List[Tuple[List[qt.Qobj], List[float], List[qt.Qobj], List[List[qt.Qobj]], List[List[float]]]]) -> List[qt.Qobj]:
        """Execute one or more circuit setups sequentially and plot the results.

        This routine accepts a list of setups (each setup is a tuple returned by the
        per-experiment setup helpers such as `serial_phase_circuit`). For each setup
        the function:
          1. Runs the unitary circuit with `run_unitary_circuit` starting from the
             most recent state.
          2. Performs the configured measurements on the resulting state.
          3. Applies correction / recovery branches and recombines them.
          4. Allows a short identity interval to let channels act, then proceeds to
             the next setup.

        Args:
            trotterer: `TrotterGroup` used for trotterized evolution.
            initial_state: Initial density matrix / state (`qutip.Qobj`).
            single_qudit_time: Time used by single-qudit operations (passed to setups).
            two_qudit_time: Time used by two-qudit operations (passed to setups).
            setup_circuits: List of setup tuples of the form
                (circuit, gate_time, measurements, recovery_ops, recovery_times).

        Returns:
            A list of `qutip.Qobj` states produced at each step of the full experiment.
        """
        # start state list
        state_list = [initial_state]

        for setup in setup_circuits:
            circuits, gate_times, measurements, recovery_ops, recovery_times = setup

            # run circuit starting from last state
            part_states = ExperimentRunner.run_unitary_circuit(trotterer, circuits, state_list[-1], gate_times)
            # avoid duplicating the current last state
            state_list.extend(part_states[1:])

            # measurement on the most recent state
            proj_results_after_measurement = [(state_list[-1] * proj).tr() for proj in measurements]
            proj_states_after_measurement = [(proj * state_list[-1] * proj.dag()) for proj in measurements]

            # apply corrections per branch
            corrected_states = []
            for i in range(len(recovery_ops)):
                if proj_results_after_measurement[i] != 0 and i not in [0, len(recovery_ops) - 1] and sum(recovery_ops[i]) != 0.0:
                    corrected_states.append(ExperimentRunner.run_unitary_circuit(trotterer, recovery_ops[i], proj_states_after_measurement[i], recovery_times[i])[-1])
                else:
                    state_current = proj_states_after_measurement[i]
                    for j in range(len(recovery_ops[i])):
                        state_current = recovery_ops[i][j] * state_current * (recovery_ops[i][j]).dag()
                    corrected_states.append(state_current)

            # recombine branches
            numerator = sum([proj_results_after_measurement[j] * corrected_states[j] for j in range(len(proj_results_after_measurement))])
            denom = numerator.tr()
            if denom == 0:
                repetition_corrected_state = state_list[-1]
            else:
                repetition_corrected_state = numerator / denom

            # apply a short identity step to let channels act
            state_list.extend(ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * 6)], repetition_corrected_state, [5])[1:])

        return state_list

    # Example setup circuits (moved from notebook for convenience)
    @staticmethod
    def serial_phase_circuit(single_qudit_time: float, two_qudit_time: float):
        circuit = []
        gate_time = []

        # Identity operation
        circuit.append(qt.tensor(*[qt.qeye(3)] * 5))
        gate_time.append(20)

        # Hadamard layer
        circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), qt.qeye(3), qt.qeye(3)))
        gate_time.append(single_qudit_time)

        # State Swap Layer
        circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), qt.qeye(3), qt.qeye(3)))
        gate_time.append(single_qudit_time)

        # CNOT Layer
        circuit.append(cnot_operator(num_qudits=5, dim=3, control_idx=0, target_idx=3, trigger=1))
        circuit.append(cnot_operator(num_qudits=5, dim=3, control_idx=1, target_idx=3, trigger=1))
        circuit.append(cnot_operator(num_qudits=5, dim=3, control_idx=1, target_idx=4, trigger=1))
        circuit.append(cnot_operator(num_qudits=5, dim=3, control_idx=2, target_idx=4, trigger=1))
        gate_time.extend([two_qudit_time] * 4)
        
        # State Swap Layer
        circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), qt.qeye(3), qt.qeye(3)))
        gate_time.append(single_qudit_time)

        # Hadamard layer
        circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), qt.qeye(3), qt.qeye(3)))
        gate_time.append(single_qudit_time)

        # Measurement operators
        phase_measurements = [
            qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()))
            for i in range(3) for j in range(3)
        ]

        # Recovery operations
        z_gate = qt.Qobj([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        r00 = r02 = r20 = r22 = r12 = r21 = [qt.tensor([qt.qeye(3)] * 5)]
        r01 = [qt.tensor(qt.qeye(3), qt.qeye(3), z_gate, qt.tensor([qt.qeye(3)] * 2))]
        r10 = [qt.tensor(z_gate, qt.tensor([qt.qeye(3)] * 4))]
        r11 = [qt.tensor(qt.qeye(3), z_gate, qt.qeye(3), qt.tensor([qt.qeye(3)] * 2))]
        recovery_ops = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
        recovery_times = [[0], [single_qudit_time], [0], [single_qudit_time], [single_qudit_time], [0], [0], [0], [0]]
        return circuit, gate_time, phase_measurements, recovery_ops, recovery_times

    @staticmethod
    def serial_erasure_circuit(single_qudit_time: float, two_qudit_time: float):
        circuit = []
        gate_time = []

        # Define gates to use in operation
        # CNOT between state qubits and ancillas
        cnot1 = cnot_operator(num_qudits=6, dim=3, control_idx=0, target_idx=3, trigger=1)
        cnot2 = cnot_operator(num_qudits=6, dim=3, control_idx=1, target_idx=4, trigger=1)
        cnot3 = cnot_operator(num_qudits=6, dim=3, control_idx=2, target_idx=5, trigger=1)
        cnot4 = cnot_operator(num_qudits=6, dim=3, control_idx=0, target_idx=2, trigger=2)
        cnot5 = cnot_operator(num_qudits=6, dim=3, control_idx=0, target_idx=1, trigger=2)
        cnot6 = cnot_operator(num_qudits=6, dim=3, control_idx=1, target_idx=0, trigger=2)
        cnot7 = cnot_operator(num_qudits=6, dim=3, control_idx=1, target_idx=2, trigger=2)
        cnot8 = cnot_operator(num_qudits=6, dim=3, control_idx=2, target_idx=0, trigger=2)
        cnot9 = cnot_operator(num_qudits=6, dim=3, control_idx=2, target_idx=1, trigger=2)
        hadamard_layer = qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), qt.qeye(3), qt.qeye(3), qt.qeye(3))
        x_gate = qt.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))

        # Identity operation for 5 time units
        circuit.append(qt.tensor(*[qt.qeye(3)] * 6))
        gate_time.append(5)

        
        # CNOT Layer
        circuit.append(cnot1)
        circuit.append(cnot2)
        circuit.append(cnot3)
        gate_time.extend([two_qudit_time] * 3)
        
        # Measurement operators
        measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag()) for i in [0, 1] for j in [0, 1] for k in [0, 1]]

        #Recovery operations
        r000 = r111 = [qt.tensor(*([qt.qeye(3)] * 6))]
        r001 = [qt.tensor(qt.qeye(3), qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3)), cnot4]
        r010 = [qt.tensor(qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), cnot5]
        r011 = [qt.tensor(qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), qt.tensor(qt.qeye(3), qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3)), cnot5, cnot4]
        r100 = [qt.tensor(x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), cnot6]
        r101 = [qt.tensor(x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), qt.tensor(qt.qeye(3), qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3)), cnot6, cnot7]
        r110 = [qt.tensor(x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), qt.tensor(qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), cnot8, cnot9]
        recovery_ops = [r000, r001, r010, r011, r100, r101, r110, r111]
        recovery_times = [
            [0.0],
            [single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time],
            [single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time],
            [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, two_qudit_time, single_qudit_time],
            [single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time],
            [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, two_qudit_time, single_qudit_time],
            [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, two_qudit_time, single_qudit_time],
            [0.0],
        ]
        return circuit, gate_time, measurements, recovery_ops, recovery_times
