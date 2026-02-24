"""Experiment runner helper class extracted from notebook.

Contains the main helper functions and a class-based API so notebooks can remain concise.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import qutip as qt

from quantum_logical.channel import AmplitudeDamping, PhaseDamping
from quantum_logical.trotter import TrotterGroup


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
       
        num_qudits = len(initial_state.dims[0])
        # start state list
        state_list = [initial_state]

        state_list.extend(ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * num_qudits)], state_list[-1], [20]))

        for setup in setup_circuits:
            circuits, gate_times, measurements, recovery_ops, recovery_times, is_phase = setup

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
                if proj_results_after_measurement[i] != 0 and i not in [0] and sum(recovery_times[i]) != 0.0:
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

            # Re-initialize ancillae to |0>
            repetition_corrected_state = qt.tensor(repetition_corrected_state.ptrace([0,1,2]), qt.tensor([qt.basis(3,0)* qt.basis(3,0).dag()] * (num_qudits - 3)))
            # apply a short identity step to let channels act
            state_list.extend(ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * num_qudits)], repetition_corrected_state, [5])[1:])
            # recombine branches
            numerator = sum([proj_results_after_measurement[j] * corrected_states[j] for j in range(len(proj_results_after_measurement))])
            denom = numerator.tr()
            if denom == 0:
                repetition_corrected_state = state_list[-1]
            else:
                repetition_corrected_state = numerator / denom

            # Re-initialize ancillae to |0>
            repetition_corrected_state = qt.tensor(repetition_corrected_state.ptrace([0,1,2]), qt.tensor([qt.basis(3,0)* qt.basis(3,0).dag()] * (num_qudits - 3)))
            # apply a short identity step to let channels act
            state_list.extend(ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * num_qudits)], repetition_corrected_state, [5])[1:])

        return state_list


    @staticmethod
    def experiment_partial_run(trotterer: TrotterGroup, initial_state: qt.Qobj, setup_circuits: List[Tuple[List[qt.Qobj], List[float], List[qt.Qobj], List[List[qt.Qobj]], List[List[float]]]]) -> List[qt.Qobj]:
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

        num_qudits = len(initial_state.dims[0])

        # start state list
        state_list = [initial_state]
        
        # Apply a short identity step to let channels act before the first setup
        state_list += ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * num_qudits)], state_list[-1], [20])[1:]

        # Initialize the list of branches for the first setup, each branch is a tuple of (state_list, probability, [phase_measurements, erasure_measurements])
        branched_current_states:list[qt.Qobj, float, list[list[int]]] = [(state_list[-1], 1.0, [[],[]])]
        # Start applying the circuits
        for setup in setup_circuits:
            circuits, gate_times, measurements, recovery_ops, recovery_times, is_phase = setup

            num_steps = 0
            # Iterate over all branches
            evolved_branches = []
            for state, prob, measurement_history in branched_current_states:
                #Run the circuit for the current branch
                evolved_states = ExperimentRunner.run_unitary_circuit(trotterer, circuits, state, gate_times)
                evolved_branches += (evolved_states[1:], prob, measurement_history)
                num_steps = len(evolved_states)

            # Add the weighted branches to the main timeline
            for i in num_steps:
                weighted_states = sum([prob * evolved_states[i] for prob, evolved_states, _ in evolved_branches])
                state_list.append(weighted_states)


            # Collapse new data to final points now that we have the timeline
            branched_current_states = [(state[-1], prob, measurement_history) for state, prob, measurement_history in evolved_branches]

            
            # Perform measurements on updated branch
            new_branches = []
            perform_recovery = False


            for state, prob, measurement_history in branched_current_states:
                for i, proj in enumerate(measurements):
                    proj_state = proj * state * proj.dag()
                    proj_result = proj_state.tr()

                    if proj_result > 1e-12:
                        # Check measurement history
                        if is_phase:
                            new_measurement_history = [measurement_history[0] + [i], measurement_history[1]]
                        else:
                            new_measurement_history = [measurement_history[0], measurement_history[1] + [i]]

                        if len(new_measurement_history[0]) == 2 or len(new_measurement_history[1]) == 3:
                            perform_recovery = True

                        # Add the new branch with its probability
                        new_branches.append((proj_state / proj_result, prob * proj_result, new_measurement_history))

            branched_current_states = new_branches

            if perform_recovery:
                # Apply recovery operations to each branch
                recovery_results = []
                for state, prob, measurement_history in branched_current_states:
                    
                    # Determine which recovery operation to apply based on measurement history
                    measurement_index = -1
                    if is_phase and len(measurement_history[0]) == 2:
                        measurement_index = int("".join(map(str, measurement_history[0])), 2)
                        new_measurement_history = [[], measurement_history[1]]
                    elif not is_phase and len(measurement_history[1]) == 3:
                        measurement_index = int("".join(map(str, measurement_history[1])), 2)
                        new_measurement_history = [measurement_history[0], []]
                    else:
                        measurement_index = -1
                        new_measurement_history = measurement_history


                    # Perform time evolution to apply the recovery operation
                    if measurement_index != 0 and sum(recovery_times[measurement_index]) != 0.0:
                        recovery_states = ExperimentRunner.run_unitary_circuit(trotterer, recovery_ops[measurement_index], state, recovery_times[measurement_index])
                    else:
                        state_current = state
                        for recovery_op in recovery_ops[measurement_index]:
                            state_current = recovery_op * state_current * recovery_op.dag()
                        recovery_states = [state_current]

                    recovery_results.append((recovery_states, prob, new_measurement_history))

                # Combine the recovery branches into the main timeline
                max_recovery_steps = max(len(recovery_states) for recovery_states, _ in recovery_results)
                for t in range(max_recovery_steps):
                    weighted_states = sum([prob * recovery_states[t] if t < len(recovery_states) else prob * recovery_states[-1] for recovery_states, prob in recovery_results])
                    state_list.append(weighted_states/ weighted_states.tr())

                branched_current_states = [(recovery_states[-1], prob, new_measurement_history) for recovery_states, prob, new_measurement_history in recovery_results]

        return state_list
    