"""Experiment runner helper class extracted from notebook.

Contains the main helper functions and a class-based API so notebooks can remain concise.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from quantum_logical.gates import hadamard_operator
import qutip as qt

from quantum_logical.channel import AmplitudeDamping, PhaseDamping
from quantum_logical.trotter import TrotterGroup


def measurement_history_to_int(history: tuple[int]) -> int:
    """Convert a measurement history of the form (measurements) to an integer index."""
    return int("".join(map(str, history)), 2)

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

        # Initialize the list of branches for the first setup, each branch is a tuple of (state_list, probability, [phase_measurements, erasure_measurements])
        branched_current_states:list[qt.Qobj, float, tuple[tuple[int]]] = [(state_list[-1], 1.0, ((),()))]
        # Start applying the circuits
        for setup in setup_circuits:
            circuits, gate_times, measurements, recovery_ops, recovery_times, is_phase = setup


            ############# PERFORM GATES ###############
            num_steps = 0
            # Iterate over all branches
            evolved_branches:list[tuple[list, float, tuple]] = []
            for state, prob, measurement_history in branched_current_states:
                #Run the circuit for the current branch
                evolved_states = ExperimentRunner.run_unitary_circuit(trotterer, circuits, state, gate_times)
                evolved_branches.append((evolved_states[1:], prob, measurement_history))
                # The number of steps should be the same for each branch since they run the same circuit
                num_steps = max(len(evolved_states), num_steps)


            for i in range(num_steps-1):
                weighted_states = sum([prob * (evolved_states[i] if i< len(evolved_states) else evolved_states[-1]) for  evolved_states, prob, _ in evolved_branches])
                state_list.append(weighted_states)

            # Collapse new data to final points now that we have the timeline
            branched_current_states:list[tuple[list[qt.Qobj], float, tuple[tuple[int], tuple[int]]]] = [(state[-1], prob, measurement_history) for state, prob, measurement_history in evolved_branches]

            print("Finished circuit unitary")

            # Check if there are measurements to perform, if not skip to next setup
            if len(measurements) == 0:
                continue

            ############# PERFORM MEASUREMENTS
            measurement_time = 1
            # Create an identity circuit for the full system
            identity_circuit = [qt.tensor([qt.qeye(3)] * num_qudits)]
            idle_gate_times = [measurement_time]
            
            evolved_idle_branches:list[tuple[list, float, tuple]] = []
            num_idle_steps = 0
            
            # Evolve each branch under the identity operator
            for state, prob, measurement_history in branched_current_states:
                idle_states = ExperimentRunner.run_unitary_circuit(trotterer, identity_circuit, state, idle_gate_times)
                evolved_idle_branches.append((idle_states[1:], prob, measurement_history))
                num_idle_steps = max(len(idle_states), num_idle_steps)
                
            # Append the idle timeline to the state_list
            for i in range(num_idle_steps - 1):
                weighted_idle_states = sum([prob * (idle_states[i] if i < len(idle_states) else idle_states[-1]) for idle_states, prob, _ in evolved_idle_branches])
                state_list.append(weighted_idle_states)

            # Update branched_current_states to point to the end of the idle measurement period
            branched_current_states = [(state[-1], prob, measurement_history) for state, prob, measurement_history in evolved_idle_branches]
            
            # Perform measurements on updated branch
            new_branches:list[tuple[list[qt.Qobj], float, tuple[tuple[int], tuple[int]]]] = []
            perform_recovery = False

            # Determine the number of bits measured based on the number of measurement operators (e.g. 2 for single qubit, 4 for two qubits, etc.)
            bits_measured = int(np.log2(len(measurements)))

            # Perform measurements on each branch
            for state, prob, measurement_history in branched_current_states:
                for i, proj in enumerate(measurements):
                    proj_state:qt.Qobj = proj * state * proj.dag()
                    proj_result:float = proj_state.tr().real
                
                    result = ()
                    if bits_measured > 1:
                        result = tuple(map(int,bin(i)[2:].zfill(bits_measured)))  # Convert index to binary and pad with zeros
                    else:
                        result = tuple([i])
                        
                    if proj_result > 1e-12:
                        # Check measurement history and append new measurement
                        if is_phase:
                            new_measurement_history = ((*measurement_history[0], *result), measurement_history[1])
                        else:
                            new_measurement_history = (measurement_history[0], (*measurement_history[1], *result))

                        # If full, perform recovery after
                        if len(new_measurement_history[0]) == 2 or len(new_measurement_history[1]) == 3:
                            perform_recovery = True

                        # Reset ancillae to |0> after measurement by applying the appropriate reset operator and add to list of branches
                        # 1. Start with the first 3 identity operators
                        op_list = [qt.qeye(3)] * 3

                        # 2. Add the reset operators for each result (if result is empty, this safely adds nothing)
                        op_list += [qt.basis(3,0) * qt.basis(3,res).dag() for res in result]

                        # 3. Add the remaining identity operators (if remaining is <= 0, this safely adds nothing)
                        remaining_qudits = num_qudits - 3 - bits_measured
                        if remaining_qudits > 0:
                            op_list += [qt.qeye(3)] * remaining_qudits

                        # 4. Create the final tensor product in one go
                        reset_operator: qt.Qobj = qt.tensor(op_list)
                        new_branches.append((reset_operator * proj_state * reset_operator.dag() / proj_result, prob * proj_result, new_measurement_history))

            branched_current_states = new_branches

            if not perform_recovery:
                continue

            ############# PERFORM RECOVERY #################
            
            # Apply recovery operations to each branch
            recovery_results = []
            for state, prob, measurement_history in branched_current_states:
                
                # Determine which recovery operation to apply based on measurement history
                measurement_index = -1
                if is_phase and len(measurement_history[0]) == 2:
                    measurement_index = measurement_history_to_int(measurement_history[0])
                    new_measurement_history = ((), measurement_history[1])
                elif not is_phase and len(measurement_history[1]) == 3:
                    measurement_index = measurement_history_to_int(measurement_history[1])
                    new_measurement_history = (measurement_history[0], ())
                else:
                    measurement_index = -1
                    new_measurement_history = measurement_history

                # Perform time evolution to apply the recovery operation
                recovery_states = ExperimentRunner.run_unitary_circuit(trotterer, recovery_ops[measurement_index], state, recovery_times[measurement_index])
                recovery_results.append((recovery_states, prob, new_measurement_history))

            # Combine the recovery branches into the main timeline
            max_recovery_steps = max(len(recovery_states) for recovery_states, _, _ in recovery_results)
            for t in range(max_recovery_steps):
                weighted_states = sum([prob * recovery_states[t] if t < len(recovery_states) else prob * recovery_states[-1] for recovery_states, prob, _ in recovery_results])
                state_list.append(weighted_states/ weighted_states.tr())


            # Can consolidate branches with the same measurement history by summing their states weighted by probability and normalizing by total probability to reduce branches.
            consolidated_branches = dict()
            consolidated_probs = dict()
            for recovery_states, prob, new_measurement_history in recovery_results:
                key = new_measurement_history  # Measurement history as key
                
                # Set the states correspondng to the same measurement history to be the last state in the recovery branch, weighted by the probability of that branch.
                consolidated_branches[key] = consolidated_branches.get(key, qt.tensor([qt.qzero(3)] * num_qudits)) + prob * recovery_states[-1]
                # Marginalize over probabilities for branches with the same measurement history
                consolidated_probs[key] = consolidated_probs.get(key, 0) + prob

            branched_current_states = [(consolidated_branches[key] / consolidated_probs[key], consolidated_probs[key], key) for key in consolidated_branches]



        return state_list
    

    