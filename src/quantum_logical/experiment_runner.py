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

        print("Num qudits:", num_qudits)
        # start state list
        state_list = [initial_state]

        state_list.extend(ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * num_qudits)], state_list[-1], [20]))

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
                if proj_results_after_measurement[i] != 0 and i not in [0, len(recovery_ops) - 1] and sum(recovery_times[i]) != 0.0:
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
        erasure_registers = []
        erasure_states = []
        phase_registers = []
        phase_states = []
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

            # run identity step to simulate measurement time
            part_states = ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * num_qudits)], state_list[-1], [.3])
            # avoid duplicating the current last state
            state_list.extend(part_states[1:])

            # Save results of measurement to appropriate register
            if is_phase:
                phase_registers.extend(proj_results_after_measurement)
                phase_states.extend(proj_states_after_measurement)
                proj_results_after_measurement = []
                proj_states_after_measurement = []
            elif len(proj_results_after_measurement) > 0:
                print("GOT HERE")
                erasure_registers.extend(proj_results_after_measurement)
                print("erasure registers:", erasure_registers)
                erasure_states.extend(proj_states_after_measurement)
                proj_results_after_measurement = []
                proj_states_after_measurement = []

            # No full register so go next iteration after resetting ancillae to |0>
            if is_phase and len(phase_registers) < 4 or (not is_phase and len(erasure_registers) < 6):
                state_list.append(qt.tensor(state_list[-1].ptrace([0,1,2]), qt.tensor([qt.basis(3,0)* qt.basis(3,0).dag()] * (num_qudits - 3))))
                continue

            corrected_states = []
            
            if len(phase_registers) == 4:
                #Transform probabilities to correspond to the 4 branches of the phase error correction circuit
                temp_phase_registers = [0 for _ in range(4)]
                temp_phase_registers[0] = phase_registers[0] * phase_registers[2]
                temp_phase_registers[1] = phase_registers[0] * phase_registers[3]
                temp_phase_registers[2] = phase_registers[1] * phase_registers[2]
                temp_phase_registers[3] = phase_registers[1] * phase_registers[3]
                phase_registers = temp_phase_registers
                # apply corrections per branch
                for i in range(len(recovery_ops)):
                    if phase_registers[i] != 0 and i not in [0, len(recovery_ops) - 1] and sum(recovery_times[i]) != 0.0:
                        corrected_states.append(ExperimentRunner.run_unitary_circuit(trotterer, recovery_ops[i], state_list[-1], recovery_times[i])[-1])
                    else:
                        state_current = state_list[-1]
                        for j in range(len(recovery_ops[i])):
                            state_current = recovery_ops[i][j] * state_current * (recovery_ops[i][j]).dag()
                        corrected_states.append(state_current)

                print("phase_registers:", phase_registers)
                numerator = sum([phase_registers[j] * corrected_states[j] for j in range(len(phase_registers))])
                phase_registers.clear()
            elif len(erasure_registers) == 6:
                temp_erasure_registers = [0 for _ in range(8)]
                temp_erasure_registers[0] = erasure_registers[0] * erasure_registers[2] * erasure_registers[4]
                temp_erasure_registers[1] = erasure_registers[0] * erasure_registers[2] * erasure_registers[5]
                temp_erasure_registers[2] = erasure_registers[0] * erasure_registers[3] * erasure_registers[4]
                temp_erasure_registers[3] = erasure_registers[0] * erasure_registers[3] * erasure_registers[5]
                temp_erasure_registers[4] = erasure_registers[1] * erasure_registers[2] * erasure_registers[4]
                temp_erasure_registers[5] = erasure_registers[1] * erasure_registers[2] * erasure_registers[5]
                temp_erasure_registers[6] = erasure_registers[1] * erasure_registers[3] * erasure_registers[4]
                temp_erasure_registers[7] = erasure_registers[1] * erasure_registers[3] * erasure_registers[5]
                erasure_registers = temp_erasure_registers
                # apply corrections per branch
                for i in range(len(recovery_ops)):
                    if erasure_registers[i] != 0 and i not in [0, len(recovery_ops) - 1] and sum(recovery_times[i]) != 0.0:
                        corrected_states.append(ExperimentRunner.run_unitary_circuit(trotterer, recovery_ops[i], state_list[-1], recovery_times[i])[-1])
                    else:
                        state_current = state_list[-1]
                        for j in range(len(recovery_ops[i])):
                            state_current = recovery_ops[i][j] * state_current * (recovery_ops[i][j]).dag()
                        corrected_states.append(state_current)

                print("erasure registers:", erasure_registers)
                numerator = sum([erasure_registers[j] * corrected_states[j] for j in range(len(erasure_registers))])
                erasure_registers.clear()
            
            denom = numerator.tr()
            if denom == 0:
                repetition_corrected_state = state_list[-1]
            else:
                repetition_corrected_state = numerator / denom

            # Re-initialize ancillae to |0>
            repetition_corrected_state = qt.tensor(repetition_corrected_state.ptrace([0,1,2]), qt.tensor([qt.basis(3,0)* qt.basis(3,0).dag()] * (num_qudits - 3)))
            # apply a short identity step to let channels act
            state_list.extend(ExperimentRunner.run_unitary_circuit(trotterer, [qt.tensor([qt.qeye(3)] * num_qudits)], repetition_corrected_state, [.5])[1:])

        return state_list
    