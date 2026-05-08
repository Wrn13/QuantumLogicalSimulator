# Import the compact ExperimentRunner and required libs
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from quantum_logical.channel import AmplitudeDamping
from quantum_logical.experiment_runner import ExperimentRunner
from quantum_logical.qubit_phase import (
    qubit_parallel_phase_circuit,
    qubit_partial_phase_circuit,
)
from quantum_logical.trotter import TrotterGroup


def single_run_n_iterations(n:int, init_rho:qt.Qobj, T1:float, T2:float, single_qubit_time:float, two_qubit_time:float, runner:ExperimentRunner, filename:str):

    setup_5 = [
    qubit_parallel_phase_circuit(single_qubit_time, two_qubit_time, num_ancillae=2)
    ] * 2*n

    setup_4 = [
    qubit_partial_phase_circuit(single_qubit_time, two_qubit_time, target_state = 0),
    qubit_partial_phase_circuit(single_qubit_time, two_qubit_time, target_state = 1),
    ] * n

    trotter_4 = runner.generate_trotterizer(0.03, T1, T2, 2, 4)
    trotter_5 = runner.generate_trotterizer(0.03, T1, T2, 2, 5)

    ancilla_zero = qt.ket2dm(qt.basis(2, 0))
    state_list_4 = runner.experiment_partial_run(trotter_4, qt.tensor(init_rho, ancilla_zero), setup_4)
    state_list_5 = runner.experiment_partial_run(trotter_5, qt.tensor(init_rho, ancilla_zero, ancilla_zero), setup_5)

    ideal_dropped_states_4 = [qt.tensor(init_rho, ancilla_zero)]
    ideal_dropped_states_5 = [qt.tensor(init_rho, ancilla_zero, ancilla_zero)]
    ideal_dropped_states_4 = runner.run_unitary_circuit(trotter_4, [qt.tensor(*[qt.qeye(2)]*4)], state_list_4[0], [37 * single_qubit_time])[:-1]
    ideal_dropped_states_5 = runner.run_unitary_circuit(trotter_5, [qt.tensor(*[qt.qeye(2)]*5)], state_list_5[0], [19 * single_qubit_time])[:-1]
    ideal_dropped_states_4.append(state_list_4[37])
    ideal_dropped_states_5.append(state_list_5[19])
    for i in range(1,n):
        ideal_dropped_states_4 += runner.run_unitary_circuit(trotter_4, [qt.tensor(*[qt.qeye(2)]*4)], state_list_4[37 * i], [37 * single_qubit_time])[1:-1]
        ideal_dropped_states_4.append(state_list_4[37*(i+1)])
    for i in range(1,2*n):
        ideal_dropped_states_5 += runner.run_unitary_circuit(trotter_5, [qt.tensor(*[qt.qeye(2)]*5)], state_list_5[19 * i], [19 * single_qubit_time])[1:-1]
        ideal_dropped_states_5.append(state_list_5[19*(i+1)])
    
    no_correction = runner.run_unitary_circuit(trotter_4, [qt.tensor(*[qt.qeye(2)]*4)], qt.tensor(init_rho, ancilla_zero), [len(ideal_dropped_states_4) * single_qubit_time]) [:-1]  

    np.savez(filename, no_correction, ideal_dropped_states_4, ideal_dropped_states_5)


def generate_haar_random_logical_state(zero_logical, one_logical):
    coeffs = np.random.rand(2) + 1j * np.random.rand(2)
    coeffs /= np.linalg.norm(coeffs)
    print(coeffs)
    return coeffs[0] * zero_logical + coeffs[1] * one_logical


def main():
    single_qubit_time = 0.03 # 30 nanosec
    two_qubit_time = 0.102 # 102 nanosec


    # Define starting states
    plus_state = qt.gates.snot() * qt.basis(2,0)
    minus_state = qt.gates.snot() * qt.basis(2,1)
    zero_logical = qt.tensor(plus_state, plus_state, plus_state)
    one_logical = qt.tensor(minus_state, minus_state, minus_state)
    plus_logical = 1/np.sqrt(2) * (zero_logical + one_logical)
    i_logical = 1/np.sqrt(2) * (zero_logical + 1j * one_logical)
    minus_i_logical = 1/np.sqrt(2) * (zero_logical - 1j * one_logical)
    minus_logical = 1/np.sqrt(2) * (zero_logical - one_logical)
    
    runner = ExperimentRunner()

    T1_list = [280, 280, 100]
    T2_list = [100, 330, 330]

    initial_states = [zero_logical, one_logical, plus_logical, minus_logical, i_logical, minus_i_logical]
    
    initial_state_names = [r"$\ket{0}_L$", r"$\ket{1}_L$", r"$\ket{+}_L$", r"$\ket{-}_L$", r"$\ket{i}_L$", r"$\ket{-i}_L$" ]

    print("Running orthogonal states:")
    for T1,T2 in zip(T1_list, T2_list):
        for initial_state, initial_state_name in zip(initial_states, initial_state_names):
            # Create the initial density matrices
            initial_rho_4q = qt.ket2dm(qt.tensor(initial_state, qt.basis(2,0)))
            initial_rho_5q = qt.ket2dm(qt.tensor(initial_state, qt.basis(2,0), qt.basis(2,0)))

            print(f"Running T1 = {T1}, T2 = {T2} for starting state {initial_state_name}")

    num_initial_states = 16
    print("Running Haar random states:")
if __name__ == "__main__":
    main()




