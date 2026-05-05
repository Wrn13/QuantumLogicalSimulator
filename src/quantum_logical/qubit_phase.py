import numpy as np
import qutip as qt
from quantum_logical.gates import cnot_operator, hadamard_operator

def qubit_partial_phase_circuit(single_qudit_time: float, two_qudit_time: float, target_state: int = 0):
    circuit = []
    gate_time = []
    N = 4

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(2), hadamard_operator(2), hadamard_operator(2), qt.qeye(2)))
    gate_time.append(single_qudit_time)

    # CNOT Layer
    circuit.append(cnot_operator(num_qudits=N, dim=2, control_idx=target_state, target_idx=3, trigger=1))
    circuit.append(cnot_operator(num_qudits=N, dim=2, control_idx=1+target_state, target_idx=3, trigger=1))
    gate_time.extend([two_qudit_time] * 2)

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(2), hadamard_operator(2), hadamard_operator(2), qt.qeye(2)))
    gate_time.append(single_qudit_time)

    # Measurement operators
    phase_measurements = [
                qt.tensor(*[qt.qeye(2)] * 3, (qt.tensor(qt.basis(2, i)) * (qt.tensor(qt.basis(2, i))).dag()))
                for i in range(2)
    ]
    # Recovery operations
    z_gate = qt.sigmaz()
    r00 = [qt.tensor([qt.qeye(2)] * N)]
    r01 = [qt.tensor(qt.qeye(2), qt.qeye(2), z_gate, qt.qeye(2))]
    r10 = [qt.tensor(z_gate, qt.qeye(2), qt.qeye(2), qt.qeye(2))]
    r11 = [qt.tensor(qt.qeye(2), z_gate, qt.qeye(2), qt.qeye(2))]
    recovery_ops = [r00, r01, r10, r11]
    recovery_times = [[single_qudit_time], [single_qudit_time], [single_qudit_time], [single_qudit_time]]
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times, True


def qubit_parallel_phase_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 2):
    """
    Docstring for qubit_parallel_phase

    :param single_qudit_time: Description
    :type single_qudit_time: float
    :param two_qudit_time: Description
    :type two_qudit_time: float
    :param num_ancillae: Description
    :type num_ancillae: int
    """
    circuit = []
    gate_time = []
    N = 3 + num_ancillae
    H = qt.gates.snot()


    # Hadamard layer
    circuit.append(qt.tensor(H, H, H, *([qt.qeye(2)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # CNOT Layer
    circuit.append(cnot_operator(num_qudits=N, dim=2, control_idx=0, target_idx=3, trigger=1) *cnot_operator(num_qudits=N, dim=2, control_idx=1, target_idx=4, trigger=1))
    circuit.append(cnot_operator(num_qudits=N, dim=2, control_idx=1, target_idx=3, trigger=1) * cnot_operator(num_qudits=N, dim=2, control_idx=2, target_idx=4, trigger=1))
    gate_time.extend([two_qudit_time] * 2)

    # Hadamard layer
    circuit.append(qt.tensor(H, H, H, *([qt.qeye(2)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Measurement operators

    phase_measurements = [
        qt.tensor(*[qt.qeye(2)] * 3, (qt.tensor(qt.basis(2, i), qt.basis(2, j)) * (qt.tensor(qt.basis(2, i), qt.basis(2, j))).dag()))
        for i in range(2) for j in range(2)
    ]

    # Recovery operations
    z_gate = qt.sigmaz()
    r00 = [qt.tensor([qt.qeye(2)] * N)]
    r01 = [qt.tensor(qt.qeye(2), qt.qeye(2), z_gate, qt.qeye(2), qt.qeye(2))]
    r10 = [qt.tensor(z_gate, qt.tensor(*[qt.qeye(2)] * (N-1)))]
    r11 = [qt.tensor(qt.qeye(2), z_gate, qt.qeye(2), qt.qeye(2), qt.qeye(2))]
    recovery_ops = [r00, r01, r10, r11]
    recovery_times = [[single_qudit_time], [single_qudit_time], [single_qudit_time], [single_qudit_time]]
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times, True