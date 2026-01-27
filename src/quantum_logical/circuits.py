import numpy as np
from quantum_logical.gates import hadamard_operator, state_swap
import qutip as qt

from quantum_logical.gates import (
    cnot_operator,
)


def serial_phase_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 2):
    circuit = []
    gate_time = []
    N = 3 + num_ancillae

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # CNOT Layer
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=0, target_idx=3, trigger=1))
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=1, target_idx=3, trigger=1))
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=1, target_idx=4, trigger=1))
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=2, target_idx=4, trigger=1))
    gate_time.extend([two_qudit_time] * 4)
    
    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Measurement operators
    phase_measurements = [
        qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()), *[qt.qeye(3)] * (num_ancillae-2))
        for i in range(3) for j in range(3)
    ]

    # Recovery operations
    z_gate = qt.Qobj([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    r00 = r02 = r20 = r22 = r12 = r21 = [qt.tensor([qt.qeye(3)] * N)]
    r01 = [qt.tensor(qt.qeye(3), qt.qeye(3), z_gate, qt.tensor(*[qt.qeye(3)] * num_ancillae))]
    r10 = [qt.tensor(z_gate, qt.tensor(*[qt.qeye(3)] * (N-1)))]
    r11 = [qt.tensor(qt.qeye(3), z_gate, qt.qeye(3), qt.tensor(*[qt.qeye(3)] * num_ancillae))]
    recovery_ops = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    recovery_times = [[0], [single_qudit_time], [0], [single_qudit_time], [single_qudit_time], [0], [0], [0], [0]]
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times


def parallel_phase_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 2):
    circuit = []
    gate_time = []
    N = 3 + num_ancillae

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # CNOT Layer
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=0, target_idx=3, trigger=1)*cnot_operator(num_qudits=N, dim=3, control_idx=1, target_idx=4, trigger=1))
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=1, target_idx=3, trigger=1)*cnot_operator(num_qudits=N, dim=3, control_idx=2, target_idx=4, trigger=1))
    gate_time.extend([two_qudit_time] * 2)
    
    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Measurement operators
    phase_measurements = [
        qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()), *[qt.qeye(3)] * (num_ancillae-2))
        for i in range(3) for j in range(3)
    ]

    # Recovery operations
    z_gate = qt.Qobj([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    r00 = r02 = r20 = r22 = r12 = r21 = [qt.tensor([qt.qeye(3)] * N)]
    r01 = [qt.tensor(qt.qeye(3), qt.qeye(3), z_gate, qt.tensor(*[qt.qeye(3)] * num_ancillae))]
    r10 = [qt.tensor(z_gate, qt.tensor(*[qt.qeye(3)] * (N-1)))]
    r11 = [qt.tensor(qt.qeye(3), z_gate, qt.qeye(3), qt.tensor(*[qt.qeye(3)] * num_ancillae))]
    recovery_ops = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    recovery_times = [[0], [single_qudit_time], [0], [single_qudit_time], [single_qudit_time], [0], [0], [0], [0]]
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times

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
    x_gate = qt.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))

    
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
    
def parallel_erasure_circuit(single_qudit_time: float, two_qudit_time: float):
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
    x_gate = qt.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))

    
    # CNOT Layer
    circuit.append(cnot1*cnot2*cnot3)
    gate_time.append(two_qudit_time)
    
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