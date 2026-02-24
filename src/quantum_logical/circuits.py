import numpy as np
import qutip as qt

from quantum_logical.gates import (
    cnot_operator,
    cnot_sqrt_iswap_decomposition,
    state_swap
)


def serial_erasure_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 3):
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
    match num_ancillae:
        case 1:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i)) * qt.tensor(qt.basis(3, i)).dag()) for i in [0, 1]]
        case 2:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j)) * qt.tensor(qt.basis(3, i), qt.basis(3, j)).dag()) for i in [0, 1] for j in [0, 1]]
        case 3:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag()) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
        case _:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag(), *([qt.qeye(3)] * (num_ancillae-3)))  for i in [0, 1] for j in [0, 1] for k in [0, 1]]

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
    return circuit, gate_time, measurements, recovery_ops, recovery_times, False
    
def parallel_erasure_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 3):
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
    match num_ancillae:
        case 1:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i)) * qt.tensor(qt.basis(3, i)).dag()) for i in [0, 1]]
        case 2:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j)) * qt.tensor(qt.basis(3, i), qt.basis(3, j)).dag()) for i in [0, 1] for j in [0, 1]]
        case 3:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag()) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
        case _:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag(), *([qt.qeye(3)] * (num_ancillae-3)))  for i in [0, 1] for j in [0, 1] for k in [0, 1]]
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
    return circuit, gate_time, measurements, recovery_ops, recovery_times, False

def serial_erasure_SqrtISWAP_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 3):
    circuit = []
    gate_time = []

    # Define gates to use in operation
    # CNOT between state qubits and ancillas
    cnot1_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=0, target_idx=3)
    cnot2_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=1, target_idx=4)
    cnot3_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=2, target_idx=5)
    cnot4 = cnot_operator(num_qudits=6, dim=3, control_idx=0, target_idx=2, trigger=2)
    cnot5 = cnot_operator(num_qudits=6, dim=3, control_idx=0, target_idx=1, trigger=2)
    cnot6 = cnot_operator(num_qudits=6, dim=3, control_idx=1, target_idx=0, trigger=2)
    cnot7 = cnot_operator(num_qudits=6, dim=3, control_idx=1, target_idx=2, trigger=2)
    cnot8 = cnot_operator(num_qudits=6, dim=3, control_idx=2, target_idx=0, trigger=2)
    cnot9 = cnot_operator(num_qudits=6, dim=3, control_idx=2, target_idx=1, trigger=2)
    x_gate = qt.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    
    # CNOT Layer
    circuit.extend(cnot1_ops)
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])
    circuit.extend(cnot2_ops)
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])
    circuit.extend(cnot3_ops)
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])

    
    # Measurement operators
    match num_ancillae:
        case 1:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i)) * qt.tensor(qt.basis(3, i)).dag()) for i in [0, 1]]
        case 2:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j)) * qt.tensor(qt.basis(3, i), qt.basis(3, j)).dag()) for i in [0, 1] for j in [0, 1]]
        case 3:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag()) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
        case _:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag(), *([qt.qeye(3)] * (num_ancillae-3)))  for i in [0, 1] for j in [0, 1] for k in [0, 1]]

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
    return circuit, gate_time, measurements, recovery_ops, recovery_times, False
    
def parallel_erasure_SqrtISWAP_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 3):
    circuit = []
    gate_time = []

    # Define gates to use in operation
    # CNOT between state qubits and ancillas
    cnot1_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=0, target_idx=3)
    cnot2_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=1, target_idx=4)
    cnot3_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=2, target_idx=5)
    cnot4_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=0, target_idx=2)
    cnot5_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=0, target_idx=1)
    cnot6_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=1, target_idx=0)
    cnot7_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=1, target_idx=2)
    cnot8_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=2, target_idx=0)
    cnot9_ops = cnot_sqrt_iswap_decomposition(num_qudits=6, dim=3, control_idx=2, target_idx=1)
    x_10 = qt.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    x_21 = state_swap(3,2)

    combined_1_2_3_ops = [cnot1_ops[i] * cnot2_ops[i] * cnot3_ops[i] for i in range(len(cnot1_ops))]
    combined_4_5_ops = [cnot4_ops[i] * cnot5_ops[i] for i in range(len(cnot4_ops))]
    combined_6_7_ops = [cnot6_ops[i] * cnot7_ops[i] for i in range(len(cnot6_ops))]
    combined_8_9_ops = [cnot8_ops[i] * cnot9_ops[i] for i in range(len(cnot8_ops))]
    
    # CNOT Layer
    circuit.extend(combined_1_2_3_ops)
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])

    
    # Measurement operators
    match num_ancillae:
        case 1:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i)) * qt.tensor(qt.basis(3, i)).dag()) for i in [0, 1]]
        case 2:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j)) * qt.tensor(qt.basis(3, i), qt.basis(3, j)).dag()) for i in [0, 1] for j in [0, 1]]
        case 3:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag()) for i in [0, 1] for j in [0, 1] for k in [0, 1]]
        case _:
            measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)) * qt.tensor(qt.basis(3, i), qt.basis(3, j), qt.basis(3, k)).dag(), *([qt.qeye(3)] * (num_ancillae-3)))  for i in [0, 1] for j in [0, 1] for k in [0, 1]]

    #Recovery operations
    r000 = r111 = [qt.tensor(*([qt.qeye(3)] * 6))]
    r001 = [qt.tensor(x_21, qt.qeye(3), x_10, qt.qeye(3), qt.qeye(3), qt.qeye(3)), *cnot4_ops, qt.tensor(x_21, qt.qeye(3), x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3))]
    r010 = [qt.tensor(x_21, x_10, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), *cnot5_ops, qt.tensor(x_21, x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3))]
    r011 = [qt.tensor(x_21, x_10, x_10, qt.qeye(3), qt.qeye(3), qt.qeye(3)), *combined_4_5_ops, qt.tensor(x_21, x_21, x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3))]
    r100 = [qt.tensor(x_10, x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3)), *cnot6_ops, qt.tensor(x_21, x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.qeye(3))]
    r101 = [qt.tensor(x_10, x_21, x_10, qt.qeye(3), qt.qeye(3), qt.qeye(3)), *combined_6_7_ops, qt.tensor(x_21, x_21, x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3))]
    r110 = [qt.tensor(x_10, x_10, x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3)), *combined_8_9_ops, qt.tensor(x_21, x_21, x_21, qt.qeye(3), qt.qeye(3), qt.qeye(3))]
    recovery_ops = [r000, r001, r010, r011, r100, r101, r110, r111]
    recovery_times = [
        [0.0],
        [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, single_qudit_time],
        [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, single_qudit_time],
        [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, single_qudit_time],
        [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, single_qudit_time],
        [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, single_qudit_time],
        [single_qudit_time, single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, single_qudit_time],
        [0.0],
    ]
    return circuit, gate_time, measurements, recovery_ops, recovery_times, False

def partial_erasure_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 3, target_state: int = 0, target_ancilla: int = 0):
    """ An erasure check on a single qutrit.

    Args:
        single_qudit_time (float): Time for single qudit gates
        two_qudit_time (float): Time for two qudit gates
        num_ancillae (int, optional): Number of ancilla qutrits to use. Defaults to 3.
        target_state (int, optional): Index of the target state qubit. Defaults to 0.
        target_ancilla (int, optional): Index of the target ancilla qubit. Defaults to 0.

    Returns:
        _type_: _description_
    """
    circuit = []
    gate_time = []
    N = 3 + num_ancillae
    # Define gates to use in operation
    # CNOT between state qubits and ancillas
    cnot1 = cnot_operator(num_qudits=N, dim=3, control_idx=target_state, target_idx=3+ target_ancilla, trigger=1)
    cnot4 = cnot_operator(num_qudits=N, dim=3, control_idx=0, target_idx=2, trigger=2)
    cnot5 = cnot_operator(num_qudits=N, dim=3, control_idx=0, target_idx=1, trigger=2)
    cnot6 = cnot_operator(num_qudits=N, dim=3, control_idx=1, target_idx=0, trigger=2)
    cnot7 = cnot_operator(num_qudits=N, dim=3, control_idx=1, target_idx=2, trigger=2)
    cnot8 = cnot_operator(num_qudits=N, dim=3, control_idx=2, target_idx=0, trigger=2)
    cnot9 = cnot_operator(num_qudits=N, dim=3, control_idx=2, target_idx=1, trigger=2)
    x_gate = qt.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))

    
    # CNOT Layer
    circuit.append(cnot1)
    gate_time.append(two_qudit_time)
    
    # Measurement operators
    measurements = [qt.tensor(qt.qeye(3), qt.qeye(3), qt.qeye(3), qt.tensor(qt.basis(3, i)) * qt.tensor(qt.basis(3, i)).dag()) for i in [0, 1]]

    #Recovery operations
    r000 = r111 = [qt.tensor(*([qt.qeye(3)] * 4))]
    r001 = [qt.tensor(qt.qeye(3), qt.qeye(3), x_gate, qt.qeye(3)), cnot4]
    r010 = [qt.tensor(qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3)), cnot5]
    r011 = [qt.tensor(qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3)), qt.tensor(qt.qeye(3), qt.qeye(3), x_gate, qt.qeye(3)), cnot5, cnot4]
    r100 = [qt.tensor(x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3)), cnot6]
    r101 = [qt.tensor(x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3)), qt.tensor(qt.qeye(3), qt.qeye(3), x_gate, qt.qeye(3)), cnot6, cnot7]
    r110 = [qt.tensor(x_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3)), qt.tensor(qt.qeye(3), x_gate, qt.qeye(3), qt.qeye(3)), cnot8, cnot9]
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
    return circuit, gate_time, measurements, recovery_ops, recovery_times, False

