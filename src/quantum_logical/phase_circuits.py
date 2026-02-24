import qutip as qt
from quantum_logical.gates import cnot_operator, cnot_sqrt_iswap_decomposition, hadamard_operator, state_swap


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
    match num_ancillae:
        case 1:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i)) * (qt.tensor(qt.basis(3, i))).dag()))
                for i in range(3)
            ]
        case 2:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()))
                for i in range(3) for j in range(3)
            ]
        case _:
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
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times, True


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
    match num_ancillae:
        case 1:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i)) * (qt.tensor(qt.basis(3, i))).dag()))
                for i in range(2)
            ]
        case 2:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()))
                for i in range(2) for j in range(2)
            ]
        case _:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()), *[qt.qeye(3)] * (num_ancillae-2))
                for i in range(2) for j in range(2)
            ]

    # Recovery operations
    z_gate = qt.Qobj([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    r00 = [qt.tensor([qt.qeye(3)] * N)]
    r01 = [qt.tensor(qt.qeye(3), qt.qeye(3), z_gate, qt.tensor(*[qt.qeye(3)] * num_ancillae))]
    r10 = [qt.tensor(z_gate, qt.tensor(*[qt.qeye(3)] * (N-1)))]
    r11 = [qt.tensor(qt.qeye(3), z_gate, qt.qeye(3), qt.tensor(*[qt.qeye(3)] * num_ancillae))]
    recovery_ops = [r00, r01, r10, r11]
    recovery_times = [[0], [single_qudit_time], [single_qudit_time], [single_qudit_time]]
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times, True


def serial_phase_SqrtISWAP_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 2):
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
    circuit.extend(cnot_sqrt_iswap_decomposition(control_idx=0, target_idx=3, num_qudits=N, dim=3))
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])
    circuit.extend(cnot_sqrt_iswap_decomposition(control_idx=1, target_idx=3, num_qudits=N, dim=3))
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])
    circuit.extend(cnot_sqrt_iswap_decomposition(control_idx=1, target_idx=4, num_qudits=N, dim=3))
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])
    circuit.extend(cnot_sqrt_iswap_decomposition(control_idx=2, target_idx=4, num_qudits=N, dim=3))
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])

    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Measurement operators
    match num_ancillae:
        case 1:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i)) * (qt.tensor(qt.basis(3, i))).dag()))
                for i in range(3)
            ]
        case 2:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()))
                for i in range(3) for j in range(3)
            ]
        case _:
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
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times, True


def parallel_phase_SqrtISWAP_circuit(single_qudit_time: float, two_qudit_time: float, num_ancillae: int = 2):
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
    # Apply target 3 and target 4 in parallel
    cnot_1_ops = cnot_sqrt_iswap_decomposition(control_idx=0, target_idx=3, num_qudits=N, dim=3)
    cnot_3_ops = cnot_sqrt_iswap_decomposition(control_idx=1, target_idx=4, num_qudits=N, dim=3)
    for i in range(len(cnot_1_ops)):
        circuit.append(cnot_1_ops[i] * cnot_3_ops[i])
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])

    cnot_2_ops = cnot_sqrt_iswap_decomposition(control_idx=1, target_idx=3, num_qudits=N, dim=3)
    cnot_4_ops = cnot_sqrt_iswap_decomposition(control_idx=2, target_idx=4, num_qudits=N, dim=3)
    for i in range(len(cnot_2_ops)):
        circuit.append(cnot_2_ops[i] * cnot_4_ops[i])
    gate_time.extend([single_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time, two_qudit_time, single_qudit_time])

    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), *([qt.qeye(3)] * num_ancillae)))
    gate_time.append(single_qudit_time)

    # Measurement operators
    match num_ancillae:
        case 1:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i)) * (qt.tensor(qt.basis(3, i))).dag()))
                for i in range(3)
            ]
        case 2:
            phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i), qt.basis(3, j)) * (qt.tensor(qt.basis(3, i), qt.basis(3, j))).dag()))
                for i in range(3) for j in range(3)
            ]
        case _:
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
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times, True


def partial_phase_circuit(single_qudit_time: float, two_qudit_time: float, target_state: int = 0):
    circuit = []
    gate_time = []
    N = 4

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), qt.qeye(3)))
    gate_time.append(single_qudit_time)

    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), qt.qeye(3)))
    gate_time.append(single_qudit_time)

    # CNOT Layer
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=target_state, target_idx=3, trigger=1))
    circuit.append(cnot_operator(num_qudits=N, dim=3, control_idx=1+target_state, target_idx=3, trigger=1))
    gate_time.extend([two_qudit_time] * 2)

    # State Swap Layer
    circuit.append(qt.tensor(state_swap(3, 2), state_swap(3, 2), state_swap(3, 2), qt.qeye(3)))
    gate_time.append(single_qudit_time)

    # Hadamard layer
    circuit.append(qt.tensor(hadamard_operator(3), hadamard_operator(3), hadamard_operator(3), qt.qeye(3)))
    gate_time.append(single_qudit_time)

    # Measurement operators
    phase_measurements = [
                qt.tensor(*[qt.qeye(3)] * 3, (qt.tensor(qt.basis(3, i)) * (qt.tensor(qt.basis(3, i))).dag()))
                for i in range(2)
    ]
    # Recovery operations
    z_gate = qt.Qobj([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    r00 = [qt.tensor([qt.qeye(3)] * N)]
    r01 = [qt.tensor(qt.qeye(3), qt.qeye(3), z_gate, qt.qeye(3))]
    r10 = [qt.tensor(z_gate, qt.qeye(3), qt.qeye(3), qt.qeye(3))]
    r11 = [qt.tensor(qt.qeye(3), z_gate, qt.qeye(3), qt.qeye(3))]
    recovery_ops = [r00, r01, r10, r11]
    recovery_times = [[0], [single_qudit_time], [single_qudit_time], [single_qudit_time]]
    return circuit, gate_time, phase_measurements, recovery_ops, recovery_times, True