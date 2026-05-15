from enum import Enum
from typing import override

import qutip as qt

from quantum_logical.noisy_gate import NoisyGate
from weylchamber import gate


class Spectator_Types(Enum):
    Qubit_Qubit = 0,
    Qubit_SNAIL = 1,
    SNAIL_Subharmonic = 2,
    Qubit_Subharmonic = 3


class SNAIL_Module():

    num_qubits:int 

    hilbert_space_dim:int

    snail_freq: float

    transmon_frequencies:list[float]

    alpha_q:list[float]

    g3:float

    eta:float

    connectivity:list[tuple[int, int]]
    """
    List of connectivity for the SNAIL module
    """

    lambda_q: list[float]
    """
    List of hybridizations g_sq/Delta
    """

    pump_frequencies:dict[tuple[int, int], list[tuple[float, str, float]]]
    """
    List of all possible drive frequencies to first order, including qubit qubit drives,
    snail qubit drives, the snail subharmonic, and the qubit subharmonics.
    """

    def __init__(
        self,
        num_qubits:int = 2,
        hilbert_space_dim:int = 2,
        snail_freq: float = 5,
        transmon_frequencies:list[float] = None,
        g_sq:list[float] = None,
        alpha_q:list[float] = None,
        g3:float = 0.06,
        eta:float = 1.8,
        connectivity:list[tuple[int, int]] = None
    ):

        self.num_qubits = num_qubits
        self.hilbert_space_dim = hilbert_space_dim
        self.snail_freq = snail_freq
        self.transmon_frequencies = transmon_frequencies
        self.g_sq = g_sq
        self.alpha_q = alpha_q
        self.g3 = g3
        self.eta = eta
         
        self.connectivity = connectivity

        self.generate_hybridization()
        self.generate_pump_frequencies()


    def _generate_pump_frequencies(self):
        pump_freqs = []

        # Qubit Qubit interactions
        for i, j in self.connectivity:
            f_i = self.transmon_frequencies[i]
            f_j = self.transmon_frequencies[j]
            coeff_strength = 3 * self.eta **2 * self.lambda_q[i] * self.lambda_q[j] * self.g3
            pump_freqs[i,j] = (abs(f_i - f_j), Spectator_Types.Qubit_Qubit, coeff_strength)

        # Qubit SNAIL interaction
        for i in range(self.num_qubits):
            f_i = self.transmon_frequencies[i]
            pump_freqs[i,self.num_qubits] = (self.snail_freq - f_i, Spectator_Types.Qubit_SNAIL, 6 * self.eta * self.lambda_q[i] * self.g3)


        # Subharmonics
        # SNAIL Subharmonic
        pump_freqs[self.num_qubits, self.num_qubits] = (self.snail_freq/2, Spectator_Types.SNAIL_Subharmonic, 3 * self.eta **2 * self.g3)

        for i in range(self.num_qubits):
            f_i = self.transmon_frequencies[i]
            pump_freqs[i,i] = (f_i/2, Spectator_Types.Qubit_Subharmonic, 3 * self.eta **2 * self.lambda_q[i]**2 * self.g3)

        self.pump_frequencies = pump_freqs

    def compute_hybridization(self):
        lambda_q = []
        for i in range(self.num_qubits):
            f_i = self.transmon_frequencies[i]
            detuning = abs(f_i - self.snail_freq)
            hybridization_strength = self.g_sq / detuning
            lambda_q.append(hybridization_strength)
        
        self.lambda_q = lambda_q
        

    def compute_gate_detuning(self, qubit_i, qubit_j, qubit_k, qubit_l):
        target_gate_freq = self.pump_frequencies[qubit_i, qubit_j]
        detuned_gate_freq = self.pump_frequencies[qubit_k, qubit_l]

        return abs(target_gate_freq - detuned_gate_freq)
    
    def compute_spectator_operator(self, qubit_i:int, qubit_j:int, type:str):
        """Compute the operator associated  with the given interaction.
        TODO: Find a way to make amplitude damping from SNAIL drives.

        Args:
            qubit_i (int): _description_
            qubit_j (int): _description_
            type (str): _description_

        Returns:
            _type_: _description_
        """

        destroy = qt.destroy(self.hilbert_space_dim)
        match type:
            case Spectator_Types.Qubit_Qubit:
                destroy_i = qt.tensor([destroy if k == qubit_i else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits+1)])
                destroy_j = qt.tensor([destroy if k == qubit_j else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits+1)])
                return destroy_i * destroy_j.dag() + destroy_i.dag() * destroy_j
            case Spectator_Types.Qubit_SNAIL:
                destroy_i = qt.tensor([destroy if k == min(qubit_i, qubit_j) else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits+1)])
                destroy_s = qt.tensor([destroy if k == self.num_qubits else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits+1)])
                return destroy_i * destroy_s.dag() +  destroy_i.dag() * destroy_s
            case Spectator_Types.SNAIL_Subharmonic:
                destroy_s = qt.tensor([destroy if k == self.num_qubits else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits+1)])
                return destroy_s + destroy_s.dag()
            case Spectator_Types.Qubit_Subharmonic:
                destroy_i = qt.tensor([destroy if k == qubit_i else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits+1)])
                return destroy_i + destroy_i.dag()

    def generate_noisy_gate(self, qubit_i:int, qubit_j:int, ideal_gate:qt.Qobj) -> NoisyGate:
        gate_freq, interaction_type, _ = self.pump_frequencies[qubit_i, qubit_j]

        assert interaction_type == Spectator_Types.Qubit_Qubit, "Currently only supports generating noisy gates for qubit-qubit interactions."
        coherent_gate_error = qt.Qobj()
        
        # Construct the noisy gate using the ideal unitary and the error strength
        for idx, vals in self.pump_frequencies.items():
            i = idx[0]
            j = idx[1]
            
            pump_freq = vals[0]
            spectator_type = vals[1]
            coeff = vals[2]


            delta = abs(gate_freq - pump_freq)
            #Target gate interaction, not spectator
            if i == j and i == qubit_i:
                continue

            coherent_gate_error += 2 * coeff / delta * self.compute_spectator_operator(i, j, spectator_type)

            
        
        total_gate = NoisyGate(ideal_gate * coherent_gate_error, self.num_qubits, self.hilbert_space_dim, None, )


        return NoisyGate(ideal_unitary, self.num_qubits, self.hilbert_space_dim, kraus_operators, duration, (qubit_i, qubit_j))
    


# TODO: Implement code for 2 module systems.
class Two_SNAIL_Module(SNAIL_Module):
    snail_freq_2:float

    def __init__(self, num_qubits = 2, hilbert_space_dim = 2, snail_freq = 5, snail_freq_2 = 6, transmon_frequencies = None, lambda_q = None, alpha_q = None, g3 = 0.06, eta = 1.8, connectivity = None):
        self.snail_freq_2 = snail_freq_2
        super().__init__(num_qubits, hilbert_space_dim, snail_freq, transmon_frequencies, lambda_q, alpha_q, g3, eta, connectivity)

        self.pump_frequencies[self.num_qubits+1, self.num_qubits+1] = (self.snail_freq_2/2, 3 * self.eta **2 * self.lambda_q[0] * self.lambda_q)

    @override
    def _generate_pump_frequencies(self):
        pump_freqs = []

        # Qubit Qubit interactions
        for i, j in self.connectivity:
            f_i = self.transmon_frequencies[i]
            f_j = self.transmon_frequencies[j]
            coeff_strength = 3 * self.eta **2 * self.lambda_q[i] * self.lambda_q[j] * self.g3
            pump_freqs[i,j] = (abs(f_i - f_j), "qubit-qubit", coeff_strength)
            pump_freqs[j,i] = (abs(f_i - f_j), "qubit-qubit", coeff_strength)

        # Qubit SNAIL interaction
        for i in range(self.num_qubits):
            f_i = self.transmon_frequencies[i]
            pump_freqs[i,self.num_qubits] = (self.snail_freq - f_i, "qubit-SNAIL", 6 * self.eta * self.lambda_q[i] * self.g3)


        # Subharmonics
        # SNAIL Subharmonic
        pump_freqs[self.num_qubits, self.num_qubits] = (self.snail_freq/2, "SNAIL subharmonic", 3 * self.eta **2 * self.g3)

        for i in range(self.num_qubits):
            f_i = self.transmon_frequencies[i]
            pump_freqs[i,i] = (f_i/2, "qubit subharmonic", 3 * self.eta **2 * self.lambda_q[i]**2 * self.g3)

        self.pump_frequencies = pump_freqs

    @override
    def compute_hybridization(self):
        lambda_q = []
        for i in range(self.num_qubits):
            f_i = self.transmon_frequencies[i]
            detuning = abs(f_i - self.snail_freq)
            hybridization_strength = self.g_sq / detuning
            lambda_q.append(hybridization_strength)
        
        self.lambda_q = lambda_q
        
    @override
    def compute_gate_detuning(self, qubit_i, qubit_j, qubit_k, qubit_l):
        target_gate_freq = self.pump_frequencies[qubit_i, qubit_j]
        detuned_gate_freq = self.pump_frequencies[qubit_k, qubit_l]

        return abs(target_gate_freq - detuned_gate_freq)
    
