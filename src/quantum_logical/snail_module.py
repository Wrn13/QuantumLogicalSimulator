from typing import override


class SNAIL_Module():

    num_qubits:int 

    hilbert_space_dim:int

    snail_freq: float

    transmon_frequencies:list[float]

    alpha_q:list[float]

    g3:float

    eta:float

    connectivity:list[tuple[int, int]]

    lambda_q: list[float]

    pump_frequencies:dict[tuple[int, int], list[float]]

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