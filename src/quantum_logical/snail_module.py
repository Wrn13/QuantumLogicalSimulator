from enum import Enum
from math import comb

import numpy as np
from quantum_logical.channel import Channel, SpectatorSNAILLoss
import qutip as qt

from quantum_logical.noisy_gate import NoisyGate


class Spectator_Types(Enum):
    Qubit_Qubit = 0,
    Qubit_SNAIL = 1


class SNAIL_Module():

    num_qubits:int 

    hilbert_space_dim:int

    snail_freq: float

    transmon_frequencies:list[float]

    alpha_q:list[float]

    g_sq :list [float]
    """Coupling between snail and qubits"""

    g3:float
    """Third order kerr nonlinearity strength"""

    eta:float
    """Pump strength"""

    connectivity:list[tuple[int, int]]
    """
    List of connectivity for the SNAIL module
    """

    lambda_q: list[float]
    """
    List of hybridizations g_sq/Delta
    """

    pump_frequencies:dict[tuple[int, int], list[tuple[float, Spectator_Types, float]]]
    """
    List of all possible drive frequencies to first order, including qubit qubit drives,
    snail qubit drives, the snail subharmonic, and the qubit subharmonics.
    Maps pairs of qubits to the drive frequency, spectator type, and strength for that interaction.
    References the SNAIL as the self.num_qubits index
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

        self._compute_hybridization()
        self._generate_pump_frequencies()


    def _generate_pump_frequencies(self):
        """
        Generates leading order pump interactions via Table 1. 
        from Specator Aware Frequency Allocation paper by McKinney et al. 
        24 Sep 2025, https://arxiv.org/pdf/2409.18262
        """
        pump_freqs = dict()

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


        self.pump_frequencies = pump_freqs

    def _compute_hybridization(self):
        """Computes hybridization strenght lambda_q = g_sq/Delta for each qubit."""
        lambda_q = []
        for i in range(self.num_qubits):
            f_i = self.transmon_frequencies[i]
            detuning = abs(f_i - self.snail_freq)
            hybridization_strength = self.g_sq[i] / detuning
            lambda_q.append(hybridization_strength)
        
        self.lambda_q = lambda_q
        

    def compute_gate_detuning(self, qubit_i, qubit_j, qubit_k, qubit_l):
        """ Computes lowercase delta the difference in pump frequencies"""
        target_gate_freq = self.pump_frequencies[qubit_i, qubit_j][0]
        detuned_gate_freq = self.pump_frequencies[qubit_k, qubit_l][0]

        return abs(target_gate_freq - detuned_gate_freq)
    
    def compute_spectator_operator(self, qubit_i:int, qubit_j:int, type:str) -> qt.Qobj:
        """Compute the operator associated  with the given interaction.

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
                destroy_i = qt.tensor([destroy if k == qubit_i else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits)])
                destroy_j = qt.tensor([destroy if k == qubit_j else qt.qeye(self.hilbert_space_dim) for k in range(self.num_qubits)])
                return destroy_i * destroy_j.dag() + destroy_i.dag() * destroy_j
            
    def compute_spectator_snail_loss_channel(
        self, qubit_i: int, chi: float, gate_duration: float
    ) -> Channel:
        return SpectatorSNAILLoss(
            chi=chi,
            gate_duration=gate_duration,
            qubit_index=qubit_i,
            num_qubits=self.num_qubits,
            hilbert_space_dim=self.hilbert_space_dim,
        )

    def generate_noisy_gate(self, qubit_i:int, qubit_j:int, ideal_gate:qt.Qobj, duration:float) -> NoisyGate:
        """
        Generates a noisy 2 qubit gate based on the ideal gate, the system parameters, and the desired interaction qubits.
        
        :param self: Description
        :param qubit_i: Description
        :type qubit_i: int
        :param qubit_j: Description
        :type qubit_j: int
        :param ideal_gate: Description
        :type ideal_gate: qt.Qobj
        :return: Description
        :rtype: NoisyGate
        """

        gate_freq, interaction_type, _ = self.pump_frequencies[qubit_i, qubit_j]

        assert interaction_type == Spectator_Types.Qubit_Qubit, "Currently only supports generating noisy gates for qubit-qubit interactions."
        
        channels = []
        coherent_gate_error = qt.Qobj(
            np.zeros(ideal_gate.shape, dtype=complex),
            dims=ideal_gate.dims,
        )
        # Construct the noisy gate using the ideal unitary and the error strength
        for idx, vals in self.pump_frequencies.items():
            i = idx[0]
            j = idx[1]
            
            pump_freq = vals[0]
            spectator_type = vals[1]
            coeff = vals[2]


            delta = abs(gate_freq - pump_freq)
            #Target gate interaction, not spectator, do not care
            if (i, j) == (qubit_i, qubit_j):
                continue

            if spectator_type == Spectator_Types.Qubit_Qubit:
                coherent_gate_error += 2 * coeff / delta * self.compute_spectator_operator(i, j, spectator_type)
            elif spectator_type == Spectator_Types.Qubit_SNAIL:
                channels.append(self.compute_spectator_snail_loss_channel(min(i, j), 2 * coeff / delta, gate_duration=duration))

        spectator_gate = qt.Qobj(-1j*coherent_gate_error, dims= ideal_gate.dims).expm()
        
        print("Spectator infidelity: ", 1-qt.average_gate_fidelity(spectator_gate, qt.tensor([qt.qeye(2)]*self.num_qubits)))
        total_gate = NoisyGate(unitary = ideal_gate * spectator_gate, num_qubits= self.num_qubits, hilbert_space_dim=self.hilbert_space_dim, channels=channels, qubit_pair= (qubit_i, qubit_j), duration=duration)

        return total_gate
    
    def generate_parallel_noisy_gate(
        self,
        gate_specs: list[tuple[int, int, qt.Qobj]],
        dt: float = 0.03,
    ) -> NoisyGate:
        """
        Generate a NoisyGate for one or more simultaneous 2-qubit gates.

        `gate_specs` entries are (qubit_i, qubit_j, ideal_gate), with
        `ideal_gate` embedded in the full N-qubit Hilbert space. All pairs
        must (i) be qubit-qubit pump entries and (ii) be on disjoint qubits.

        Each pump's on-resonant drive is absorbed into its ideal unitary;
        every other Table-1 transition (including the targets of OTHER
        active pumps) contributes a 2 c_s / |f_p - f_s| spectator term.
        """
        if not gate_specs:
            raise ValueError("gate_specs must be non-empty")

        # ---- validation ----
        used = set()
        active_pairs: list[tuple[int, int]] = []
        for qi, qj, _ in gate_specs:
            key = (qi, qj) if (qi, qj) in self.pump_frequencies else (qj, qi)
            if key not in self.pump_frequencies:
                raise KeyError(f"No pump entry for pair ({qi},{qj})")
            if self.pump_frequencies[key][1] != Spectator_Types.Qubit_Qubit:
                raise ValueError(f"Pair {key} is not a qubit-qubit interaction")
            if used & {qi, qj}:
                raise ValueError(f"Qubit re-use in parallel set: {qi},{qj}")
            used.update({qi, qj})
            active_pairs.append(key)

        active_pumps = [
            (qi, qj, self.pump_frequencies[qi, qj][0]) for qi, qj in active_pairs
        ]

        # ---- combined ideal unitary (commutes pairwise: disjoint supports) ----
        combined_ideal = gate_specs[0][2]
        for _, _, U in gate_specs[1:]:
            combined_ideal = U * combined_ideal

        # ---- accumulate spectator coherent error and incoherent channels ----
        coherent_gate_error = qt.Qobj(
            np.zeros(combined_ideal.shape, dtype=complex),
            dims=combined_ideal.dims,
        )
        channels: list[Channel] = []

        for spec_idx, (f_s, spec_type, c_s) in self.pump_frequencies.items():
            for active_qi, active_qj, f_p in active_pumps:
                # absorb pump p's own on-resonant target into the ideal unitary
                if spec_idx == (active_qi, active_qj):
                    continue

                delta = abs(f_p - f_s)
                if delta < 1e-9:
                    # frequency collision: parallelizing these gates is unphysical
                    raise ValueError(
                        f"Near-degenerate pump for active {(active_qi, active_qj)} "
                        f"and spectator {spec_idx}: delta={delta:.3e}"
                    )
                strength = 2 * c_s / delta

                if spec_type == Spectator_Types.Qubit_Qubit:
                    i, j = spec_idx
                    coherent_gate_error += strength * self.compute_spectator_operator(
                        i, j, spec_type
                    )
                elif spec_type == Spectator_Types.Qubit_SNAIL:
                    qubit = min(spec_idx)  # (qubit_index, num_qubits)
                    channels.append(
                        self.compute_spectator_snail_loss_channel(
                            qubit, strength, dt=dt
                        )
                    )

        spectator_gate = (-1j * coherent_gate_error).expm()

        return NoisyGate(
            combined_ideal * spectator_gate,
            self.num_qubits,
            self.hilbert_space_dim,
            channels,
            0,
            tuple(active_pairs),
        )


    #def generate_noisy_gate(self, qubit_i, qubit_j, ideal_gate):
    #    return self.generate_parallel_noisy_gate([(qubit_i, qubit_j, ideal_gate)])

