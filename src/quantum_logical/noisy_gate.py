"""Gate-attached error channels for the Trotterized simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from cirq import q
import numpy as np
from qutip import Qobj


@dataclass
class NoisyGate:
    """A discrete unitary bundled with its gate-specific Kraus error channel.

    The unitary is applied via Trotterized fractional powers in
    `TrotterGroup.apply` (so continuous T1/T2 decay correctly accumulates
    during `duration`). The Kraus channel is applied *once* at the end of
    the gate. This matches the conventional decomposition of gate errors
    into (decoherent decay) + (coherent / twirled stochastic) terms.

    Parameters
    ----------
    unitary : Qobj
        Ideal gate unitary on the full Hilbert space.
    kraus_operators : sequence of ndarray
        Kraus operators of the discrete error channel, each shaped
        (D, D) where D = total Hilbert-space dimension. Must satisfy
        sum_k E_k^dagger E_k = I to numerical tolerance.
    duration : float
        Gate duration in the same units as TrotterGroup.trotter_dt
        (continuous channels integrate over this).
    qubit_pair : tuple[int, int], optional
        Indices of the qubits the gate physically acts on; carried
        for bookkeeping / debugging only.
    """
    unitary: Qobj
    num_qubits: int
    hilbert_space_dim: int
    kraus_operators: Sequence[np.ndarray]
    duration: float
    qubit_pair: Optional[tuple[int, int]] = None

    def __post_init__(self):
        if not Qobj(self.unitary).isunitary:
            raise ValueError("NoisyGate.unitary must be unitary.")
        D = self.unitary.shape[0]
        completeness = sum(E.conj().T @ E for E in self.kraus_operators)
        if not np.allclose(completeness, np.eye(D), atol=1e-6):
            raise ValueError(
                "Kraus operators do not satisfy sum_k E_k^dag E_k = I."
            )

    def apply_kraus(self, rho: np.ndarray) -> np.ndarray:
        return sum(E @ rho @ E.conj().T for E in self.kraus_operators)
    

class SpectatorErrorGate(NoisyGate):
    def __init__(
        self,
        num_qubits: int = 2,
        hilbert_space_dim: int = 2,
        eta:float = 1.8,
        g3:float = 60e6,
        lambda_q:list[float] = None,
        alpha_q: list[float] = None,
        frequencies: list[float] = None,
        qubit_pair: Optional[tuple[int, int]] = None
        
    ):
        
        super.__init__(num_qubits = num_qubits, hilbert_space_dim = hilbert_space_dim, qubit_pair = qubit_pair)