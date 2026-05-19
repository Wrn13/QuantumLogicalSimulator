"""Gate-attached error channels for the Trotterized simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qutip import Qobj

from quantum_logical.channel import Channel


@dataclass
class NoisyGate:
    """A discrete unitary bundled with its gate-specific Kraus error channel.

    The unitary is applied via Trotterized fractional powers in
    `TrotterGroup.apply` (so continuous T1/T2 decay correctly accumulates
    during `duration`). The Kraus channel is applied *once* at the end of
    the gate.

    Parameters
    ----------
    unitary : Qobj
        Ideal gate unitary on the full Hilbert space.
    kraus_operators : list of Qobj
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
    channels: list[Channel]
    duration: float
    qubit_pair: Optional[tuple[int, int]] = None

    def __post_init__(self):
        if not Qobj(self.unitary).isunitary:
            raise ValueError("NoisyGate.unitary must be unitary.")
       
    
