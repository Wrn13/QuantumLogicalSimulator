from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


def ideal_plot_results_linear_plot(trotter_dt: float, state_list_4: List[qt.Qobj], state_list_5: List[qt.Qobj], no_correction:list[qt.Qobj], output_path: str, t1, t2, init_rho) -> None:
    """
    Plot results on a linear plot
    
    :param trotter_dt: The discritized time length
    :type trotter_dt: float
    :param state_list_4: The list of 4 qubit states with correction
    :type state_list_4: List[qt.Qobj]
    :param state_list_5: The list of 5 qubit states with correction
    :type state_list_5: List[qt.Qobj]
    :param no_correction: The list of uncorrected 4 qubit states
    :type no_correction: list[qt.Qobj]
    :param output_path: The output path to send the images to
    :type output_path: str
    :param t1: The T1 value
    :type t1: float 
    :param t2: The T2 value
    :type t2: float
    :param init_rho: The initial quantum state
    :type init_rho: qt.Qobj
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    time_steps_no = np.linspace(0, trotter_dt * (len(no_correction)), len(no_correction))
    time_steps_4 = np.linspace(0, trotter_dt * (len(state_list_4)), len(state_list_4))
    time_steps_5 = np.linspace(0, trotter_dt * (len(state_list_5)), len(state_list_5))

    probabilities_4 = [state.ptrace([0, 1, 2]).diag().real for state in state_list_4]
    probabilities_4 = np.array(probabilities_4)

    probabilities_5 = [state.ptrace([0, 1, 2]).diag().real for state in state_list_5]
    probabilities_5 = np.array(probabilities_5)

    fidelities_4 = [qt.fidelity(state.ptrace([0, 1, 2]), init_rho) for state in state_list_4]
    fidelities_4 = np.array(fidelities_4)

    fidelities_5 = [qt.fidelity(state.ptrace([0, 1, 2]), init_rho) for state in state_list_5]
    fidelities_5 = np.array(fidelities_5)

    fidelity_no_correction = [qt.fidelity(state.ptrace([0, 1, 2]), init_rho) for state in no_correction]
    fidelity_no_correction = np.array(fidelity_no_correction)

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps_4, fidelities_4, label="4Q Fidelity")
    plt.plot(time_steps_5, fidelities_5, label="5Q Fidelity")
    plt.plot(time_steps_no, fidelity_no_correction, label="No Correction Fidelity")
    plt.xlabel(r"Time Step ($\mu$s)")
    plt.ylabel("Fidelities")
    plt.suptitle(f"Fidelity between initial state vs time with T1 = {t1:.1f}μs, T2 = {t2:.1f}μs")
    plt.title(f"Final Fidelity: 4Q: {fidelities_4[-1]:.4f}, 5Q: {fidelities_5[-1]:.4f}")
    plt.legend()
    plt.savefig(f"{output_path}/linear_fidelities_plot.png")
    plt.show()


def ideal_plot_results_log_plot(trotter_dt:float, state_list_4: List[qt.Qobj], state_list_5:List[qt.Qobj], no_correction:List[qt.Qobj],
                                  output_path:str, t1:float, t2:float, init_rho:qt.Qobj):
    """
    Plot results on a log scale plot.
    
    :param trotter_dt: The discritized time length
    :type trotter_dt: float
    :param state_list_4: The list of 4 qubit states with correction
    :type state_list_4: List[qt.Qobj]
    :param state_list_5: The list of 5 qubit states with correction
    :type state_list_5: List[qt.Qobj]
    :param no_correction: The list of uncorrected 4 qubit states
    :type no_correction: list[qt.Qobj]
    :param output_path: The output path to send the images to
    :type output_path: str
    :param t1: The T1 value
    :type t1: float 
    :param t2: The T2 value
    :type t2: float
    :param init_rho: The initial quantum state
    :type init_rho: qt.Qobj
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    time_steps_4 = np.linspace(0, trotter_dt * len(state_list_4), len(state_list_4))
    time_steps_5 = np.linspace(0, trotter_dt * len(state_list_5), len(state_list_5))
    time_steps_nc = np.linspace(0, trotter_dt * len(no_correction), len(no_correction))

    # qt.fidelity returns sqrt-fidelity; square it to get the standard Uhlmann F
    def infid(states):
        F = np.array([qt.fidelity(s.ptrace([0, 1, 2]), init_rho) for s in states]) ** 2
        return 1 - F

    infid_4  = infid(state_list_4)
    infid_5  = infid(state_list_5)
    infid_nc = infid(no_correction)

    # Guard against exactly-1 fidelity producing zero infidelity
    floor = 1e-12
    infid_4  = np.clip(infid_4,  floor, None)
    infid_5  = np.clip(infid_5,  floor, None)
    infid_nc = np.clip(infid_nc, floor, None)

    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_steps_4,  infid_4, label='4-qubit (sequential)')
    ax.plot(time_steps_5,  infid_5, label='5-qubit (parallel)')
    ax.plot(time_steps_nc, infid_nc, label='no correction')

    ax.set_yscale('log')

    nines = range(1, 5)
    tick_positions = [10.0 ** (-k) for k in nines]
    tick_labels = ['0.' + '9' * k for k in nines]

    ax.set_ylim(10 ** (-(max(nines) + 0.3)), 10 ** (-(min(nines) - .9)))
    ax.yaxis.set_major_locator(FixedLocator(tick_positions))
    ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.invert_yaxis()
    ax.set_ylabel('Fidelity')
    ax.set_xlabel('Time (μs)')
    ax.legend()
    ax.grid(True, which='major', alpha=0.3)

    ax.set_title(f"Fidelity between initial state vs time with T1 = {t1:.1f}μs, T2 = {t2:.1f}μs")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'log_fidelities_plot.png'), dpi=150)
    plt.show()
