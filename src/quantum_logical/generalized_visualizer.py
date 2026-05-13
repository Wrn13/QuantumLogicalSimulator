"""
visualizer.py

Plotting + validation pipeline for the 3-qubit phase-flip code data produced
by `single_run_n_iterations` in your generator script.

Each .npz file contains three arrays of qt.Qobj density matrices:
    arr_0 = no-correction trajectory             (4-qubit Hilbert space)
    arr_1 = ideal_dropped 4-qubit (sequential)   (4-qubit Hilbert space)
    arr_2 = ideal_dropped 5-qubit (parallel)     (5-qubit Hilbert space)

Filename conventions:
    Key states:  <n_iter>_itr_T1_<T1>_T2_<T2>_<state>.npz
    Haar states: <index>_<n_iter>_itr_T1_<T1>_T2_<T2>.npz

Directory layout assumed:
    <root>/
        key_initial_states/<n>_iterations/*.npz
        haar_random_states/*.npz

Usage:
    python visualizer.py [data_root]   # default data_root = "./data"
"""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =============================================================================
# Configuration
# =============================================================================

TROTTER_DT = 0.03    # microseconds per stored trotter step (matches save script)
N_DATA = 3           # logical data qubits (first three positions in the tensor)

# Map plot label -> array key in the .npz file
PROTOCOLS = {
    "no correction":         "arr_0",
    "4-qubit (sequential)":  "arr_1",
    "5-qubit (parallel)":    "arr_2",
}

PROTOCOL_COLOR = {
    "no correction":         "#666666",
    "4-qubit (sequential)":  "#0072B2",
    "5-qubit (parallel)":    "#D55E00",
}

# Logical-state palette (Wong, Nature Methods 8, 441 (2011))
STATE_ORDER = ["0", "1", "+", "-", "i", "-i"]
STATE_LABELS = {
    "0":  r"$|0\rangle_L$",
    "1":  r"$|1\rangle_L$",
    "+":  r"$|+\rangle_L$",
    "-":  r"$|-\rangle_L$",
    "i":  r"$|{+i}\rangle_L$",
    "-i": r"$|{-i}\rangle_L$",
}
PALETTE = {
    "0":  "#0072B2", "1":  "#56B4E9",
    "+":  "#D55E00", "-":  "#E69F00",
    "i":  "#009E73", "-i": "#CC79A7",
}


# =============================================================================
# Filename parsing
# =============================================================================

KEY_FILENAME_RE = re.compile(
    r"^(?P<iter>\d+)_itr_T1_(?P<T1>\d+)_T2_(?P<T2>\d+)_(?P<state>[+\-i0-9]+)\.npz$"
)
HAAR_FILENAME_RE = re.compile(
    r"^(?P<index>\d+)_(?P<iter>\d+)_itr_T1_(?P<T1>\d+)_T2_(?P<T2>\d+)\.npz$"
)


def parse_filename(path: Path) -> dict | None:
    """Return metadata dict, or None if filename matches neither pattern."""
    m = KEY_FILENAME_RE.match(path.name)
    if m:
        return {
            "kind":  "key",
            "iter":  int(m.group("iter")),
            "T1":    int(m.group("T1")),
            "T2":    int(m.group("T2")),
            "state": m.group("state"),
            "path":  path,
        }
    m = HAAR_FILENAME_RE.match(path.name)
    if m:
        return {
            "kind":  "haar",
            "iter":  int(m.group("iter")),
            "T1":    int(m.group("T1")),
            "T2":    int(m.group("T2")),
            "index": int(m.group("index")),
            "path":  path,
        }
    return None


# =============================================================================
# Density-matrix handling
# =============================================================================

def _coerce_to_numpy(rho) -> np.ndarray:
    """Accept qt.Qobj or numpy array, return 2D complex ndarray."""
    if hasattr(rho, "full"):
        return rho.full()
    return np.asarray(rho)


def stack_trajectory(arr) -> np.ndarray:
    """Convert stored object array / list of Qobjs / 3D ndarray into (T, D, D)."""
    if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 3:
        return arr.astype(complex, copy=False)
    return np.array([_coerce_to_numpy(r) for r in arr], dtype=complex)


def partial_trace_ancilla(rho_traj: np.ndarray, n_data: int = N_DATA) -> np.ndarray:
    """
    Trace out the last (n_total - n_data) qubits.
    Assumes qt.tensor(data_qubits, ancillae) ordering -- data first.
    Input  shape: (T, D, D) with D = 2**n_total
    Output shape: (T, d_data, d_data) with d_data = 2**n_data
    """
    T, D, _ = rho_traj.shape
    n_total = int(round(np.log2(D)))
    n_anc = n_total - n_data
    d_data = 2 ** n_data
    d_anc = 2 ** n_anc

    if n_anc < 0:
        raise ValueError(f"D={D} is smaller than 2**n_data=2**{n_data}")
    if n_anc == 0:
        return rho_traj

    # rho[i_data * d_anc + i_anc, j_data * d_anc + j_anc]
    rho5 = rho_traj.reshape(T, d_data, d_anc, d_data, d_anc)
    return np.einsum("tikjk->tij", rho5)


def fidelity_to_initial(rho_data_traj: np.ndarray) -> np.ndarray:
    """
    F(t) = Tr(rho_data(0) * rho_data(t))
    For pure rho_data(0), this is the state fidelity <psi_0 | rho(t) | psi_0>.
    """
    rho_0 = rho_data_traj[0]
    return np.real(np.einsum("ij,tji->t", rho_0, rho_data_traj))


# =============================================================================
# Loading (robust to corrupted files)
# =============================================================================

def process_file(
    path: Path,
    n_data: int = N_DATA,
    trotter_dt: float = TROTTER_DT,
    sanity_tol: float = 1e-6,
) -> dict | None:
    """Load one .npz, return record dict, or None on failure (with a warning)."""
    meta = parse_filename(path)
    if meta is None:
        return None

    try:
        with np.load(path, allow_pickle=True) as data:
            available = set(data.files)
            if not all(k in available for k in PROTOCOLS.values()):
                warnings.warn(
                    f"{path.name}: missing keys "
                    f"(have {sorted(available)}, need {list(PROTOCOLS.values())})"
                )
                return None

            protocols = {}
            for label, key in PROTOCOLS.items():
                traj_full = stack_trajectory(data[key])
                traj_data = partial_trace_ancilla(traj_full, n_data=n_data)
                F = fidelity_to_initial(traj_data)
                if abs(F[0] - 1.0) > sanity_tol:
                    warnings.warn(
                        f"{path.name} [{label}]: F(0) = {F[0]:.6f} "
                        f"(expected 1.0). Either rho(0) is not pure, "
                        f"or qubit ordering differs from data-first convention."
                    )
                t = np.arange(F.size) * trotter_dt
                protocols[label] = {"time": t, "F": F}
    except (EOFError, OSError, ValueError, KeyError) as e:
        warnings.warn(f"FAILED to load {path.name}: {type(e).__name__}: {e}")
        return None

    meta["protocols"] = protocols
    return meta


def load_dataset(
    root: Path,
    n_data: int = N_DATA,
    trotter_dt: float = TROTTER_DT,
    verbose: bool = True,
) -> list[dict]:
    """Walk root recursively, parse + process every .npz it finds."""
    paths = sorted(root.rglob("*.npz"))
    if verbose:
        print(f"[load_dataset] {root}: found {len(paths)} .npz file(s)")
    records, failed = [], []
    for p in paths:
        rec = process_file(p, n_data=n_data, trotter_dt=trotter_dt)
        if rec is None:
            failed.append(p)
        else:
            records.append(rec)
    if verbose and failed:
        print(f"[load_dataset] {len(failed)} file(s) failed:")
        for p in failed:
            print(f"    {p}")
    return records


def validate_files(root: Path) -> None:
    """Quick standalone integrity check: lists corrupted/incomplete files."""
    print(f"\n=== validate_files: {root} ===")
    paths = sorted(root.rglob("*.npz"))
    print(f"Found {len(paths)} .npz files")
    for p in paths:
        size_mb = p.stat().st_size / 1e6
        try:
            with np.load(p, allow_pickle=True) as data:
                missing = [k for k in PROTOCOLS.values() if k not in data.files]
                if missing:
                    print(f"  [MISSING KEYS {missing}] {p.name}  ({size_mb:.1f} MB)")
                    continue
                # Touch each array to force a read
                for k in PROTOCOLS.values():
                    arr = data[k]
                    _ = len(arr)  # trigger lazy load
            print(f"  [OK]                          {p.name}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  [{type(e).__name__}] {p.name}  ({size_mb:.1f} MB): {e}")


# =============================================================================
# Plotting helpers
# =============================================================================

def _safe_infidelity(F: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.clip(1.0 - F, floor, None)


def _slice_trace(t: np.ndarray, F: np.ndarray, skip_initial: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Drop the first `skip_initial` samples from both arrays."""
    if skip_initial <= 0:
        return t, F
    return t[skip_initial:], F[skip_initial:]


def configure_fidelity_yaxis(ax, label: str = "Fidelity",
                              ylim: tuple[float, float] | None = None) -> None:
    """
    Display the y-axis in 'number of 9's' format (..., 0.9, 0.99, 0.999, ...)
    while internally plotting infidelity (1 - F) on a log scale. Major ticks
    sit on decades of (1 - F); minor ticks are unlabeled to keep the panel
    readable.

    `ylim` is given in INFIDELITY coordinates, e.g. (1e-6, 1e-1) shows the
    band of fidelities between 0.9 and 0.999999. If None, matplotlib picks
    bounds automatically from the data.
    """
    ax.set_yscale("log")

    def fmt(y, _pos):
        if y <= 0:
            return ""
        if y >= 1:
            return "0"
        F = 1.0 - y
        log_y = -np.log10(y)
        n = int(round(log_y))
        if n < 1:
            return f"{F:.2g}"
        return f"{F:.{n}f}"

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel(label)
    # Invert so high fidelity is at the top (lower 1-F at top).
    # Matplotlib's get_ylim returns (bottom, top); after set_yscale("log") on
    # positive data, bottom < top. Reversing gives the conventional fidelity
    # orientation. Setting limits explicitly (rather than calling
    # invert_yaxis()) makes the operation idempotent if called twice.
    if ylim is not None:
        ax.set_ylim(ylim[1], ylim[0])
    else:
        lo, hi = ax.get_ylim()
        if lo < hi:
            ax.set_ylim(hi, lo)


def _group(records: list[dict], **filters) -> list[dict]:
    return [r for r in records if all(r.get(k) == v for k, v in filters.items())]


def _stack_states(panel_records: list[dict], protocol_label: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, F_stack) where F_stack has shape (n_states, T)."""
    panel_records = sorted(panel_records,
                           key=lambda r: STATE_ORDER.index(r["state"]))
    t_ref = panel_records[0]["protocols"][protocol_label]["time"]
    F_stack = []
    for r in panel_records:
        p = r["protocols"][protocol_label]
        if not np.allclose(p["time"], t_ref):
            raise ValueError(
                f"time grids differ across states for protocol {protocol_label!r}"
            )
        F_stack.append(p["F"])
    return t_ref, np.array(F_stack)


# =============================================================================
# Plot 1: protocol comparison (the main validation figure)
# =============================================================================

def plot_protocol_comparison(
    key_records: list[dict],
    save_path: Path | str | None = None,
    show_band: bool = True,
    skip_initial: int = 3,
    ylim: tuple[float, float] | None = None,
):
    """
    2D grid (rows: iterations, cols: T2). Each panel shows the 6-state
    average for each protocol. Shaded bands span min..max across the 6
    logical states -- wide bands signal strong state dependence.

    `skip_initial` drops the leading samples (default 3) so the y-axis
    isn't dominated by the F=1 starting point bleeding to -inf on the log
    scale.

    `ylim` is in infidelity (1-F) coordinates. Pass e.g. (1e-5, 1e-1) to
    fix the visible band to 0.9 ... 0.99999 across all panels.
    """
    key_records = [r for r in key_records if r["kind"] == "key"]
    iters = sorted({r["iter"] for r in key_records})
    T2s   = sorted({r["T2"]   for r in key_records})
    fig, axes = plt.subplots(
        len(iters), len(T2s),
        figsize=(5.0 * len(T2s), 3.6 * len(iters)),
        sharey=True, squeeze=False,
    )

    for i, n_iter in enumerate(iters):
        for j, T2 in enumerate(T2s):
            ax = axes[i, j]
            panel = _group(key_records, iter=n_iter, T2=T2)
            if len(panel) == 0:
                ax.set_visible(False)
                continue

            for label in PROTOCOLS:
                t, F_stack = _stack_states(panel, label)
                t_plot, F_avg_plot = _slice_trace(t, F_stack.mean(axis=0), skip_initial)
                ax.plot(
                    t_plot, _safe_infidelity(F_avg_plot),
                    color=PROTOCOL_COLOR[label],
                    lw=2.0, label=label,
                )
                if show_band:
                    _, F_hi = _slice_trace(t, F_stack.max(axis=0), skip_initial)
                    _, F_lo = _slice_trace(t, F_stack.min(axis=0), skip_initial)
                    ax.fill_between(
                        t_plot,
                        _safe_infidelity(F_hi),
                        _safe_infidelity(F_lo),
                        color=PROTOCOL_COLOR[label],
                        alpha=0.15, lw=0,
                    )

            T1 = panel[0]["T1"]
            ax.set_title(
                f"{n_iter} iter, "
                rf"$T_1={T1}\,\mu\mathrm{{s}}$, $T_2={T2}\,\mu\mathrm{{s}}$",
                fontsize=10,
            )
            ax.grid(True, which="both", alpha=0.25)
            if i == len(iters) - 1:
                ax.set_xlabel(r"Time ($\mu$s)")
            configure_fidelity_yaxis(
                ax,
                label=(r"Avg fidelity $\langle F\rangle_{6}$" if j == 0 else ""),
                ylim=ylim,
            )

    axes[0, -1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Phase-flip code: average logical fidelity by protocol",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig, axes


# =============================================================================
# Plot 2: per-state breakdown (codeword vs off-axis asymmetry)
# =============================================================================

def plot_per_state_breakdown(
    key_records: list[dict],
    n_iter: int, T2: int,
    save_path: Path | str | None = None,
    skip_initial: int = 3,
    ylim: tuple[float, float] | None = None,
):
    """3 panels (one per protocol); each panel overlays the 6 state curves."""
    panel = sorted(
        _group(key_records, kind="key", iter=n_iter, T2=T2),
        key=lambda r: STATE_ORDER.index(r["state"]),
    )
    if not panel:
        print(f"[plot_per_state_breakdown] no data for n_iter={n_iter}, T2={T2}")
        return None, None

    fig, axes = plt.subplots(
        1, len(PROTOCOLS),
        figsize=(5.0 * len(PROTOCOLS), 3.8),
        sharey=True,
    )

    for ax, label in zip(axes, PROTOCOLS):
        F_stack = []
        for r in panel:
            p = r["protocols"][label]
            t_plot, F_plot = _slice_trace(p["time"], p["F"], skip_initial)
            ax.plot(
                t_plot, _safe_infidelity(F_plot),
                color=PALETTE[r["state"]],
                lw=1.0, alpha=0.85,
                label=STATE_LABELS[r["state"]],
            )
            F_stack.append(p["F"])
        t_ref = panel[0]["protocols"][label]["time"]
        F_avg = np.mean(F_stack, axis=0)
        t_avg, F_avg_plot = _slice_trace(t_ref, F_avg, skip_initial)
        ax.plot(
            t_avg, _safe_infidelity(F_avg_plot),
            color="black", lw=2.2, zorder=10,
            label="avg over 6",
        )
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        configure_fidelity_yaxis(ax, label="", ylim=ylim)

    axes[0].set_ylabel("Fidelity")
    axes[-1].legend(loc="lower right", fontsize=8, ncol=2)

    T1 = panel[0]["T1"]
    fig.suptitle(
        rf"Per-state breakdown: $T_1 = {T1}\,\mu$s, "
        rf"$T_2 = {T2}\,\mu$s, {n_iter} iter",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig, axes


# =============================================================================
# Plot: states-by-protocol (transpose of per_state_breakdown)
# =============================================================================

def plot_states_by_protocol(
    key_records: list[dict],
    states: list[str] = ("0", "1", "+"),
    n_iter: int = 600,
    T2: int | None = None,
    save_path: Path | str | None = None,
    skip_initial: int = 3,
    ylim: tuple[float, float] | None = None,
):
    """
    One panel per requested logical state; all three protocols overlaid.

    This is the transpose of `plot_per_state_breakdown`. It answers
    "for this input state, which protocol is winning?" rather than
    "for this protocol, which states does it protect?" --- the natural
    framing when the diagnostic question is whether the code helps a
    particular state at all.

    Parameters
    ----------
    key_records
        Output of load_dataset on the key_initial_states directory.
    states
        Logical state labels to plot, left-to-right. Default ("0", "1", "+")
        contrasts the two codewords against the off-axis state |+>_L.
        Any subset of {"0", "1", "+", "-", "i", "-i"} works.
    n_iter
        Iteration count to filter on.
    T2
        T2 value (microseconds). If None, every T2 in the dataset becomes
        a separate row of the figure.
    save_path
        If given, saves both `<save_path>` (PDF) and `<save_path>.png` so
        the figure is ready for both a paper and a slide deck.
    skip_initial
        Leading samples to drop -- avoids the F=1 cliff at t=0 dominating
        the log y-axis.
    ylim
        Infidelity-coordinate y-limits, e.g. (1e-5, 1e-1) maps to fidelities
        between 0.9 and 0.99999. None lets matplotlib autoscale.

    Returns
    -------
    fig, axes
        matplotlib Figure and 2D ndarray of Axes (rows = T2 values,
        cols = requested states).
    """
    key_records = [r for r in key_records if r["kind"] == "key"]

    if T2 is None:
        T2s = sorted({
            r["T2"] for r in key_records
            if r["iter"] == n_iter and r["state"] in states
        })
        if not T2s:
            print(f"[plot_states_by_protocol] no data for n_iter={n_iter}")
            return None, None
    else:
        T2s = [T2]

    n_rows, n_cols = len(T2s), len(states)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.7 * n_cols, 3.6 * n_rows),
        sharey=True, sharex="col", squeeze=False,
    )

    T1_seen = None

    for i, T2_val in enumerate(T2s):
        for j, state in enumerate(states):
            ax = axes[i, j]
            matches = _group(key_records, iter=n_iter, T2=T2_val, state=state)
            if not matches:
                ax.text(
                    0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            r = matches[0]
            T1_seen = r["T1"]

            for label in PROTOCOLS:
                p = r["protocols"][label]
                t_plot, F_plot = _slice_trace(p["time"], p["F"], skip_initial)
                ax.plot(
                    t_plot, _safe_infidelity(F_plot),
                    color=PROTOCOL_COLOR[label],
                    lw=1.8, label=label,
                )

            title = STATE_LABELS[state]
            if T2 is None:
                title = rf"{title}, $T_2={T2_val}\,\mu\mathrm{{s}}$"
            ax.set_title(title, fontsize=11)
            ax.grid(True, which="both", alpha=0.25)
            if i == n_rows - 1:
                ax.set_xlabel(r"Time ($\mu$s)")
            configure_fidelity_yaxis(
                ax,
                label="Fidelity" if j == 0 else "",
                ylim=ylim,
            )

    axes[0, -1].legend(loc="best", fontsize=8)

    suptitle_parts = []
    if T1_seen is not None:
        suptitle_parts.append(rf"$T_1={T1_seen}\,\mu$s")
    if T2 is not None:
        suptitle_parts.append(rf"$T_2={T2}\,\mu$s")
    suptitle_parts.append(f"{n_iter} iter")
    fig.suptitle(", ".join(suptitle_parts), fontsize=12, y=1.00)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        fig.savefig(save_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    return fig, axes


# =============================================================================
# Plot 3: Haar / 6-state consistency check (2-design)
# =============================================================================

def plot_haar_consistency(
    key_records: list[dict], haar_records: list[dict],
    n_iter: int, T2: int,
    save_path: Path | str | None = None,
    skip_initial: int = 3,
    ylim: tuple[float, float] | None = None,
):
    """
    Per-protocol comparison of 6-state average vs Haar Monte Carlo average.
    For d=2, the 6 logical Pauli eigenstates form a state 2-design, so these
    two should agree to within MC noise of the Haar sample size.

    NB: your generator uses np.random.rand for the Haar coefficients, which
    is NOT a uniform Haar sample (it concentrates the Bloch sphere distribution
    in the first quadrant). Even with perfect simulation the two curves will
    not agree perfectly until the sampler is switched to complex standard
    Gaussians (Ginibre method).
    """
    key  = _group(key_records,  iter=n_iter, T2=T2, kind="key")
    haar = _group(haar_records, iter=n_iter, T2=T2, kind="haar")
    if not key or not haar:
        print(f"[plot_haar_consistency] missing data for n_iter={n_iter}, T2={T2}")
        return None, None

    n_haar = len(haar)

    fig, axes = plt.subplots(
        1, len(PROTOCOLS),
        figsize=(5.0 * len(PROTOCOLS), 4.0),
        sharey=True,
    )

    for ax, label in zip(axes, PROTOCOLS):
        t_ref, F6_stack = _stack_states(key, label)
        F6 = F6_stack.mean(axis=0)

        Fh_stack = np.array([r["protocols"][label]["F"] for r in haar])
        for r in haar:
            assert np.allclose(r["protocols"][label]["time"], t_ref), (
                "haar time grid differs from key time grid"
            )
        F_haar = Fh_stack.mean(axis=0)
        sem = Fh_stack.std(axis=0, ddof=1) / np.sqrt(n_haar)

        t_plot, F6_plot     = _slice_trace(t_ref, F6,     skip_initial)
        _,      F_haar_plot = _slice_trace(t_ref, F_haar, skip_initial)
        _,      sem_plot    = _slice_trace(t_ref, sem,    skip_initial)

        ax.plot(t_plot, _safe_infidelity(F6_plot),     lw=2.0, label="6-state avg")
        ax.plot(t_plot, _safe_infidelity(F_haar_plot), lw=2.0,
                label=f"Haar avg, N={n_haar}")
        ax.fill_between(
            t_plot,
            _safe_infidelity(F_haar_plot + sem_plot),
            _safe_infidelity(F_haar_plot - sem_plot),
            alpha=0.25, lw=0,
        )
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        configure_fidelity_yaxis(ax, label="", ylim=ylim)

    axes[0].set_ylabel("Fidelity")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        rf"2-design consistency check: $T_2 = {T2}\,\mu$s, {n_iter} iter",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    return fig, axes


# =============================================================================
# Driver
# =============================================================================

def main(root: Path = Path("data"), out_dir: Path = Path("plots")):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Integrity check first --- prints which files are corrupted
    validate_files(root / "key_initial_states")
    # validate_files(root / "haar_random_states")

    # 2. Load (skips corrupted files automatically with warnings)
    print("\n=== Loading key states ===")
    key = load_dataset(root / "key_initial_states", verbose=True)
    print(f"Loaded {len(key)} key records")

    # print("\n=== Loading Haar states ===")
    # haar = load_dataset(root / "haar_random_states", verbose=True)
    # print(f"Loaded {len(haar)} Haar records")

    # 3. Main result: protocol comparison
    if key:
        plot_protocol_comparison(
            key, save_path=out_dir / "protocol_comparison.pdf"
        )

        # 4. Per-state breakdown for every (iter, T2) combination
        iters = sorted({r["iter"] for r in key})
        T2s   = sorted({r["T2"]   for r in key})
        for n_iter in iters:
            for T2 in T2s:
                plot_per_state_breakdown(
                    key, n_iter=n_iter, T2=T2,
                    save_path=out_dir / f"per_state_{n_iter}iter_T2_{T2}.pdf",
                )

        # 4b. States-by-protocol: |0>_L, |1>_L, |+>_L compared across schemes
        for n_iter in iters:
            for T2 in T2s:
                plot_states_by_protocol(
                    key, states=("0", "1", "+"),
                    n_iter=n_iter, T2=T2,
                    save_path=out_dir / f"states_by_protocol_{n_iter}iter_T2_{T2}.pdf",
                )

    # # 5. Haar/6-state consistency (one plot per (iter, T2))
    # if key and haar:
    #     iters = sorted({r["iter"] for r in haar})
    #     T2s   = sorted({r["T2"]   for r in haar})
    #     for n_iter in iters:
    #         for T2 in T2s:
    #             plot_haar_consistency(
    #                 key, haar, n_iter=n_iter, T2=T2,
    #                 save_path=out_dir / f"haar_check_{n_iter}iter_T2_{T2}.pdf",
    #             )

    plt.show()


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    main(root)