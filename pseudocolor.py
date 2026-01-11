import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Union


def _as_1d_float_vector(x: Union[str, Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert a cell value into a 1D float vector.

    Supports:
    - list/tuple/np.ndarray of numbers
    - strings like "[0.1 0.2 0.3]" or "[0.1, 0.2, 0.3]"
    """
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=float).ravel()

    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float).ravel()

    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.replace(",", " ")
        v = np.fromstring(s, sep=" ", dtype=float)
        if v.size == 0:
            raise ValueError(f"Could not parse numeric vector from string: {x[:80]}...")
        return v.ravel()

    return np.asarray(x, dtype=float).ravel()


def _pick_column(arc_df, candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in arc_df.columns:
            return c
    raise KeyError(
        f"None of the candidate columns exist: {list(candidates)}. "
        f"Available columns: {list(arc_df.columns)}"
    )


def _stack_field(arc_df, col: str, R: np.ndarray) -> np.ndarray:
    """Stack a dataframe column of 1D profiles into a 2D array (NZ x NR)."""
    R = np.asarray(R).ravel()
    profiles = []
    for i, cell in enumerate(arc_df[col].values):
        v = _as_1d_float_vector(cell)
        if v.size != R.size:
            raise ValueError(
                f"Column '{col}', row {i}: profile length {v.size} does not match len(R)={R.size}."
            )
        profiles.append(v)
    return np.stack(profiles, axis=0)

def _stack_field_with_x(arc_df, field_col: str, x_col: str):
    """
    Stack a dataframe column of 1D profiles into a 2D array (NZ x NX)
    and also return the corresponding X coordinates (NZ x NX).
    """
    profiles = []
    xcoords = []

    for i, (cell_f, cell_x) in enumerate(
        zip(arc_df[field_col].values, arc_df[x_col].values)
    ):
        v = _as_1d_float_vector(cell_f)
        x = _as_1d_float_vector(cell_x)

        if v.size != x.size:
            raise ValueError(
                f"Row {i}: field '{field_col}' has length {v.size} "
                f"but '{x_col}' has length {x.size}."
            )

        profiles.append(v)
        xcoords.append(x)

    return np.stack(xcoords, axis=0), np.stack(profiles, axis=0)

def plot_arccolumn_isolines(
    arc_df,
    Zcoord: np.ndarray,
    R: np.ndarray,
    Ra: np.ndarray,
    cmap: str = "viridis",
):
    """
    Heatmaps for magnetic field (Bθ), temperature (T), and axial velocity (Vz)
    over the (R, Z) plane.

    Boundaries (color scaling):
    1. Magnetic field: full range (vmin=min, vmax=max)
    2. Temperature: vmin=300 K, vmax=max available
    3. Velocity: vmin=0 m/s, vmax=max available
    """
    R = np.asarray(R).ravel()
    Zcoord = np.asarray(Zcoord).ravel()

    col_B = _pick_column(arc_df, ["Bθ", "Btheta", "B_th", "B"])
    col_T = _pick_column(arc_df, ["T", "Temp", "Temperature"])
    col_V = _pick_column(arc_df, ["Vz", "V_z", "V", "velocity"])

    F_B = _stack_field(arc_df, col_B, R)
    F_T = _stack_field(arc_df, col_T, R)
    F_V = _stack_field(arc_df, col_V, R)

    # col_B = _pick_column(arc_df, ["Bθ", "Btheta", "B_th", "B"])
    # col_T = _pick_column(arc_df, ["T", "Temp", "Temperature"])
    # col_V = _pick_column(arc_df, ["Vz", "V_z", "V", "velocity"])

    # X_B, F_B = _stack_field_with_x(arc_df, col_B, "Xcoord1")
    # X_T, F_T = _stack_field_with_x(arc_df, col_T, "Xcoord2")
    # X_V, F_V = _stack_field_with_x(arc_df, col_V, "Xcoord2")

    if F_B.shape[0] != Zcoord.size:
        raise ValueError(f"len(Zcoord)={Zcoord.size} does not match number of profiles NZ={F_B.shape[0]}")

    # Z2d_B = np.repeat(Zcoord[:, None], X_B.shape[1], axis=1)
    # Z2d_T = np.repeat(Zcoord[:, None], X_T.shape[1], axis=1)
    # Z2d_V = np.repeat(Zcoord[:, None], X_V.shape[1], axis=1)
    R2d, Z2d = np.meshgrid(R, Zcoord)

    # --- Magnetic field ---
    fig1, ax1 = plt.subplots()
    pcm1 = ax1.pcolormesh(
        R2d, Z2d, F_B, shading="auto",
        vmin=float(np.nanmin(F_B)), vmax=float(np.nanmax(F_B)),
        cmap=cmap
    )
    ax1.plot(Ra, Zcoord, color="red", linewidth=1.5, label="Arc shape")
    ax1.scatter(Ra, Zcoord, s=12, color="red")
    ax1.set_title("Magnetic field Bθ (T)")
    ax1.set_xlabel("R (m)")
    ax1.set_ylabel("Z (m)")
    ax1.grid(True, alpha=0.3)
    fig1.colorbar(pcm1, ax=ax1, label="Bθ (T)")
    ax1.legend(loc="best")
    fig1.tight_layout()

    # --- Temperature ---
    fig2, ax2 = plt.subplots()
    pcm2 = ax2.pcolormesh(
        R2d, Z2d, F_T, shading="auto",
        vmin=5000.0, vmax=float(np.nanmax(F_T)),
        cmap=cmap
    )
    ax2.plot(Ra, Zcoord, color="red", linewidth=1.5, label="Arc shape")
    ax2.scatter(Ra, Zcoord, s=12, color="red")
    ax2.set_title("Temperature T (K)")
    ax2.set_xlabel("R (m)")
    ax2.set_ylabel("Z (m)")
    ax2.grid(True, alpha=0.3)
    fig2.colorbar(pcm2, ax=ax2, label="T (K)")
    ax2.legend(loc="best")
    fig2.tight_layout()

    # --- Axial velocity ---
    fig3, ax3 = plt.subplots()
    pcm3 = ax3.pcolormesh(
        R2d, Z2d, F_V, shading="auto",
        vmin=0.0, vmax=float(np.nanmax(F_V)),
        cmap=cmap
    )
    ax3.plot(Ra, Zcoord, color="red", linewidth=1.5, label="Arc shape")
    ax3.scatter(Ra, Zcoord, s=12, color="red")
    ax3.set_title("Axial velocity Vz (m/s)")
    ax3.set_xlabel("R (m)")
    ax3.set_ylabel("Z (m)")
    ax3.grid(True, alpha=0.3)
    fig3.colorbar(pcm3, ax=ax3, label="Vz (m/s)")
    ax3.legend(loc="best")
    fig3.tight_layout()

    return 
