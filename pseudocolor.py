import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Union


def _as_1d_float_vector(x: Union[str, Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Convert heterogeneous input data into a one-dimensional NumPy float array.

    The function standardizes numerical data originating from CSV cells,
    Pandas DataFrames, or in-memory objects. It supports direct numerical
    containers as well as string-encoded vectors and guarantees a flattened
    one-dimensional output suitable for numerical processing and plotting.

    Parameters
    ----------
    x : str, Sequence[float], or np.ndarray
        Input data representing a numerical vector. Supported formats include:
        - NumPy arrays,
        - Python lists or tuples,
        - strings encoding numeric vectors, e.g. "[0.1 0.2 0.3]" or
          "[0.1, 0.2, 0.3]".

    Returns
    -------
    np.ndarray
        One-dimensional NumPy array of floats.

    Raises
    ------
    ValueError
        If a string input cannot be parsed into a numeric vector.
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
    """
    Select the first available column from a list of candidate names.

    The function iterates over a prioritized list of possible column names
    and returns the name of the first one that exists in the given DataFrame.
    This utility enables backward compatibility and flexible handling of
    datasets with slightly different column naming conventions.

    Parameters
    ----------
    arc_df : pandas.DataFrame
        DataFrame from which a column name must be selected.
    candidates : Sequence[str]
        Ordered list of candidate column names, ranked by preference.

    Returns
    -------
    str
        Name of the first column found in the DataFrame.

    Raises
    ------
    KeyError
        If none of the candidate column names exist in the DataFrame.
    """
    for c in candidates:
        if c in arc_df.columns:
            return c
    raise KeyError(
        f"None of the candidate columns exist: {list(candidates)}. "
        f"Available columns: {list(arc_df.columns)}"
    )


def _stack_field(arc_df, col: str, R: np.ndarray) -> np.ndarray:
    """
    Assemble a two-dimensional field from a DataFrame column of radial profiles.

    The function converts a column of one-dimensional radial profiles—stored as
    heterogeneous cell values—into a two-dimensional NumPy array with dimensions
    (N_Z × N_R), where N_Z is the number of axial positions and N_R the number of
    radial points. Strict consistency between the radial grid and profile lengths
    is enforced.

    Parameters
    ----------
    arc_df : pandas.DataFrame
        DataFrame containing radial profile data for multiple axial positions.
    col : str
        Name of the column holding the radial profiles.
    R : np.ndarray
        Radial coordinate array defining the expected profile length.

    Returns
    -------
    np.ndarray
        Two-dimensional array of shape (N_Z, N_R) containing stacked profiles.

    Raises
    ------
    ValueError
        If any profile length does not match the length of the radial grid.
    """
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


def plot_arccolumn_isolines(
    arc_df,
    Zcoord: np.ndarray,
    Ra: np.ndarray,
    cmap: str = "viridis"):
    """
    Plot two-dimensional isoline (heatmap) representations of arc column
    physical fields in the (R, Z) plane.

    The function reconstructs 2D fields of azimuthal magnetic field (Bθ),
    temperature (T), and axial velocity (Vz) from one-dimensional radial
    profiles stored in a DataFrame. Each field is displayed as a color map
    over the radial–axial domain, with the arc shape overlaid for reference.

    Color scaling is applied as follows:
        1. Magnetic field: full available data range,
        2. Temperature: lower bound fixed to a reference plasma temperature,
        3. Axial velocity: lower bound fixed to zero.

    Parameters
    ----------
    arc_df : pandas.DataFrame
        DataFrame containing arc column data. It must include columns for
        magnetic field, temperature, and axial velocity, stored as radial
        profiles for each axial position.
    Zcoord : np.ndarray
        Axial coordinate array corresponding to the rows of `arc_df`.
    Ra : np.ndarray
        Arc radius as a function of axial position, used to overlay the arc
        shape on the heatmaps.
    cmap : str, optional
        Matplotlib colormap name used for all heatmaps (default is "viridis").

    Returns
    -------
    None
        The function generates and displays three Matplotlib figures.
    """
    R = np.linspace(0, 0.01, 201)
    R = np.asarray(R).ravel()
    Zcoord = np.asarray(Zcoord).ravel()

    col_B = _pick_column(arc_df, ["Bθ", "Btheta", "B_th", "B"])
    col_T = _pick_column(arc_df, ["T", "Temp", "Temperature"])
    col_V = _pick_column(arc_df, ["Vz", "V_z", "V", "velocity"])

    F_B = _stack_field(arc_df, col_B, R)
    F_T = _stack_field(arc_df, col_T, R)
    F_V = _stack_field(arc_df, col_V, R)

    if F_B.shape[0] != Zcoord.size:
        raise ValueError(f"len(Zcoord)={Zcoord.size} does not match number of profiles NZ={F_B.shape[0]}")
    R2d, Z2d = np.meshgrid(R, Zcoord)

    # --- Magnetic field ---
    fig1, ax1 = plt.subplots()
    pcm1 = ax1.pcolormesh(
        R2d, Z2d, F_B, shading="auto",
        vmin=float(np.nanmin(F_B)), vmax=float(np.nanmax(F_B)),
        cmap=cmap)
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
        vmin=10000.0, vmax=float(np.nanmax(F_T)),
        cmap=cmap)
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
        cmap=cmap)
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
