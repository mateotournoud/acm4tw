import matplotlib.pyplot as plt
import numpy as np

def shape_of_the_arc(Ra: np.ndarray,
                     Zcoord: np.ndarray,
                     Rc: float,
                     arclength: float,
                     gas_name: str = "Argon",
                     current: float = 200):
    """
    Plot the arc shape as a scatter of Ra (x) vs Z (y) with requested reference lines.

    Parameters:
        Ra        : array-like of arc radius values (mm). May contain NaNs.
        Zcoord    : array-like of axial positions (mm); typically from 0 at the cathode down to -arclength at the anode.
        Rc        : cathode spot radius (mm).
        arclength : arc length (mm).
        gas_name  : name of the shielding gas or mixture (for title).
        current   : arc current (A) (for title).

    Returns:
        ax: matplotlib Axes object with the plot.
    """
    # Ensure arrays and mask out invalid values
    Ra = np.asarray(Ra, dtype=float)
    Zcoord = np.asarray(Zcoord, dtype=float)

    # Create figure and axes
    plt.figure()
    ax = plt.subplot()

    # Scatter with red dots
    ax.scatter(Ra, Zcoord, s=12, marker='o', color='red', label='Arc shape')
    
    # Reference lines
    ax.hlines(y=0.0, xmin=0.0, xmax=float(Rc), linewidth=3, colors='green', label='Spot Cathode Rc')
    ax.hlines(y=-float(arclength), xmin=0.0, xmax=0.01, linewidth=3, colors='blue', label='Anode')

    # Labels and title
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_xlim(0,0.01)
    ax.set_title(f"Shape of the arc for {gas_name}, {current} A and {arclength} m")

    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return

def normalized_shape_of_the_arc(Ra: np.ndarray,
                                Zcoord: np.ndarray,
                                Pr_number: float,
                                Rc: float,
                                gas_name: str,
                                current: float,
                                arclength: float):
    """
    Plot the normalized arc shape for a single current.

    Parameters
    ----------
    Ra : np.ndarray
        Radial coordinates (same length as Zcoord).
    Zcoord : np.ndarray
        Axial coordinates (negative downwards).
    Pr_number : float
        Prandtl number.
    Rc : float
        Characteristic radius.
    gas_name : str
        Name of the gas (for the title).
    current : float
        Arc current in amperes.
    arclength : float
        Arc length in meters.
    """

    pr_exp = 0.85
    color = {"Argon": "red", "Helium": "orange", "Argon/Helium": "green"}

    # Convert to numpy arrays
    Ra = np.asarray(Ra, dtype=float)
    Zcoord = np.asarray(Zcoord, dtype=float)

    if Ra.shape != Zcoord.shape:
        raise ValueError("Ra and Zcoord must have the same shape.")

    # Normalize coordinates
    X = (Ra / Rc) * (Pr_number ** pr_exp)
    Y = (Zcoord / Rc) * (Pr_number ** pr_exp)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X, Y, s=8, marker='D', color=color[gas_name])
    ax.set_xlabel(r"(Ra/Rc) Pr$^{0.85}$")
    ax.set_ylabel(r"(Z/Rc) Pr$^{0.85}$")
    ax.set_ylim(-5, 0)
    ax.set_xlim(0.5, 3)
    ax.set_title(f"Normalized shape of the arc for {gas_name}, {current} A and {arclength} m")
    return


def _plot_profile(x, y, Ra, arclength, pr_number, pr_exp, xlabel, ylabel, xlim, ylim, title=None):
    Ra_at_L = Ra[-1:]
    Racoord = (Ra_at_L**0.5)/(arclength**0.5)*pr_number**pr_exp
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, y, linewidth=2, color="blue", label= "Interaction profile")
    # ax.vlines(Racoord, ylim[0], ylim[1], linewidth=2, color='red', label= r"R = $R_a$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title:
        ax.set_title(title)
        print(f"X coordinate of the {title}: {Racoord.item():.3f}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return

def plot_heat(xi_q, q_norm, Ra, arclength, pr_number, gas_name, pr_exp=0.2):
    return _plot_profile(
        xi_q, q_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,3], ylim = [0,1.2],
        xlabel=r"$(R/\sqrt{R_a L})\,\mathrm{Pr}^{0.2}$",
        ylabel=r"$q/q_{\max}$",
        title=f"Heat flux profile for {gas_name}"
    )

def plot_current(xi_J, J_norm, Ra, arclength, pr_number, gas_name, pr_exp=0.3):
    return _plot_profile(
        xi_J, J_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,3], ylim = [0,1.2],
        xlabel=r"$(R/\sqrt{R_a L})\,\mathrm{Pr}^{0.3}$",
        ylabel=r"$J/J_{\max}$",
        title=f"Current density profile for {gas_name}"
    )

def plot_pressure(xi_P, P_norm, Ra, arclength, pr_number, gas_name, pr_exp=0.8):
    return _plot_profile(
        xi_P, P_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,2.5], ylim = [0,1.2],
        xlabel=r"$(R/\sqrt{R_a L})\,\mathrm{Pr}^{0.8}$",
        ylabel=r"$P/P_{\max}$",
        title=f"Arc pressure profile for {gas_name}"
    )

def plot_shear(xi_S, S_norm, Ra, arclength, pr_number, gas_name, pr_exp=0):
    return _plot_profile(
        xi_S, S_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,3], ylim = [0,1.2],
        xlabel=r"$R/\sqrt{R_a L}$",
        ylabel=r"$S/S_{\max}$",
        title=f"Shear stress profile for {gas_name}"
    )

def _as_vector(cell):
    """Convierte una celda del CSV (string o array) en un np.ndarray 1D."""
    if isinstance(cell, np.ndarray):
        return cell.astype(float)

    if isinstance(cell, str):
        s = cell.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.replace(",", " ")
        return np.fromstring(s, sep=" ")

    return np.asarray(cell, dtype=float)


def plot_normalized_profiles(arc_df, Pr_number, gas_name):
    """
    Grafica:
      - B/Bmax° vs Xcoord1
      - T/Tmax° vs Xcoord2
      - V/Vmax° vs Xcoord2

    usando 15 perfiles en Z:
      - primero (Z=0)
      - último (Z=min)
      - 13 interiores equiespaciados
    """

    # ---------------------------
    # Selección de índices en Z
    # ---------------------------
    Z = arc_df["Zcoord"].values
    nz = len(Z)

    interior_idx = np.linspace(1, nz - 2, 13, dtype=int)
    sel_idx = np.concatenate(([0], interior_idx, [nz - 1]))

    # ---------------------------
    # Estilos
    # ---------------------------
    markers = ["o", "s", "^"]  # círculos, cuadrados, triángulos
    colors = plt.cm.viridis(np.linspace(0, 1, len(sel_idx)))

    # ===========================
    # 1) B/Bmax° vs Xcoord1
    # ===========================
    fig1, ax1 = plt.subplots(figsize=(8, 4.8))

    for i, idx in enumerate(sel_idx):
        x = _as_vector(arc_df.loc[idx, "Xcoord1"])
        y = _as_vector(arc_df.loc[idx, "Bθ/Bmax°"])

        ax1.scatter(
            x, y,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=10,
            label=f"Z = {1000*Z[idx]:.3f} mm"
        )

    ax1.vlines(1/(Pr_number**0.5), 0, 1.2, linewidth=2, color="r", label="R = $R_a$")
    ax1.set_xlabel(r"$R/Ra*Pr^{-0.5}$")
    ax1.set_ylabel(r"$B/B_{max}^\circ$")
    ax1.set_ylim(0,1.2)
    ax1.set_xlim(0,5)
    ax1.set_title(f"Magnetic field profiles for {gas_name}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)
    fig1.tight_layout()

    # ===========================
    # 2) T/Tmax° vs Xcoord2
    # ===========================
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))

    for i, idx in enumerate(sel_idx):
        x = _as_vector(arc_df.loc[idx, "Xcoord2"])
        y = _as_vector(arc_df.loc[idx, "T/Tmax°"])

        ax2.scatter(
            x, y,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=10,
            label=f"Z = {1000*Z[idx]:.3f} mm"
        )

    ax2.vlines(1/Pr_number, 0, 1.2, linewidth=2, color="r", label="R = $R_a$")
    ax2.vlines(0, 0, 1.1, linewidth=1.5, color="blue", linestyles="--")
    ax2.set_xlabel(r"$R/Ra*Pr^{-1}$")
    ax2.set_ylabel(r"$T/T_{max}^\circ$")
    ax2.set_ylim(0,1.2)
    ax2.set_xlim(0,3)
    ax2.set_title(f"Temperature profiles for {gas_name}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)
    fig2.tight_layout()

    # ===========================
    # 3) V/Vmax° vs Xcoord2
    # ===========================
    fig3, ax3 = plt.subplots(figsize=(8, 4.8))

    for i, idx in enumerate(sel_idx):
        x = _as_vector(arc_df.loc[idx, "Xcoord2"])
        y = _as_vector(arc_df.loc[idx, "Vz/Vmax°"])

        ax3.scatter(
            x, y,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=10,
            label=f"Z = {1000*Z[idx]:.3f} mm"
        )

    ax3.vlines(1/Pr_number, 0, 1.2, linewidth=2, color="r", label="R = $R_a$")
    ax3.set_xlabel(r"$R/Ra*Pr^{-1}$")
    ax3.set_ylabel(r"$V/V_{max}^\circ$")
    ax3.set_xlim(0,6)
    ax3.set_ylim(0,1.2)
    ax3.set_title(f"Axial velocity profiles for {gas_name}")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)
    fig3.tight_layout()
    return