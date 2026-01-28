import matplotlib.pyplot as plt
import numpy as np

def shape_of_the_arc(Ra: np.ndarray,
                     Zcoord: np.ndarray,
                     Rc: float,
                     arclength: float,
                     gas_name: str = "Argon",
                     current: float = 200):
    """
    Visualize the geometrical shape of an electric arc by plotting the arc radius
    as a function of the axial coordinate, together with reference lines
    representing the cathode spot and the anode position.

    The function converts the input data into NumPy arrays, creates a scatter
    plot of the arc shape (Ra vs Zcoord), and overlays two horizontal reference
    lines: one at the cathode plane (Z = 0) with a width equal to the cathode
    spot radius Rc, and one at the anode plane (Z = -arclength).

    Parameters
    ----------
    Ra (np.ndarray):
        Radial position of the arc column as a function of Z (in meters).
    Zcoord (np.ndarray):
        Axial coordinate along the arc axis (in meters), with 0 at the
        cathode and negative values toward the anode.
    Rc (float):
        Radius of the cathode spot (in meters).
    arclength (float):
        Total arc length between cathode and anode (in meters).
    gas_name (str, optional):
        Name of the shielding gas or gas mixture, used in the plot title.
    current (float, optional):
        Arc current in amperes, used in the plot title.

    Returns
    -------
    None
        The function generates and displays a Matplotlib figure.
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
    Plot the normalized geometrical shape of an electric arc using Prandtl
    number scaling.

    The function applies a similarity transformation to the radial and axial
    coordinates of the arc by normalizing them with the characteristic radius
    Rc and scaling them by the Prandtl number raised to a fixed exponent.
    The resulting dimensionless coordinates are displayed as a scatter plot
    for a given gas, current, and arc length.

    Parameters
    ----------
    Ra : np.ndarray
        Radial coordinates of the arc column.
    Zcoord : np.ndarray
        Axial coordinates along the arc axis (negative values toward the anode).
    Pr_number : float
        Prandtl number of the plasma-forming gas.
    Rc : float
        Characteristic radius used for normalization.
    gas_name : str
        Name of the gas or gas mixture, used to select the plot color and title.
    current : float
        Arc current in amperes.
    arclength : float
        Total arc length in meters.

    Returns
    -------
    None
        The function generates and displays a Matplotlib figure.
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
    """
    Plot a one-dimensional interaction profile and highlight the arc radius
    position at the anode using Prandtl-scaled coordinates.

    The function computes a characteristic normalized arc radius from the
    arc radius evaluated at the anode position, scales it using the arc length
    and the Prandtl number, and displays it as a vertical reference line on
    top of a continuous profile curve.

    Parameters
    ----------
    x : np.ndarray
        Abscissa values of the interaction profile.
    y : np.ndarray
        Ordinate values of the interaction profile.
    Ra : np.ndarray
        Arc radius values along the axial direction.
    arclength : float
        Total arc length (used for normalization).
    pr_number : float
        Prandtl number of the gas.
    pr_exp : float
        Exponent applied to the Prandtl number in the normalization.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    xlim : tuple
        Lower and upper bounds of the x-axis.
    ylim : tuple
        Lower and upper bounds of the y-axis.
    title : str, optional
        Title of the plot. If provided, the corresponding x-coordinate of the
        normalized arc radius is printed to the console.

    Returns
    -------
    None
        The function generates and displays a Matplotlib figure.
    """
    Ra_at_L = Ra[-1:]
    Racoord = (Ra_at_L**0.5)/(arclength**0.5)*pr_number**pr_exp
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot(x, y, linewidth=2, color="blue", label= "Interaction profile")
    ax.vlines(Racoord, ylim[0], ylim[1], linewidth=2, color='red', label= r"R = $R_a$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.legend()
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title:
        ax.set_title(title)
        print(f"X coordinate of the {title}: {Racoord.item():.3f}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return

def plot_heat(xi_q, q_norm, Ra, arclength, pr_number, gas_name, pr_exp=-0.2):
    """
    Plot the normalized radial heat flux profile and indicate the arc radius
    position using Prandtl-number scaling.

    This function is a high-level wrapper around the internal `_plot_profile`
    utility. It configures axis labels, limits, and the Prandtl exponent
    specifically for normalized heat flux visualization, and delegates the
    actual plotting to `_plot_profile`.

    Parameters
    ----------
    xi_q : np.ndarray
        Dimensionless radial coordinate for the heat flux profile.
    q_norm : np.ndarray
        Normalized heat flux values, typically scaled by the maximum heat flux.
    Ra : np.ndarray
        Arc radius values along the axial direction.
    arclength : float
        Total arc length between cathode and anode.
    pr_number : float
        Prandtl number of the plasma-forming gas.
    gas_name : str
        Name of the gas or gas mixture, used in the plot title.
    pr_exp : float, optional
        Exponent applied to the Prandtl number in the normalization
        (default is 0.2).

    Returns
    -------
    None
        The function generates and displays a Matplotlib figure.
    """
    return _plot_profile(
        xi_q, q_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,3], ylim = [0,1.2],
        xlabel=r"$(R/\sqrt{R_a L})\,\mathrm{Pr}^{-0.2}$",
        ylabel=r"$q/q_{\max}$",
        title=f"Heat flux profile for {gas_name}"
    )

def plot_current(xi_J, J_norm, Ra, arclength, pr_number, gas_name, pr_exp=-0.3):
    """
    Plot the normalized radial current density profile and indicate the arc
    radius position using Prandtl-number scaling.

    This function is a specialized wrapper around the internal `_plot_profile`
    utility. It defines the Prandtl exponent, axis labels, limits, and title
    appropriate for current density visualization, while delegating all
    plotting operations to `_plot_profile`.

    Parameters
    ----------
    xi_J : np.ndarray
        Dimensionless radial coordinate for the current density profile.
    J_norm : np.ndarray
        Normalized current density values, typically scaled by the maximum
        current density.
    Ra : np.ndarray
        Arc radius values along the axial direction.
    arclength : float
        Total arc length between cathode and anode.
    pr_number : float
        Prandtl number of the plasma-forming gas.
    gas_name : str
        Name of the gas or gas mixture, used in the plot title.
    pr_exp : float, optional
        Exponent applied to the Prandtl number in the normalization
        (default is 0.3).

    Returns
    -------
    None
        The function generates and displays a Matplotlib figure.
    """
    return _plot_profile(
        xi_J, J_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,3], ylim = [0,1.2],
        xlabel=r"$(R/\sqrt{R_a L})\,\mathrm{Pr}^{-0.3}$",
        ylabel=r"$J/J_{\max}$",
        title=f"Current density profile for {gas_name}"
    )

def plot_pressure(xi_P, P_norm, Ra, arclength, pr_number, gas_name, pr_exp=-0.8):
    """
    Plot the normalized radial arc pressure profile and indicate the arc radius
    position using Prandtl-number scaling.

    This function is a high-level wrapper around the internal `_plot_profile`
    utility. It configures the normalization exponent, axis labels, plotting
    limits, and title specifically for arc pressure visualization, while all
    plotting operations are handled by `_plot_profile`.

    Parameters
    ----------
    xi_P : np.ndarray
        Dimensionless radial coordinate of the pressure profile.
    P_norm : np.ndarray
        Normalized arc pressure values, scaled by the maximum pressure.
    Ra : np.ndarray
        Arc radius values along the axial direction.
    arclength : float
        Total arc length between cathode and anode.
    pr_number : float
        Prandtl number of the plasma-forming gas.
    gas_name : str
        Name of the gas or gas mixture, used in the plot title.
    pr_exp : float, optional
        Exponent applied to the Prandtl number in the normalization
        (default is 0.8).

    Returns
    -------
    None
        The function generates and displays a Matplotlib figure.
    """
    return _plot_profile(
        xi_P, P_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,2.5], ylim = [0,1.2],
        xlabel=r"$(R/\sqrt{R_a L})\,\mathrm{Pr}^{-0.8}$",
        ylabel=r"$P/P_{\max}$",
        title=f"Arc pressure profile for {gas_name}"
    )

def plot_shear(xi_S, S_norm, Ra, arclength, pr_number, gas_name, pr_exp=0):
    """
    Plot the normalized radial shear stress profile and indicate the arc radius
    position without Prandtl-number scaling.

    This function is a specialized wrapper around the internal `_plot_profile`
    utility. Unlike other profile plots, the Prandtl exponent is set to zero,
    meaning that the radial coordinate normalization depends only on geometric
    scaling and not on transport properties.

    Parameters
    ----------
    xi_S : np.ndarray
        Dimensionless radial coordinate for the shear stress profile.
    S_norm : np.ndarray
        Normalized shear stress values, typically scaled by the maximum shear
        stress.
    Ra : np.ndarray
        Arc radius values along the axial direction.
    arclength : float
        Total arc length between cathode and anode.
    pr_number : float
        Prandtl number of the plasma-forming gas (included for interface
        consistency, but not used when pr_exp = 0).
    gas_name : str
        Name of the gas or gas mixture, used in the plot title.
    pr_exp : float, optional
        Exponent applied to the Prandtl number in the normalization
        (default is 0, i.e. no Prandtl scaling).

    Returns
    -------
    None
        The function generates and displays a Matplotlib figure.
    """
    return _plot_profile(
        xi_S, S_norm, Ra, arclength, pr_number, pr_exp,
        xlim = [0,3], ylim = [0,1.2],
        xlabel=r"$R/\sqrt{R_a L}$",
        ylabel=r"$S/S_{\max}$",
        title=f"Shear stress profile for {gas_name}"
    )

def _as_vector(cell):
    """
    Convert a CSV cell containing numerical data into a one-dimensional NumPy array.

    The function accepts multiple possible input formats commonly encountered
    when reading CSV files, including NumPy arrays, strings representing lists
    of numbers, or scalar-like objects. It standardizes all inputs into a
    one-dimensional NumPy array of floats.

    Parameters
    ----------
    cell : str, np.ndarray, or array-like
        A CSV cell containing numerical data, which may be represented as:
        - a NumPy array,
        - a string encoding a list of numbers (e.g. "[1, 2, 3]" or "1 2 3"),
        - or a scalar / array-like object.

    Returns
    -------
    np.ndarray
        One-dimensional NumPy array of floats extracted from the input cell.
    """
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
    Plot normalized radial profiles of magnetic field, temperature, and axial
    velocity at multiple axial positions along the arc.

    The function extracts a fixed number of axial locations (15 in total),
    including the cathode plane, the anode plane, and uniformly distributed
    interior positions. For each selected axial position, normalized radial
    profiles are plotted using Prandtl-scaled radial coordinates. Three
    separate figures are generated:
        1) Normalized azimuthal magnetic field (B/B_max) vs Xcoord1,
        2) Normalized temperature (T/T_max) vs Xcoord2,
        3) Normalized axial velocity (V/V_max) vs Xcoord2.

    Parameters
    ----------
    arc_df : pandas.DataFrame
        DataFrame containing arc simulation or experimental results. It must
        include the columns:
            - "Zcoord"
            - "Xcoord1", "Bθ/Bmax°"
            - "Xcoord2", "T/Tmax°"
            - "Vz/Vmax°"
        where the Xcoord columns may contain strings or arrays representing
        radial profiles.
    Pr_number : float
        Prandtl number of the plasma-forming gas.
    gas_name : str
        Name of the gas or gas mixture, used in the figure titles.

    Returns
    -------
    None
        The function generates and displays three Matplotlib figures.
    """
    # Z index selection
    Z = arc_df["Zcoord"].values
    nz = len(Z)

    interior_idx = np.linspace(1, nz - 2, 13, dtype=int)
    sel_idx = np.concatenate(([0], interior_idx, [nz - 1]))

    # Styles
    markers = ["o", "s", "^"]  # círculos, cuadrados, triángulos
    colors = plt.cm.viridis(np.linspace(0, 1, len(sel_idx)))

    # 1) B/Bmax° vs Xcoord1
    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    for i, idx in enumerate(sel_idx):
        x = _as_vector(arc_df.loc[idx, "Xcoord1"])
        y = _as_vector(arc_df.loc[idx, "Bθ/Bmax°"])
        ax1.scatter(x, y,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=10,
            label=f"Z = {1000*Z[idx]:.3f} mm")
    ax1.vlines(1/(Pr_number**0.5), 0, 1.2, linewidth=2, color="r", label="R = $R_a$")
    ax1.set_xlabel(r"$R/Ra*Pr^{-0.5}$")
    ax1.set_ylabel(r"$B/B_{max}^\circ$")
    ax1.set_ylim(0,1.2)
    ax1.set_xlim(0,5)
    ax1.set_title(f"Magnetic field profiles for {gas_name}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)
    fig1.tight_layout()

    # 2) T/Tmax° vs Xcoord2
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    for i, idx in enumerate(sel_idx):
        x = _as_vector(arc_df.loc[idx, "Xcoord2"])
        y = _as_vector(arc_df.loc[idx, "T/Tmax°"])
        ax2.scatter(x, y,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=10,
            label=f"Z = {1000*Z[idx]:.3f} mm")
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

    # 3) V/Vmax° vs Xcoord2
    fig3, ax3 = plt.subplots(figsize=(8, 4.8))
    for i, idx in enumerate(sel_idx):
        x = _as_vector(arc_df.loc[idx, "Xcoord2"])
        y = _as_vector(arc_df.loc[idx, "Vz/Vmax°"])
        ax3.scatter(x, y,
            color=colors[i],
            marker=markers[i % len(markers)],
            s=10,
            label=f"Z = {1000*Z[idx]:.3f} mm")
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