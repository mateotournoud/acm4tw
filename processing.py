def arc_column_characteristics(current, arclength, gas_name, Pr_number, Zcoord, Ra, table5):
    import pandas as pd
    import numpy as np
    import acm4tw

    gascomp = {"Argon": "ar", "Helium":"he", "Argon/Helium":"ar/he"}

    R = np.linspace(0, 0.01, 201)
    columnames = ["Zcoord", "Xcoord1", "Xcoord2", "Bθ/Bmax°", "Bθ", "Bmax°",
                    "T/Tmax°", "T", "Tmax°", "Vz/Vmax°", "Vz", "Vmax°"]
    arc_column = pd.DataFrame(columns = columnames)
    arc_column["Zcoord"] = Zcoord

    # Magnetic field (2x)
    row21 = table5.loc[table5["equation"] == "Btheta_over_Bmaxo"].iloc[0]
    a21, b21, c21, d21, e21 = row21["a"], row21["b"], row21["c"], row21["d"], row21["e"]
    row22 = table5.loc[(table5["equation"] == "global_max__B") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    a22, b22, c22 = row22["a"], row22["b"], row22["c"]
    Bmax_glob = acm4tw.global_max(current, arclength, a22, b22, c22)
    print(f"Global Maximum Magnetic Flux Density: Bmax = {Bmax_glob} T")

    # Temperature (3x)
    row31 = table5.loc[table5["equation"] == "T_over_Tmaxo"].iloc[0]
    a31, b31, c31, d31, e31, f31, g31 = row31["a"], row31["b"], row31["c"], row31["d"], row31["e"], row31["f"], row31["g"]
    row32 = table5.loc[(table5["equation"] == "global_max__T") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    a32, b32, c32 = row32["a"], row32["b"], row32["c"]
    Tmax_glob = acm4tw.global_max(current, arclength, a32, b32, c32)
    print(f"Global Maximum Temperature: Tmax = {Tmax_glob} K")

    # Axial arc velocity (4x)
    row41 = table5.loc[table5["equation"] == "Vz_over_Vmaxo"].iloc[0]
    a41, b41, c41, d41, e41 = row41["a"], row41["b"], row41["c"], row41["d"], row41["e"]
    row42 = table5.loc[(table5["equation"] == "global_max__V") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    a42, b42, c42 = row42["a"], row42["b"], row42["c"]
    Vmax_glob = acm4tw.global_max(current, arclength, a42, b42, c42)
    print(f"Global Maximum Axial Arc Velocity: Vmax = {Vmax_glob} m/s")

    row23 = table5.loc[(table5["equation"] == "local_max_at_Z__B") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    row33 = table5.loc[(table5["equation"] == "local_max_at_Z__T") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    row43 = table5.loc[(table5["equation"] == "local_max_at_Z__V") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    a23, b23, c23, d23, e23, f23 = row23["a"], row23["b"], row23["c"], row23["d"], row23["e"], row23["f"]
    a33, b33, c33, d33, e33, f33 = row33["a"], row33["b"], row33["c"], row33["d"], row33["e"], row33["f"]
    a43, b43, c43, d43, e43, f43 = row43["a"], row43["b"], row43["c"], row43["d"], row43["e"], row43["f"]
    for i, Zi in enumerate(Zcoord):
        Ra_i = Ra[i]
        xcoord_Btheta   = acm4tw.rhat(R, Ra_i, Pr_number, p=-0.5)
        xcoord_Temp     = acm4tw.rhat(R, Ra_i, Pr_number, p=-1.0)
        xcoord_Vz       = acm4tw.rhat(R, Ra_i, Pr_number, p=-1.0)
        Btheta_norm     = acm4tw.Btheta_over_Bmaxo(xcoord_Btheta, a21, b21, c21, d21, e21)
        Temp_norm       = acm4tw.T_over_Tmaxo(xcoord_Temp, a31, b31, c31, d31, e31, f31, g31)
        Vz_norm         = acm4tw.Vz_over_Vmaxo(xcoord_Vz, a41, b41, c41, d41, e41)
        Btheta_maxlocal = acm4tw.local_max_at_Z(current, arclength, abs(Zi), Bmax_glob, a23, b23, c23, d23, e23, f23)
        T_maxlocal      = acm4tw.local_max_at_Z(current, arclength, abs(Zi), Tmax_glob, a33, b33, c33, d33, e33, f33)
        Vz_maxlocal     = acm4tw.local_max_at_Z(current, arclength, abs(Zi), Vmax_glob, a43, b43, c43, d43, e43, f43)
        Temp_norm[Temp_norm < 0] = 0
        arc_column.at[i, "Xcoord1" ] = xcoord_Btheta
        arc_column.at[i, "Xcoord2" ] = xcoord_Temp
        arc_column.at[i, "Bθ/Bmax°"] = Btheta_norm
        arc_column.at[i, "T/Tmax°" ] = Temp_norm
        arc_column.at[i, "Vz/Vmax°"] = Vz_norm
        arc_column.at[i, "Bmax°"]    = Btheta_maxlocal
        arc_column.at[i, "Tmax°"]    = T_maxlocal
        arc_column.at[i, "Vmax°"]    = Vz_maxlocal
        arc_column.at[i, "Bθ"]       = Btheta_norm * Btheta_maxlocal
        arc_column.at[i, "T" ]       = Temp_norm * T_maxlocal
        arc_column.at[i, "Vz"]       = Vz_norm * Vz_maxlocal
    return arc_column

def arc_weld_pool_interactions(current, arclength, gas_name, Pr_number, Ra, table5):
    import numpy as np
    import acm4tw

    Ra_at_L = Ra[-1:]
    R = np.linspace(1e-10, 0.015, 101)
    gascomp = {"Argon": "ar", "Helium":"he", "Argon/Helium":"ar/he"}
    interactions = {}

    # Heat flux (5x)
    row51 = table5.loc[table5["equation"] == "q_over_qmax"].iloc[0]
    a51, b51, c51, d51, e51, f51, g51 = row51["a"], row51["b"], row51["c"], row51["d"], row51["e"], row51["f"], row51["g"]
    row52 = table5.loc[(table5["equation"] == "interaction_max__q") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    a52, b52, c52 = row52["a"], row52["b"], row52["c"]
    interactions["xi_q"] = acm4tw.xi_qJP(R, Ra_at_L, arclength, Pr_number, p=0.2)
    interactions["q_norm"] = acm4tw.q_over_qmax(interactions["xi_q"], a51, b51, c51, d51, e51, f51, g51)
    q_max = acm4tw.interaction_max(current, arclength, a52, b52, c52)
    interactions["q_profile"] = interactions["q_norm"] * q_max
    print(f"Maximum heat density: q_max = {q_max/1000:.3e} kW/m^2")

    # Current Density (6x)
    row61 = table5.loc[table5["equation"] == "J_over_Jmax"].iloc[0]
    a61, b61, c61, d61, e61, f61, g61, h61 = row61["a"], row61["b"], row61["c"], row61["d"], row61["e"], row61["f"], row61["g"], row61["h"]
    row62 = table5.loc[(table5["equation"] == "interaction_max__J") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    a62, b62, c62 = row62["a"], row62["b"], row62["c"]
    interactions["xi_J"] = acm4tw.xi_qJP(R, Ra_at_L, arclength, Pr_number, p=0.3)
    interactions["J_norm"] = acm4tw.J_over_Jmax(interactions["xi_J"], a61, b61, c61, d61, e61, f61, g61, h61)
    J_max = acm4tw.interaction_max(current, arclength, a62, b62, c62)
    interactions["J_profile"] = interactions["J_norm"] * J_max
    print(f"Maximum current density: J_max = {J_max/1000:.3e} kA/m^2")

    # Pressure (7x)
    row71 = table5.loc[table5["equation"] == "P_over_Pmax"].iloc[0]
    a71, b71, c71, d71, e71, f71, g71 = row71["a"], row71["b"], row71["c"], row71["d"], row71["e"], row71["f"], row71["g"]
    row72 = table5.loc[(table5["equation"] == "interaction_max__P") & (table5["gas"] == gascomp[gas_name])].iloc[0]
    a72, b72, c72 = row72["a"], row72["b"], row72["c"]
    interactions["xi_P"] = acm4tw.xi_qJP(R, Ra_at_L, arclength, Pr_number, p=0.8)
    interactions["P_norm"] = acm4tw.P_over_Pmax(interactions["xi_P"], a71, b71, c71, d71, e71, f71, g71)
    P_max = acm4tw.interaction_max(current, arclength, a72, b72, c72)
    interactions["P_profile"] = interactions["P_norm"] * P_max
    print(f"Maximum arc pressure: P_max = {P_max/1000:.3f} kPa")

    # Shear stress (8x)
    if gas_name == "Argon":
        row81 = table5.loc[table5["equation"] == "S_over_Smax"].iloc[0]
        a81, b81, c81, d81, e81 = row81["a"], row81["b"], row81["c"], row81["d"], row81["e"]
        row82 = table5.loc[(table5["equation"] == "interaction_max__S") & (table5["gas"] == gascomp[gas_name])].iloc[0]
        a82, b82, c82 = row82["a"], row82["b"], row82["c"]
        interactions["xi_S"] = acm4tw.xi_S(R, Ra_at_L, arclength)
        interactions["S_norm"] = acm4tw.S_over_Smax(interactions["xi_S"], a81, b81, c81, d81, e81)
        S_max = acm4tw.shear_stress_max(current, arclength, a82, b82, c82)
        interactions["S_profile"] = interactions["S_norm"] * S_max
        print(f"Maximum shear stress: tau_max = {S_max/1000:.3e} kPa")

    return interactions