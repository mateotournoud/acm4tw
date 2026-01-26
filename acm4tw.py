import numpy as np

# ---------- Core geometry ----------
def Rc_from_I(I, Jc=6.5e7):
    """
    Eq. (1) (Delgado-Álvarez et. al., 2021).
    Rc = sqrt(I / (pi * Jc))
    """
    return np.sqrt(I / (np.pi * Jc))

def Ra_from_Z(Z, Rc, Pr, a, b, pr_exp=0.85):
    """
    Eq. (2) (Delgado-Álvarez et. al., 2021).
    (Ra/Rc) * Pr^0.85 = ln( a + b * (Z/Rc) * Pr^0.85 )
    Rearranged: Ra = Rc/Pr^0.85 * ln( a + b * (Z/Rc) * Pr^0.85 )
    """
    arg = a + (b * Z * Pr**pr_exp)/Rc
    Ra = (Rc / (Pr**pr_exp)) * np.log(arg)
    return Ra

# ---------- Maxima relations (global and local along Z) ----------
def local_max_at_Z(I, L, Z, PQ_max, a, b, c, d, e, f):
    """
    Eq. (6) (Delgado-Álvarez et. al., 2021).
    PhysicalQuantity°_max(Z) = [ a + b*I + c*L + d*Z + e*PQ_max ]^f
    """
    base = a + b*I + c*L + d*Z + e*PQ_max
    base = base**f
    return base

def global_max(I, L, a, b, c):
    """
    Eq. (7) (Delgado-Álvarez et. al., 2021).
    PhysicalQuantity_max = a + b*I + c*L
    """
    return a + b*I + c*L

def local_max_at_Z_Ar1(L, Z, PQ_max, Rc, a, b, c, d, e, f, g):
    """
    Eq. (7) from Table 1 (Delgado-Álvarez et. al., 2019).
    Calculation of the local maxima for Argon gas (Z/L < 0.3).
    """
    x = Z/(Rc*L)**0.5
    return PQ_max * (a+b*x**0.5-c*x+d*x**1.5-e*x**2+f*x**2.5-g*x**3)

def local_max_at_Z_Ar2(I, L, Z, Rc, a, b, c, d, e, f, g, h):
    """
    Eq. (8) from Table 1 (Delgado-Álvarez et. al., 2019).
    Calculation of the local maxima for Argon gas (Z/L ≥ 0.3).
    """
    if Z == 0:
        Z = 1e-10  # Avoid log(0)
    if Z == L:
        Z = L * 0.9999  # Avoid log(1)=0
    x = np.log(Z/L)
    arg = a + b*x + c*x**-1 + d*x**2 + e*x**-2 + f*x**3 + g*x**-3 + h*x**4
    return Rc**0.5 * I**0.1 * L**-0.2 * arg

# ---------- Dimensionless radial coordinates for arc-column profiles ----------
def rhat(R, Ra, Pr, p):
    """
    Returns the dimensionless radius r̂ = (R/Ra) * Pr^{p} used in arc-column radial profiles.
    For Bθ profile in Eq. (3) (Delgado-Álvarez et. al., 2021)., use p = -0.5}.
    For T° profile in Eq. (4) (Delgado-Álvarez et. al., 2021), use p = -1.0}.
    For Vz profile in Eq. (5) (Delgado-Álvarez et. al., 2021), use p = -1.0}.
    """
    return (R / Ra) * (Pr ** p)

# ---------- Arc-column radial profiles (normalized) ----------
def Btheta_over_Bmaxo(x, a, b, c, d, e):
    """
    Eq. (3) (Delgado-Álvarez et. al., 2021).
    Bθ/Bmax° = (a + b*x**0.5 + c*x) / (1 + d*x**0.5 + e*x)
    """
    return (a + b*x**0.5 + c*x) / (1 + d*x**0.5 + e*x)

def T_over_Tmaxo(x, a, b, c, d, e, f, g):
    """
    Eq. (4) (Delgado-Álvarez et. al., 2021).
    T/Tmax° = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6
    """
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6

def T_over_Tmaxo_Ar(R, Ra, a, b, c, d, e, f, g, h, i, j):
    """
    Eq. (4) (Delgado-Álvarez et. al., 2019)
    Used for modelling Argon temperature profile.
    """
    x = R / Ra
    return a + b*x**2 + c*x**4 + d*x**6 + e*x**8 + f*x**10 + g*x**12 + h*x**14 + i*x**16 + j*x**18

def Vz_over_Vmaxo(x, a, b, c, d, e):
    """
    Eq. (5) (Delgado-Álvarez et. al., 2021).
    Vz/Vmax° = (a + b*x + c*x**2) / (1 + d*x + e*x**2)
    """
    return (a + b*x + c*x**2) / (1 + d*x + e*x**2)

# ---------- Anode interactions: dimensionless radius and profiles ----------
def xi_qJP(R, Ra_at_L, L, Pr, p):
    """
    Returns the dimensionless radius r = R/[(Ra(L)*L]**0.5*Pr**p. (Delgado-Álvarez et. al., 2021).
    Use p = 0.2 for anode heat flux, p = 0.3 for current density, or p = 0.8 for arc pressure.
    """
    return (R / np.sqrt(Ra_at_L * L)) * (Pr ** p)

def xi_S(R, Ra_at_L, L):
    """ 
    Returns the dimensionless radius xi_S = R/[(Ra(L)*L]**0.5 for shear stress.
    (Delgado-Álvarez et. al., 2019).
    """
    xi = R/np.sqrt(Ra_at_L*L)
    return xi

def q_over_qmax(xi, a, b, c, d, e, f, g):
    """
    Eq. (8) (Delgado-Álvarez et. al., 2021).
    q/qmax at the anode as a function of xi_q.
    """
    nom = a + b*xi**2 + c*xi**4 + d*xi**6
    den = 1 + e*xi**2 + f*xi**4 + g*xi**6
    return nom/den

def J_over_Jmax(xi, a, b, c, d, e, f, g, h):
    """
    Eq. (9) (Delgado-Álvarez et. al., 2021).
    J/Jmax as a function of xi_J.
    """
    nom = a + b*xi**2 + c*xi**4 + d*xi**6
    den = 1 + e*xi**2 + f*xi**4 + g*xi**6 + h*xi**8
    return nom/den

def P_over_Pmax(xi, a, b, c, d, e, f, g):
    """
    Eq. (10) (Delgado-Álvarez et. al., 2021).
    P/Pmax as a function of xi_P.
    """
    nom = a + b*xi**2 + c*xi**4 + d*xi**6
    den = 1 + e*xi**2 + f*xi**4 + g*xi**6
    return nom/den

def S_over_Smax(xi, a, b, c, d, e):
    """
    Eq. (15) (Villareal-Medina et. al., 2023).
    Applied to calculate the shear stress profile at the anode when modelling 100% Argon shield.
    """
    num = a + b*np.log(xi) + c*np.log(xi)**2
    den = 1 + d*np.log(xi) + e*np.log(xi)**2
    return num/den

# ---------- Interaction maxima at the anode ----------
def interaction_max(I, L, a, b, c):
    """
    Eq. (7) (Delgado-Álvarez et. al., 2021).
    Interaction_max = exp( a + b*I + c*L )
    """
    return np.exp(a + b*I + c*L)

def shear_stress_max(I, L, a, b, c):
    """
    Eq. (16) (Villareal-Medina et. al., 2023).
    Interaction_max = exp( a + b*L + c*I )
    Applied to calculate maximum shear stress at the anode when modelling 100% Argon shield.
    """
    return a + b*L + c*I