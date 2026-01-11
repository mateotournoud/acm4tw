import numpy as np

# ---------- Core geometry ----------
def Rc_from_I(I, Jc=6.5e7):
    """
    Eq. (1): Rc = sqrt(I / (pi * Jc))
    Jc default from cathodic thermionic emission correlation.
    Source: Eq. (1)【2-51】.
    """
    return np.sqrt(I / (np.pi * Jc))

def Ra_from_Z(Z, Rc, Pr, a, b, pr_exp=0.85):
    """
    Eq. (2): (Ra/Rc) * Pr^0.85 = ln( a + b * (Z/Rc) * Pr^0.85 )
    Rearranged: Ra = Rc/Pr^0.85 * ln( a + b * (Z/Rc) * Pr^0.85 )
    Valid in the arc column, not at the anode bell or cathode fall【2-56】【2-57】.
    """
    arg = a + (b * Z * Pr**pr_exp)/Rc
    Ra = (Rc / (Pr ** pr_exp)) * np.log(arg)
    return Ra

# ---------- Maxima relations (global and local along Z) ----------
def global_max(I, L, a, b, c):
    """
    Eq. (7): PhysicalQuantity_max = a + b*I + c*L
    Works for Bmax, Tmax, Vmax (each has its own a,b,c)【2-63】.
    """
    return a + b*I + c*L

def local_max_at_Z(I, L, Z, PQ_max, a, b, c, d, e, f):
    """
    Eq. (6): PhysicalQuantity°_max(Z) = [ a + b*I + c*L + d*Z + e*PQ_max ]^f
    Where PQ is B, T, or V. Coefficients differ per variable and gas【2-63】.
    """
    base = a + b*I + c*L + d*Z + e*PQ_max
    base = base**f
    return base

# ---------- Dimensionless radial coordinates for arc-column profiles ----------
def rhat(R, Ra, Pr, p):
    """
    For Bθ profile in Eq. (3), use (R/Ra)*Pr^{-0.5}【2-58】.
    For T° profile in Eq. (4), use (R/Ra)*Pr^{-1.0}【2-58】.
    For Vz profile in Eq. (5), use (R/Ra)*Pr^{-1.0}【2-61】.
    """
    return (R / Ra) * (Pr ** p)

# ---------- Arc-column radial profiles (normalized) ----------
def Btheta_over_Bmaxo(x, a, b, c, d, e):
    """
    Eq. (3) normalized magnetic field profile Bθ/Bmax° as a function of x = rhat_B(R,Ra,Pr).
    The closed-form from the paper uses a rational form with x; write directly with Table 5 coeffs:
      Bθ/Bmax° = (a + b*x**0.5 + c*x) / (1 + d*x**0.5 + e*x)
    Confirm exponents and coefficients from Table 5【2-58】【2-60】.
    """
    return (a + b*x**0.5 + c*x) / (1 + d*x**0.5 + e*x)

def T_over_Tmaxo(x, a, b, c, d, e, f, g):
    """
    Eq. (4) normalized temperature profile T/Tmax° as a function of x = rhat_T(R,Ra,Pr).
    Reported as a Gaussian-like fit; one convenient implementation:
      T/Tmax° = a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6
    Use the exact functional form and coefficients you extracted from Table 5【2-58】【2-60】.
    """
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6

def Vz_over_Vmaxo(x, a, b, c, d, e):
    """
    Eq. (5) normalized axial velocity profile Vz/Vmax° as a function of x = rhat_V(R,Ra,Pr).
    A simple accurate representation consistent with the paper:
      Vz/Vmax° = a + b*x + c*x**2 + d*x**3
    Use your Table 5 coefficients for (a,b,c,d)【2-61】【2-60】.
    """
    return (a + b*x + c*x**2) / (1 + d*x + e*x**2)

# ---------- Anode interactions: dimensionless radius and profiles ----------
def xi_qJP(R, Ra_at_L, L, Pr, p):
    """
    This function calculates the coefficient xi = [R / (Ra(L) * L)**0.5] * Pr^{+p}.
    For heat flux, use p=0.2.
    For current density, p=0.3.
    For pressure, p=0.8.
    """
    return (R / np.sqrt(Ra_at_L * L)) * (Pr ** p)

def xi_S(R, Ra_at_L, L):
    xi = R/np.sqrt(Ra_at_L*L)
    return xi

def q_over_qmax(xi, a, b, c, d, e, f, g):
    """
    Eq. (8): q/qmax at the anode as a function of xi_q.
    Representative Gaussian-like series form (fill with your Table 5 coefficients)【2-66】:
      q/qmax = a + b/(1 + e*xi**2) + c*xi**4 + d*xi**6 + g*xi**4
    Adjust to the exact fitted structure you digitized from Table 5.
    """
    nom = a + b*xi**2 + c*xi**4 + d*xi**6
    den = 1 + e*xi**2 + f*xi**4 + g*xi**6
    return nom/den

def J_over_Jmax(xi, a, b, c, d, e, f, g, h):
    """
    Eq. (9): J/Jmax as a function of xi_J; similar Gaussian-like fit【2-66】.
    Provide the exact coefficients and terms from your Table 5 CSV.
    """
    nom = a + b*xi**2 + c*xi**4 + d*xi**6
    den = 1 + e*xi**2 + f*xi**4 + g*xi**6 + h*xi**8
    return nom/den

def P_over_Pmax(xi, a, b, c, d, e, f, g):
    """
    Eq. (10): P/Pmax as a function of xi_P; similar structure【2-66】.
    """
    nom = a + b*xi**2 + c*xi**4 + d*xi**6
    den = 1 + e*xi**2 + f*xi**4 + g*xi**6
    return nom/den

def S_over_Smax(xi, a, b, c, d, e):
    """
    Eq. (15) (Villareal-Medina et. al., 2023).
    Only applies for Argon.
    """
    num = a + b*np.log(xi) + c*np.log(xi)**2
    den = 1 + d*np.log(xi) + e*np.log(xi)**2
    return num/den

# ---------- Interaction maxima at the anode ----------
def interaction_max(I, L, a, b, c):
    """
    Eq. (11): Interaction_max = exp( a + b*I + c*L )
    Works for qmax, Jmax, Pmax (each has its own a,b,c)【2-65】【2-66】.
    """
    return np.exp(a + b*I + c*L)

def shear_stress_max(I, L, a, b, c):
    """
    Eq. (16) (Villareal-Medina et. al., 2023).
    Only applies for Argon.
    """
    return a + b*L + c*I