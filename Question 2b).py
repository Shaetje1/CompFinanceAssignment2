# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:20:00 2026

@author: Shae
"""
#Question 2a) Moment matching
import numpy as np
from scipy.stats import norm
S1_0 = 100
S2_0 = 100
r = 0.02
T = 1
K = 100
sigma1 = 0.25
sigma2 = 1.5
rho = 0.0
Tau=(1-np.exp(-r*T))/r
Labda=4*S2_0/(sigma2**2*Tau)
Scale=np.exp(r*T)*sigma2**2*Tau/4
E_S1_cu = S1_0**3 * np.exp(3*r*T + 3*sigma1**2*T)
E_S2_cu =Scale**3* Labda**3+12*Labda**2+24*Labda
E_S1_sq = (S1_0**2) * np.exp((2 * r + sigma1**2) * T)
E_S2_sq = Scale**2*Labda**2 + 4*Labda
E_S1=S1_0*np.exp(r*T)
E_S2=S2_0*np.exp(r*T)
E_S1_S2 = E_S1*E_S2
E_S1_S2_sq = E_S1*E_S2_sq
E_S1_sq_S2 = E_S1_sq*E_S2
M3 = (1/8) * (E_S1_cu + E_S2_cu) + (3/8) * (E_S1_S2_sq + E_S1_sq_S2)
M2 = 0.25 * E_S1_sq + 0.25 * E_S2_sq + 0.5 * E_S1_S2
M1 = 0.5 * S1_0 * np.exp(r * T) + 0.5 * S2_0 * np.exp(r * T)
Var=np.log(M2/(M1**2))
Vol=np.sqrt(Var)
d1=(np.log(M1/K)+0.5*Var)/Vol
d2=d1-Vol
N_d1=norm.cdf(d1)
N_d2=norm.cdf(d2)
BS_call=np.exp(-r*T)*(M1*N_d1-K*N_d2)
print(BS_call)
M1_option_price=(100*np.exp(r*T)-100)*np.exp(-r*T)
