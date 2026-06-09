#Question 2a) Moment matching
import numpy as np
from scipy.stats import norm
S1_0 = 100
S2_0 = 100
r = 0.02
T = 1
K = 100
sigma1 = 0.2
sigma2 = 0.3
rho = 0.3
E_S1_sq = (S1_0**2) * np.exp((2 * r + sigma1**2) * T)
E_S2_sq = (S2_0**2) * np.exp((2 * r + sigma2**2) * T)
E_S1_S2 = (S1_0 * S2_0) * np.exp((2 * r + rho * sigma1 * sigma2) * T)

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









