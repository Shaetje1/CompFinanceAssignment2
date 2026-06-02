#Computational Finance Asiignment 2, question 1a)
import numpy as np
r=0.02
S_0=100 #Same for S1 and 2 so just 1 variable
Sigma1=0.2
Sigma2=0.3
rho=0.3
T=1

def CorrBrownian(R): #R is the amount of replications
    Z=np.random.normal(loc=0,scale=np.sqrt(T),size=(2,R)) #Could've used standard normal, but this is more general
    L=np.array([[1,0],[rho,np.sqrt(1-rho**2)]])
    CorrZ=np.matmul(L,Z)
    return CorrZ

