# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:11:23 2026

@author: shaew
"""
#Computational Finance Assignment 2 question 1b)
#This code uses a lot of memory, if needed cut of some entries in line 47 and 108
import numpy as np
import matplotlib.pyplot as plt
r=0.02
S_0=100 #Same for S1 and 2 so just 1 variable
Sigma1=0.25
Sigma2=1.5
kappa=0.015
Theta=100
rho=0.1
T=1
K=100


def CorrBrownian(dt,R): #R is the amount of replications
    np.random.seed(5)
    N=int(T/dt)
    Z=np.random.normal(loc=0,scale=np.sqrt(dt),size=(2,N*R)) 
    L=np.array([[1,0],[rho,np.sqrt(1-rho**2)]])
    CorrZ=np.matmul(L,Z)
    return CorrZ.reshape(2,N,R) #We return a 3D matrix since thats the amount of information we need

def Simulate(CorrZ):
    N=CorrZ.shape[1]
    dt=T/N
    R=CorrZ.shape[2]
    S_1=np.zeros((N+1,R))
    S_2=np.zeros((N+1,R))
    S_1[0]=np.array([S_0]*R)
    S_2[0]=np.array([S_0]*R)
    for i in range(0,N):
        dW_1=CorrZ[0][i]
        dW_2=CorrZ[1][i]
        S_1[i+1]=S_1[i]*np.exp((r-0.5*Sigma1**2)*dt+Sigma1*dW_1)
        S_2[i+1]=S_2[i]+r*S_2[i]*dt+Sigma2*np.sqrt(np.maximum(S_2[i],0))*dW_2
        #We use the truncation method to make sure the sqrt doesnt go negative
    return np.exp(-r*T)*np.maximum(0.5*S_1[-1]+0.5*S_2[-1]-K,0)

DTs=[0.01]
Rs=[10**4,10**5,10**6,5*10**6,10**7]
print("fixed dt")
print(f"{'Replications':<15} {'dt':<10} {'Mean Price':<15} {'Std Error':<12} {'95% CI':<20}")
print("-" * 67)
means=[]
SEs=[]
CIs=[]
for i in Rs:
    for j in DTs:
        DiscountedPayoffs=Simulate(CorrBrownian(j,i))
        Mean=np.mean(DiscountedPayoffs)
        means.append(Mean)
        SE=np.std(DiscountedPayoffs)/np.sqrt(i)
        SEs.append(SE)
        lower_ci = Mean - 1.96 * SE
        upper_ci = Mean + 1.96 * SE
        CIs.append(SE*1.96)

        print(f"{i:<15} {j:<10.3f} {Mean:<15.4f} {SE:<12.4f} [{lower_ci:.4f}, {upper_ci:.4f}]")


plt.figure(figsize=(10, 6))

# Use log-log scale to see the 1/sqrt(R) relationship clearly
plt.loglog(Rs, SEs, 'o-', label=f'Standard Error (dt={DTs[0]})',alpha=0.6)

# Optional: Plot a reference line for 1/sqrt(R) to show perfect convergence
# This helps prove your code is statistically "correct"
reference_line = [SEs[0] * np.sqrt(Rs[0]) / np.sqrt(r) for r in Rs]
plt.loglog(Rs, reference_line, '--', color='red', alpha=0.6, label='Theoretical 1/√R')

plt.xlabel('Number of Replications (R)')
plt.ylabel('Standard Error (SE)')

plt.title('Monte Carlo Convergence: Standard Error vs. Replications, dt=0.01')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()
log_R = np.log(Rs)
log_SE = np.log(SEs)
slope, intercept = np.polyfit(log_R, log_SE, 1)

print(f"Calculated Convergence Rate: {slope:.4f}")
ReferencePrice=means[-1]
plt.figure(figsize=(10, 8))
plt.show()
plt.loglog(Rs, means, marker='o', label='Mean Price')
plt.errorbar(Rs, means, yerr=CIs, fmt='-o', capsize=5, label='Mean Price w/ 95% CI')
plt.axhline(y=ReferencePrice, color='r', linestyle='--', label=f'Best Estimate ({ReferencePrice:.4f})')
plt.xlabel('Sample Size (R)')
plt.ylabel('Estimated Price')
plt.title('Monte Carlo Convergence')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()






DTs=[1,0.5,0.25,0.1,0.05,0.1**2,0.1**2*0.5,0.1**3]
Rs=[10**6]
print("fixed R")
print(f"{'Replications':<15} {'dt':<10} {'Mean Price':<15} {'Std Error':<12} {'95% CI':<20}")
print("-" * 67)
means=[]
SEs=[]
CIs=[]
for i in Rs:
    for j in DTs:
        DiscountedPayoffs=Simulate(CorrBrownian(j,i))
        Mean=np.mean(DiscountedPayoffs)
        means.append(Mean)
        SE=np.std(DiscountedPayoffs)/np.sqrt(i)
        SEs.append(SE)
        lower_ci = Mean - 1.96 * SE
        upper_ci = Mean + 1.96 * SE
        CIs.append(1.96*SE)
        print(f"{i:<15} {j:<10.3f} {Mean:<15.4f} {SE:<12.4f} [{lower_ci:.4f}, {upper_ci:.4f}]")

ReferencePrice=means[-1]
plt.figure(figsize=(12, 8))
plt.show()
plt.gca().invert_xaxis()
plt.errorbar(DTs, means, yerr=CIs, fmt='-o', capsize=5, label='Mean Price w/ 95% CI')
plt.loglog(DTs, means, marker='o', label='Mean Price')
plt.axhline(y=ReferencePrice, color='r', linestyle='--', label=f'Best Estimate ({ReferencePrice:.4f})')
plt.xlabel('dt')
plt.ylabel('Estimated Price')
plt.title('time discretization error')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()
        
