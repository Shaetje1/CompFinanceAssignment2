# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:58:39 2026

@author: shaew
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
r=0.02
sigma1=0.2
sigma2=0.3
S1_0=100
S2_0=100
K=100
T=1
x_0=[np.log(S1_0/K),np.log(S2_0/K)]
xi_1=[(r-0.5*sigma1**2),(r-0.5*sigma2**2)]
xi_2=[sigma1**2,sigma2**2]
a=min(x_0[0]+xi_1[0]-10*np.sqrt(xi_2[0]),x_0[1]+xi_1[1]-10*np.sqrt(xi_2[1]))
b=max(x_0[0]+xi_1[0]+10*np.sqrt(xi_2[0]),x_0[1]+xi_1[1]+10*np.sqrt(xi_2[1]))

def phi(u1,u2):
    return np.exp(-0.025j*u2*T-0.5*T*(0.04*u1**2+0.09*u2**2))
def dct2_python(matrix):
    return dct(dct(matrix, axis=0, type=2), axis=1, type=2)
print(f"{'Q=N1=N2':<6} | {'Option Price':<15}")
print("-" * 24)
Qs=[16,32,64,128,256,512,1024,2048,4096]
Values=[]
for q in Qs:
    N1=N2=Q = q #Must be geq than Max(N1,N2), so for simplicity choose them the same
    
    g_matrix = np.zeros((Q, Q)) 
    #Now we let G equal to the payoff on the values of y1 y2 on this grid
    #We need to discretize accros y values, and then give the payoff
    y_1=[a+0.5*(b-a)/Q]
    y_2=[a+0.5*(b-a)/Q]
    for i in range(0,Q-1):
        y_1.append(y_1[-1]+(b-a)/Q)
        y_2.append(y_2[-1]+(b-a)/Q)
    def g(y1,y2):
        return max(0.5*(np.exp(y1)+np.exp(y2)-2)*K,0)
    for i in range(Q):
        for j in range(Q):
            g_matrix[i, j] = g(y_1[i], y_2[j])
    # 2. Compute the 2D DCT
    dct_coefficients = dct2_python(g_matrix)
    V_DCT = dct_coefficients * (1 / (Q * Q))
    
    
    ValueAt0=0
    for k1 in range(0,N1):
        for k2 in range (0,N2):
                if k1==0 and k2==0:
                    #We use T instead of Delta t since T - 0 = T
                    ValueAt0+=0.25*np.exp(-r*T)*0.5*(((phi(k1*np.pi/(b-a),k2*np.pi/(b-a))*np.exp(1j*k1*np.pi*-a/(b-a)+1j*k2*np.pi*-a/(b-a))).real)+(phi(k1*np.pi/(b-a),-k2*np.pi/(b-a))*np.exp(1j*k1*np.pi*-a/(b-a)-1j*k2*np.pi*-a/(b-a))).real)*V_DCT[k1][k2]
                elif k1==0 or k2==0:
                    ValueAt0+=0.5*np.exp(-r*T)*0.5*(((phi(k1*np.pi/(b-a),k2*np.pi/(b-a))*np.exp(1j*k1*np.pi*-a/(b-a)+1j*k2*np.pi*-a/(b-a))).real)+(phi(k1*np.pi/(b-a),-k2*np.pi/(b-a))*np.exp(1j*k1*np.pi*-a/(b-a)-1j*k2*np.pi*-a/(b-a))).real)*V_DCT[k1][k2]
                else:
                    ValueAt0+=np.exp(-r*T)*0.5*(((phi(k1*np.pi/(b-a),k2*np.pi/(b-a))*np.exp(1j*k1*np.pi*-a/(b-a)+1j*k2*np.pi*-a/(b-a))).real)+(phi(k1*np.pi/(b-a),-k2*np.pi/(b-a))*np.exp(1j*k1*np.pi*-a/(b-a)-1j*k2*np.pi*-a/(b-a))).real)*V_DCT[k1][k2]
                    
    Values.append(ValueAt0)              
    print(f"{q:<6} | {ValueAt0:<15.6f}")
    
    
    


# Using q=4096 as the proxy for the true value
true_value = Values[-1]
errors = np.abs(Values - true_value)

plt.figure(figsize=(7, 4.5))

# Plotting on a log-y scale
plt.semilogy(Qs, errors, 'o-', color='darkblue', label='COS Absolute Error')

plt.xlabel('Number of Fourier Coefficients ($q$)', fontsize=11)
plt.ylabel('Absolute Error (Log Scale)', fontsize=11)
plt.title('Error Convergence Rate of the 2D COS Method', fontsize=12, fontweight='bold')
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()

plt.show()