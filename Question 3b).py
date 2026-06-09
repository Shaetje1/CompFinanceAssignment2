# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:46:57 2026

@author: shaew

"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
r=0.02
sigma1=0.25
sigma2=1.5
T=1
Tau=(1-np.exp(-r*T))/r
S1_0=100
S2_0=100
K=100
Labda=4*S2_0/(sigma2**2*Tau)
scale=np.exp(r*T)*sigma2**2*Tau/4
x_0=[np.log(S1_0/K),S2_0]
xi_1=[(r-0.5*sigma1**2),scale*Labda]
xi_2=[sigma1**2,scale**2*4*Labda]
xi_4=[0,scale**4*48*Labda/(4*Labda**2)]
a1=x_0[0]+xi_1[0]-10*np.sqrt(xi_2[0]+np.sqrt(xi_4[0]))
b1=x_0[0]+xi_1[0]+10*np.sqrt(xi_2[0]+np.sqrt(xi_4[0]))
a2=x_0[1]+xi_1[1]-10*np.sqrt(xi_2[1]+np.sqrt(xi_4[1]))
b2=x_0[1]+xi_1[1]+10*np.sqrt(xi_2[1]+np.sqrt(xi_4[1]))
def phi(u1,u2):
    phi_asset1 = np.exp(1j * u1 *(r-0.5*sigma1**2) - 0.5 * (sigma1**2) * T * u1**2)
    phi_asset2 = np.exp((1j * u2*scale * Labda) / (1.0 - 2.0 * 1j * u2*scale))
    return phi_asset1*phi_asset2
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
    y_1=[a1+0.5*(b1-a1)/Q]
    y_2=[a2+0.5*(b2-a2)/Q]
    for i in range(0,Q-1):
        y_1.append(y_1[-1]+(b1-a1)/Q)
        y_2.append(y_2[-1]+(b2-a2)/Q)
    def g(y1,y2):
        return max(0.5*(np.exp(y1)-2)*K+0.5*y2,0)
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
                    ValueAt0+=0.25*np.exp(-r*T)*0.5*(((phi(k1*np.pi/(b1-a1),k2*np.pi/(b2-a2))*np.exp(1j*k1*np.pi*-a1/(b1-a1)+1j*k2*np.pi*-a2/(b2-a2))).real)+(phi(k1*np.pi/(b1-a1),-k2*np.pi/(b2-a2))*np.exp(1j*k1*np.pi*-a1/(b1-a1)-1j*k2*np.pi*-a2/(b2-a2))).real)*V_DCT[k1][k2]
                elif k1==0 or k2==0:
                    ValueAt0+=0.5*np.exp(-r*T)*0.5*(((phi(k1*np.pi/(b1-a1),k2*np.pi/(b2-a2))*np.exp(1j*k1*np.pi*-a1/(b1-a1)+1j*k2*np.pi*-a2/(b2-a2))).real)+(phi(k1*np.pi/(b1-a1),-k2*np.pi/(b2-a2))*np.exp(1j*k1*np.pi*-a1/(b1-a1)-1j*k2*np.pi*-a2/(b2-a2))).real)*V_DCT[k1][k2]
                else:
                    ValueAt0+=np.exp(-r*T)*0.5*(((phi(k1*np.pi/(b1-a1),k2*np.pi/(b2-a2))*np.exp(1j*k1*np.pi*-a1/(b1-a1)+1j*k2*np.pi*-a2/(b2-a2))).real)+(phi(k1*np.pi/(b1-a1),-k2*np.pi/(b2-a2))*np.exp(1j*k1*np.pi*-a1/(b1-a1)-1j*k2*np.pi*-a2/(b2-a2))).real)*V_DCT[k1][k2]
                    
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