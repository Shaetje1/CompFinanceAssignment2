#Computational Finance Asiignment 2, question 1a)
import numpy as np
import matplotlib.pyplot as plt
r=0.02
S_0=100 #Same for S1 and 2 so just 1 variable
Sigma1=0.2
Sigma2=0.3
rho=0.3
T=1
K=100
np.random.seed(0)

def CorrBrownian(R): #R is the amount of replications
    Z=np.random.normal(loc=0,scale=np.sqrt(T),size=(2,R)) #Could've used standard normal, but this is more general
    L=np.array([[1,0],[rho,np.sqrt(1-rho**2)]])
    CorrZ=np.matmul(L,Z)
    return CorrZ
def BasketValues(CorrZ):
    S_1=S_0*np.exp((r-0.5*Sigma1**2)*T+Sigma1*CorrZ[0])
    S_2=S_0*np.exp((r-0.5*Sigma2**2)*T+Sigma2*CorrZ[1])
    return np.exp(-r*T)*np.maximum(0.5*S_1+0.5*S_2-K,0)

def Results(lst, R):
    mean = np.mean(lst)
    SE = np.std(lst) / np.sqrt(len(lst))
    
    lower_ci = mean - 1.96 * SE
    upper_ci = mean + 1.96 * SE
    
    print(f"R: {R:<10} | Price: {mean:.4f} | SE: {SE:.6f} | 95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]")
    
    return mean, SE

SampleSizes=[10**4,10**5,10**6,10**7,10**8]
Means=[]
SEs=[]
for i in SampleSizes:
    (mean,SE)=Results(BasketValues(CorrBrownian(i)),i)
    Means.append(mean)
    SEs.append(SE)
    
ReferencePrice=Means[-1]

    
plt.loglog(SampleSizes, Means, marker='o', label='Mean Price')
plt.axhline(y=ReferencePrice, color='r', linestyle='--', label=f'Best Estimate ({ReferencePrice:.4f})')
plt.xlabel('Sample Size (R)')
plt.ylabel('Estimated Price')
plt.title('Monte Carlo Convergence')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()



log_R = np.log(SampleSizes)
log_SE = np.log(SEs)
slope, intercept = np.polyfit(log_R, log_SE, 1)

print(f"Calculated Convergence Rate: {slope:.4f}")


plt.loglog(SampleSizes, SEs, 'bo-', label=f'Empirical SE (Slope: {slope:.4f})')
theoretical_se = SEs[0] * (SampleSizes[0] / np.array(SampleSizes))**0.5
plt.loglog(SampleSizes, theoretical_se, 'r', label='Theoretical O(1/√R)')

plt.xlabel('Number of Replications (R)')
plt.ylabel('Standard Error')
plt.title('Monte Carlo Convergence Rate Analysis')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

