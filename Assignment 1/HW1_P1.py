import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Take Data As Pandas
df = pd.read_csv('data.txt', sep = '\s+')
D = df.drop('Index', 1).as_matrix() # 60 x 16 
n = D.shape[0]      # n = 60
p = D.shape[1]-1    # p = 15

# Normalize
nD = (D - D.min(0))/(D.max(0)-D.min(0))

# Create X and y
X = np.append(np.ones((n,1)),nD[:,0:p], axis = 1)
y = nD[:,p]

# Training Testing Split 
Xtr = X[0:48,:]
Xts = X[48:60,:]


ytr = y[0:48]
yts = y[48:60]

ntr = ytr.shape[0]
nts = yts.shape[0]

# Initiate : Gradient Descent
theta = np.zeros(p+1,) 
alp = 0.01
lam = 1
eps = 0.1

# Gradient Descent
itr = 0
J = []
J.append(np.dot(ytr,ytr)/(2*ntr))
# 
while(1):
    ytrh = np.matmul(Xtr,theta)
    T = ytrh - ytr
    update = np.zeros(p+1)
    for i in range(0,p+1):
        update[i] = np.dot(T,Xtr[:,i]) + lam*theta[i]
    theta = theta - (alp/ntr)*update
    T2 = np.matmul(Xtr,theta)-ytr
    J.append((np.dot(T2,T2)+lam*np.dot(theta,theta))/(2*ntr))
    itr = itr + 1
    dcost = 100*abs((J[itr-1]-J[itr])/J[itr-1])
    if dcost < eps:
        break;
        
for i in range(0,p+1):
    if abs(theta[i]) < 5e-3:
        theta[i] = 0
        
# Plot Loss Function
plt.plot(np.array(J))
plt.xlabel('Number of Iterations ($k$)')   
plt.ylabel('Loss Function ( $J_k$)')   
plt.title('Ridge Regression')
plt.savefig('F1.eps', format='eps')

# Testing Data
Ts = np.matmul(Xts,theta) - yts
tse = np.dot(Ts,Ts)/nts    
np.sum(theta > 0)