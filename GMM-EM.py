import pandas as pd
import numpy as np
data=pd.read_csv(filepath_or_buffer='西瓜数据集4-0.csv',sep=',')[['密度','含糖率']].values

def caculProb(x,mean,vari):
    n=len(x)
    #x=np.mat(x)
    #mean=np.mat(mean)
    #vari=np.mat(vari)
    p=1/((2*np.pi)**(n/2)*np.linalg.det(np.mat(vari))**0.5)*np.exp(-np.mat(np.dot((x-mean),np.linalg.pinv(vari)))*np.mat((x-mean)).T/2)
    return float(p)

k=3 #the numbers of cluster
alpha=1/k*np.ones((k,)) #the probability of hiddlen variable
m,b=np.shape(data)
u=np.zeros((k,b))
sigma=np.zeros((k,b,b))
for i in range(b):
    u[:,i]=np.linspace(np.amin(data,axis=0)[i],np.amax(data,axis=0)[i],num=2*k+1)[1::2]
    #u[:,i]=(np.amax(data,axis=0)[i]-np.amin(data,axis=0)[i])/2+np.amin(data,axis=0)[i]

u[0,:]=data[5]
u[1,:]=data[21]
u[2,:]=data[26]

for i in range(k):
    sigma[i,:,:]=0.1*np.eye(b)

caculProb(data[0],u[0],sigma[0])


P=np.zeros((m,k))
maxliter=100
for q in range(maxliter):
    for j in range(m):
        for i in range(k):
            P[j,i]=alpha[i]*caculProb(data[j],u[i,:],sigma[i])
        temp=np.sum(P[j,:])
        P[j,:]=P[j,:]/temp
    print(P)

    for i in range(k):
        u[i,:]=np.sum([data[j]*P[j,i] for j in range(m)],axis=0)/np.sum(P[:,i])
        sigma[i]=np.sum([np.dot(np.mat(data[j]-u[i]).T,np.mat(data[j]-u[i]))*P[j,i] \
                         for j in range(m)],axis=0)/np.sum(P[:,i])#+1e-3*np.eye(b)
        alpha[i]=np.sum(P[:,i])/m

print('he')
print(P)
import matplotlib.pyplot as plt
'''
from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3,covariance_type='full',random_state=0)
gmm.fit(data)
y=gmm.predict(data)
'''

y=np.argmax(P,axis=1)

X1=[]
X2=[]
X3=[]
for a,b in zip(data,y):
    if b==0:
        X1.append(a)
    elif b==1:
        X2.append(a)
    else:
        X3.append(a)

X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
plt.scatter(X1[:,0],X1[:,1],color='b')
plt.scatter(X2[:,0],X2[:,1],color='r')
plt.scatter(X3[:,0],X3[:,1],color='g')
plt.show()
