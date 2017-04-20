import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
def loaddata():
    file_object=open(os.getcwd()+'\data\ex2data1.txt')
    X=[]
    y=[]
    for line in file_object:
        a,b,c=line.strip().split(',')
        X.append([float(a),float(b),1])
        y.append(float(c))
    X=np.array(X)
    y=np.array(y)
    #print(X)
    return X,y

def sigmoid(z):
    return 1/(1+np.exp(-z))

#theta=np.zeros((3,))
def cost(theta,X,y):
    return np.sum(-y*np.dot(X,theta)+np.log(1+np.exp(np.dot(X,theta))))

def gradient(theta,X,y):
    return -np.array(np.dot(np.matrix(y),X)).reshape((3,))\
    +np.dot(X.T,(np.exp(np.dot(X,theta))/(1+np.exp(np.dot(X,theta)))))

def predict(theta,X):
    return np.array([1 if a>0.5 else 0 for a in sigmoid(np.dot(X,theta))])



'''
os.chdir('D:\sugoudownload\machine \
learning in python materials\ipython-notebooks-master\ipython-notebooks-master')
'''
print(os.getcwd())
X,y=loaddata()
X1=[]
X2=[]

for a,b in zip(X,y):
    if b==1:
        X1.append(a)
    else:
        X2.append(a)
X1=np.array(X1)
X2=np.array(X2)

theta=np.zeros((3,))

'''
plt.figure(figsize=(12,8))
plt.plot(X1[:,0],X1[:,1],'o')
plt.plot(X2[:,0],X2[:,1],'r+',markersize=12)
plt.show()
'''
c=cost(theta,X,y)
d=gradient(theta,X,y)
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
yp=predict(result[0],X)

correct=np.sum([1 if a==b else 0 for (a,b) in zip(y,yp)])/len(y)

print(c)
print(d)

print("correct rate is {0}".format(correct))