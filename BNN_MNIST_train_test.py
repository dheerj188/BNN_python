import numpy as np
import random
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

%matplotlib inline
dataset = MNIST(root='data/',download=True)
dataset_n=[]
for i in range(len(dataset)):
    dataset_x=np.array(dataset[i][0]).reshape(784,1)
    dataset_x = dataset_x - np.mean(dataset_x)
    #dataset_x = dataset_x/(np.sum((dataset_x-np.mean(dataset_x))**2))
    dataset_x = np.array([-1 if x<0 else 1 for x in dataset_x])
    dataset_x = np.reshape(dataset_x,(784,1))
    dataset_y=np.array([np.float32(0) for i in range(10)])
    for k in range(10):
        if(k==dataset[i][1]):
            dataset_y[k]=np.float32(1)
   
    d=[dataset_x,dataset_y]
    dataset_n.append(d)

dataset_n[7][1]

def lin_bin(x):
    for i in range(len(x)):
        if(x[i]<0):
            x[i]=-1
        else:
            x[i]=1
    return x


def clip(x,g,lr):
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j]=x[i][j]-lr*(g[i][j])
            if(x[i][j]<-1):
                x[i][j]=-1
            elif(x[i][j]>1):
                x[i][j]=1
            
    return x       

def xnor(x,y):
    c=[]
    for i in range(len(x)):
        if(x[i]==y[i]):
            c.append(1)
        else:
            
            c.append(-1)
    return np.array(c)


def popcnt(x):
    out=0
    for i in range(len(x)):
        if(x[i]==-1):
            out-=1
        else:
            out+=1
    return out


def fast_mul(A,x):
    final=[]
    A=A.T
    for i in range(len(A.T)):
        des=[]
        des.append(xnor(A[i],x))
        o=popcnt(des[0])
        final.append(o)
    A=A.T
    return np.array(final)
       
F=fast_mul(np.array([[-1,-1,1],[1,-1,1],[1,-1,-1]]),[1,-1,1])
print(F)

def ReLU(x):
    o=[]
    for i in range(len(x)):
        if(x[i]>=0):
            o.append(x[i])
        else:
            o.append(0)
    return o

def sat_grad(G):
    for i in range(len(G)):
        for j in range(len(G[0])):
            if(G[i][j]<-1 or G[i][j]>1):
                G[i][j]=0
    return G          

def derivative_cost(y,yd):
    cost=[]
    for i in range(len(y)):
        cost.append(2*(y[i]-yd[i]))
    return np.array(cost)
    
def derivative_relu(x):
    r_dev=[]
    for i in range(len(x)):
        if(x[i]==0):
            r_dev.append(0)
        elif(x>0):
            r_dev.append(1)
        else:
            r_dev.append(0)
    return r_dev

class BNN:
    def __init__(self,inp_layers,output_layers):
        self.inp_layers=inp_layers
        self.ouput_layers=output_layers
        
    def Binarize(self,W):
        for i in range(len(W)):
            for j in range(len(W[1])):
                if(W[i][j]<0):
                    W[i][j]=-1
                elif(W[i][j]>0):
                    W[i][j]=1
                    
        return W            

"""Forward"""
inp=np.array([random.choice((-1,1)) for i in range(784)])
bnn=BNN(784,10)
W=np.array([[np.random.randn() for i in range(784)] for i in range(10)])
alpha=(1/(784*10))*abs(np.sum((W)))
W=bnn.Binarize(W)
#print(W)
y=fast_mul(W,inp.T)
print(y)


"""back_prop"""
lr=0.1
train_set=dataset_n[:45]

print(train_set[0][0].shape)
W=np.array([[np.random.randn() for i in range(784)] for i in range(10)])
eta=0.1
for sample in train_set:
    
    ip=sample[0].T
    label=sample[1]
    
    out=fast_mul(W,ip.T)
    
    out=lin_bin(out)
    Gal=derivative_cost(out,label)
    Gal=np.reshape(Gal,(10,1))
    ip=np.reshape(ip,(1,784))
    Gw=np.multiply(Gal,ip)
    W=clip(W,Gw,lr)

W=bnn.Binarize(W)
print(W)

test=train_set[7][0]
plt.imshow(test)
output=fast_mul(W,test)
print(output)
