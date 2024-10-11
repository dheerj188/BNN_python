import numpy as np 
import random

### Creating a Custom Clip function for BNN activation
def clip(x):
    y=np.array([[0 for i in range(x[0])] for i in range(x)])
    for i in range(len(x)):
        for j in range(len(x[0])):
            if((abs(x[i][j])<1)==0):
                if(x<0):
                    y[i][j]=-1
                else:
                    y[i][j]=1
            else:
                y[i][j]=x[i][j]
    return y
                


def ReLU(x):
    if(x>=0):
        return x
    else: 
        return 0

### numpy version of MSE
def MSE_loss(y,yd):
    L=np.array([0 for i in range(len(y))])
    for i in range(len(y)):
        L[i]=((y[i]-yd[i])**2)/2
    return L


        
### BNN Behaviour definition
class BNN:
    def __init__(self,weights):
        self.weights=weights
    
    def multiply(self,w,x):
        product=np.array([0 for i in range(len(x))])
        for i in range(len(w)):
            out=0
            for j in range(len(w[0])):
                out+=w[i][j]*x[j]
            product[i]=out
        return product
            
    def binarize_1D(self,a):
        out=np.array([0 for i in range(len(a))])
        for i in range(len(a)):
            if(a[i]>0.5):
                out[i]=1
            else:
                out[i]=-1
        return out
    def binarize_2D(self,w):
        W_binary=np.array([[0 for i in range(len(w[0]))] for i in range(len(w))])
        Sum=0 
    #binary weight vector
        for i in range(len(w)):
            for j in range(len(w[0])):
                if(w[i][j]>0):
                    W_binary[i][j]=1
                else:
                    W_binary[i][j]=-1
                Sum+=w[i][j]
                alpha=Sum/(len(w)*len(w[0])) 
        return (alpha,W_binary)
    
    def STE_mask(self,gab):
        ga=np.array([0 for i in range(len(gab))])
        for i in range(len(gab)):
            if(abs(gab[i])<=1):
                ga[i]=gab[i]
        return ga
        
    def forward_pass(self,ip):
        out=self.multiply(self.weights[0],ip)
        count=0
        outs=[out]
        for w in self.weights[1:]:
            w_out=self.binarize_2D(w)[1]
            out=self.multiply(w_out,out)
            count+=1
            if(count<len(self.weights)-1):
                out=self.binarize_1D(out)
            outs.append(out)
        return outs
    
    def backward_pass_update(self,Y,Yd,A,eta):
        do_C=np.array([0 for i in range(len(X))])
        do_W=[]
        for i in range(len(Yd)):
            do_C[i]=Y[i]-Yd[i]
        
        gal=do_C
        L=len(self.weights)
        k=len(self.weights)
        ga=gal
        for w in self.weights[::-1]:
            
            if(k<L):
                ga=self.STE_mask(ga)
           
            ga_prev=np.array(self.multiply(np.transpose(w),ga))
            ga=np.array([ga])
            Aq=np.array([A[k-1]])
            gw=np.matmul(np.transpose(ga),Aq)
            ga=ga_prev
            do_W.append(gw)
            
            k-=1
           
        
        
        for i in range(len(self.weights)):
            nabla_W=do_W
            self.weights[i]=np.subtract(self.weights[i],nabla_W[i])
            #self.weights[i]=clip(w_update)
        
        return 0       

weights=[np.array([[random.randint(-1,1)+random.random() for i in range(10)] for i in range(10)]) for i in range(8)]
X=np.array([i for i in range(10)])
Y=np.array([2*i for i in range(10)])
model=BNN(weights)

As=model.forward_pass(X)
print(As)

epochs=1
for epoch in range(epochs):
    As=model.forward_pass(X)
    y=model.forward_pass(X)[-1]
    model.backward_pass_update(y,Y,As,1)
    
print(model.forward_pass(X)[-1])
