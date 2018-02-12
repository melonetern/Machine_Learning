#Logistic Regression
import os
os.chdir("F:/ML/data")
import scipy.io
D1_Training=scipy.io.loadmat("D1_Training.mat")
#D1_Training.keys()
y=D1_Training['y']
X=D1_Training['X']
x1=X[:,0]
x2=X[:,1]

def ZeichneStreudiagramm(x1,x2,y,fig,plt):
    Einsen = find(y==1)
    Nullen = find(y==0)
    fig
    plt
    xlabel('Test 1(x1)')
    ylabel('Test 2(x2)')
    plot(x1[Einsen],x2[Einsen],'g+')
    plot(x1[Nullen],x2[Nullen],'ro')
    legend(['accepted','not accepted'])
    return
ZeichneStreudiagramm(x1,x2,y,figure(1),plot(111))
    
#c)
def sigmoid(z):
    g=1/(1+exp(-z))
    return g

z=linspace(-10,10,100)
fig2=figure(2)
plt2=plot(z,sigmoid(z))

#d)
def LossFunction(beta,X,y):
    """
    X is the features-matrix with help-vector x0
    """
    m= len(y)
    h= sigmoid(dot(X,beta))
    C= -sum(dot(y.T,log(h))+dot((1-y).T,log(1-h)))/m
    gC= dot(X.T,sigmoid(dot(X,beta))-y)/m
    return C,gC

#e)
X_new=hstack((ones((len(y),1)),X))
beta=zeros((3,1))
LossFunction(beta,X_new,y)

#f)
#gradien descent
def BGD(X,y,alpha,beta,IterNum):
    """
    Batch Gradient Descent
    """
    for i in range(IterNum):
        C,gC = LossFunction(beta,X,y)
        print "Iteration %d |Cost: %f" % (i,C)
        beta=beta-alpha*gC
    return beta

#draw decision boundary
def draw(beta,fig,plt):
    """
    Decision boundary: 
    0=beta[0]+beta[1]*x1+beta[2]*x2
    """
    x_plt=linspace(0,100,200)
    y_plt=-(beta[0]+beta[1]*x_plt)/beta[2]
    fig
    plt
    plot(x_plt,y_plt)

#model test  1   
alpha=0.001
IterNum=400
beta_new = BGD(X_new,y,alpha,beta,IterNum)
draw(beta_new,figure(1),plot(111))
#model test 2
alpha=0.001
IterNum=10000
beta_new = BGD(X_new,y,alpha,beta,IterNum)
draw(beta_new,figure(1),plot(111))
#model test 3
alpha=0.001
IterNum=100000
beta_new = BGD(X_new,y,alpha,beta,IterNum)
draw(beta_new,figure(1),plot(111))
#model test 4
alpha=0.001
IterNum=1000000
beta_new = BGD(X_new,y,alpha,beta,IterNum) 
#result:array([[-8.83215253],[ 0.10395482],[ 0.10515304]])
draw(beta_new,figure(1),plot(111))
#without feature scaling, it convergences very slow







#g)
#------repeat with feature scaling------------------
def featScal(X_new):
    m=len(X_new[:,0])
    X_scal=ones((m,1))
    for i in (1,2):
        t=X_new[:,i].reshape(m,1)
        t=(t-mean(t))/std(t)
        X_scal=hstack((X_scal,t))
    return X_scal


    
def betaScalToNew(beta_scal,X_new):
    n = len(beta_scal)-1 #number of features
    m = len(X_new[:,0]) #number of samples
    beta_new=beta_scal
    for i in (1,n):
        t=X_new[:,i].reshape(m,1)
        beta_new[i]=beta_new[i]/std(t)
        beta_new[0]=beta_new[0]- beta_new[i]*mean(t)
    return beta_new
#
alpha=0.5
IterNum=400
X_scal=featScal(X_new)
beta_scal=BGD(X_scal,y,alpha,beta,IterNum)  #cost=0.232445
#result of beta_scal: array([[ 1.9323593 ],[ 2.90257857],[ 2.97776722]])
beta_new = betaScalToNew(beta_scal,X_new)
#result of beta_new: array([[-8.66255801],[ 0.10168347],[ 0.10288889]])
draw(beta_new,figure(1),plot(111))

#Conclusion:
#without feature scaling:
#alpha=0.001, Iteration:1000000, and we get result
#array([[-8.83215253],[ 0.10395482],[ 0.10515304]])
#with feature scaling:
#alpha=0.001, Iteration:400, and we get result
#array([[-8.66255801],[ 0.10168347],[ 0.10288889]])
#most of the time, gradient descent always with feature scaling

#h)
D1V=scipy.io.loadmat("D1_Validierung.mat")
yv=D1V['yv']
Xv=D1V['Xv']


def classification(X,y,beta):
    m=len(y) #number of samples
    x0=ones((m,1))
    X_new=hstack((x0,X))
    z=dot(X_new,beta)
    t=find(z>=0)
    
    #build compare matrix
    compMatrix=zeros((m,1))
    for i in t:
        compMatrix[i,0]=1
    
    #count right predictions
    count=0.0 #remember set float type
    for i in range(m):
        if compMatrix[i]==y[i]:
            count+=1
    ER=count/m
        
    return ER

classification(Xv,yv,beta_new)

#i)
#BFGS
#scipy.optimize.minimize(fun, x0, args=(), method='BFGS', 
#jac=None, tol=None, callback=None, options={'disp': False, 
#'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 
#'return_all': False, 'maxiter': None, 'norm': inf})

def costfunction(beta,X,y):
    """
    X is the features-matrix with help-vector x0
    """
    m= len(y)
    h= sigmoid(dot(X,beta))
    cost=zeros((m,1))
    for i in range(m):
        if y[i]==1:
            cost[i]=-log(h[i])
        else:
            cost[i]=-log(1-h[i])
    C=sum(cost)/m
    return C
    
#!!!!!!! log(1-h), make beta not start with 0, otherwise error!!!!!
beta=zeros((3,1))

res = scipy.optimize.minimize(costfunction,beta,args=(X_new,y),method='BFGS',options={'disp': True})
#scipy.optimize.minimize(object_function,object_arg_initial, args=(arg1,arg2))
beta_new = res.x
# array([-9.29813692,  0.10920503,  0.1102504 ])

#j)
def polyFeatures(x1,x2):
    '''
    a: maximum degree of polynom
    '''
    t=ones((len(x1),1))
    x1=x1.reshape(len(x1),1)
    x2=x2.reshape(len(x2),1)
    t=hstack((t,x1,x2,x1**2,x2**2,x1*x2))
    return t


Xq=polyFeatures(X[:,0],X[:,1])

beta=zeros((6,1))
res = scipy.optimize.minimize(costfunction,beta,args=(Xq,y),method='BFGS',options={'disp': True})
beta_new=res.x
#result:array([ -1.44945234e+01,   1.69448335e-01,   1.77778142e-01,
       # -9.37941959e-04,  -8.44028524e-04,   3.27064876e-03])
    
#plot
def Cal_x2(x1p,beta_new):
    """
    beta0-5
    return: x2 points
    """
    a=beta_new[4]
    b=beta_new[2]+beta_new[5]*x1p
    c=beta_new[0]+beta_new[1]*x1p+beta_new[3]*(x1p**2)
    res=(sqrt(abs(b**2-4*a*c))-b)/(2*a)
    return res
    
x1p=linspace(1,100,100)    
x2p=Cal_x2(x1p,beta_new)    
figure(2)
plot(x1p,x2p)

#k)
Xvq=polyFeatures(Xv[:,0],Xv[:,1])
beta_new=beta_new.reshape(6,1)
Xvq=delete(Xvq,0,1)
classification(Xvq,yv,beta_new)
#result:ER=0.948



#-------------------------------------
#Logistic Regression with Regularization

#a)
import os
os.chdir("F:/ML/data")
import scipy.io
D2_Training=scipy.io.loadmat("D2_Training.mat")
#D2_Training.keys()
y=D2_Training['y']
X=D2_Training['X']
x1=X[:,0]
x2=X[:,1]

#b)
ZeichneStreudiagramm(x1,x2,y,figure(1),plot(111))
axis([-2,2,-2,2])

#c)
def buildPoly(x1,x2,a):
    """
    a: degree
    return: Xq, polynom Matrix
            n, number of features
    """
    Xq=ones((len(x1),1)) 
    x1=x1.reshape(len(x1),1)
    x2=x2.reshape(len(x2),1)
    for i in range(1,a+1):
        for k in range(i+1):
            xi=(x1**(i-k))*(x2**k)
            Xq=append(Xq,xi,axis=1)
    n=size(Xq)/len(Xq)-1 
    return Xq,n

Xq_a2,n_a2=buildPoly(x1,x2,2)


beta=zeros((n_a2+1,1))
res = scipy.optimize.minimize(costfunction,beta,args=(Xq_a2,y),method='BFGS',options={'disp': True})
#scipy.optimize.minimize(object_function,object_arg_initial, args=(arg1,arg2))
beta_a2 = res.x    
#Optimization terminated successfully.
         #Current function value: 0.323327
         #Iterations: 35
         #Function evaluations: 296
         #Gradient evaluations: 37
         
#plot
"""
see matlab function in source file
solution by prof.
zeichneEntscheidungsschwelle.m

"""

