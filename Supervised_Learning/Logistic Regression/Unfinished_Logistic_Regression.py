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

def drawPoints(x1,x2,y):
    PS=find(y==1) # find index of y when y==1
    NS=find(y==0)
    xlabel("x1")
    ylabel("x2")
    plot(x1[PS],x2[PS],'g+') #points y==1
    plot(x1[NS],x2[NS],'ro')
    legend(['','y==1','y==0'])
    return
    
figure(1)
plot(111)
drawPoints(x1,x2,y)
    
#c)
def sigmoid(z):
    g=1/(1+exp(-z))
    return g

z=linspace(-10,10,100)
fig2=figure(2)
plt2=plot(z,sigmoid(z))

#d)
def helpMatrix(X):
    """
    add help-vector: x0
    """
    m = len(X[:,0])
    x0=ones((m,1))
    X_help=hstack((x0,X))
    return X_help
    
#cost function
def costfunction(beta,X_help,y):
    """
    Input:    
        beta: nx1-array (n: number of features)
        X: features-matrix with help-vector x0=1,1,1,...
        y: labels; output
    Output:
        C: cost function
        gC: gradient of C
    """
    m= len(y)
    h= sigmoid(dot(X_help,beta))
    cost=zeros((m,1))
    for i in range(m):
        if y[i]==1:
            cost[i]=-log(h[i])
        else:
            cost[i]=-log(1-h[i])
    C = sum(cost)/m
    gC = dot(X_help.T,h-y)/m
    return C,gC
    
#e)
X_help=helpMatrix(X)
beta=zeros((3,1))
costfunction(beta,X_help,y)

#f)
#gradien descent
def BGD(X_help,y,alpha,beta,IterNum):
    """
    Batch Gradient Descent
    """
    for i in range(IterNum):
        C,gC = costfunction(beta,X_help,y)
        print "Iteration %d |Cost: %f" % (i,C)
        beta=beta-alpha*gC
    return beta


#build polynomial matrix or array
def buildPoly(x1,x2,deg):
    """
    Input:
        x1,x2: two feature-vectors or two numbers
        deg: maximum degree of polynomial function
    Output:
        Xq: polynomial 'features'-matrix with help-vector 
            OR array([1,x1,x2,x1^2,x2^2,...])
        x0=1,1,1,... 
    """
    t=isinstance(x1,np.ndarray)
    if not t:
        Xq=array([1])
    else:
        m=len(x1)
        Xq = ones((m,1))
        x1=x1.reshape(m,1)
        x2=x2.reshape(m,1)
        
    for i in range(1,deg+1):
        for k in range(i+1):
            xi=(x1**(i-k))*(x2**k)
            Xq=hstack((Xq,xi))
    return Xq    
#draw decision boundary
def drawBoundary(beta,X_help,y,Deg):
    """
    This function not working for more than 2 features yet,
    for more than 2 features, need to be debugged
    """
    #boundary(linear): 0=beta0+beta1*x1+beta2*x2
    if size(X_help,axis=1)<=3: 
        x_plt=linspace(0,100,200)
        y_plt=-(beta[0]+beta[1]*x_plt)/beta[2]
        plot(x_plt,y_plt)
    
    #boundary(polynomial): use contour(x,y,z)
    else:
        u=linspace(1,100,100)
        v=linspace(1,100,100)
        z=zeros((len(u),len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j]=dot(buildPoly(u[i],v[j],Deg),beta)
        z=z.T
        contour(u,v,z,0)

#model test 1   
alpha=0.001
IterNum=400
beta_new = BGD(X_help,y,alpha,beta,IterNum)
drawBoundary(beta_new,X_help,y,1)

#model test 2
alpha=0.001
IterNum=100000
beta_new = BGD(X_help,y,alpha,beta,IterNum)
drawBoundary(beta_new,X_help,y,1)

#model test 3 (close to the optimization)
alpha=0.001
IterNum=1000000
beta_new = BGD(X_help,y,alpha,beta,IterNum)
#result:array([[-8.83215253],[ 0.10395482],[ 0.10515304]])
drawBoundary(beta_new,X_help,y,1)
#without feature scaling, it convergences very slow



#g)
#repeat with feature scaling
def featureScaling(X_help):
    """
    m: number of samples
    n: number of features +1
    ~x = (x-mean(x))/std(x)
    """
    m = len(X_help[:,0])    
    n = size(X_help)/len(X_help[:,0])
    X_scal=ones((m,1))
    for i in range(1,n):
        t= X_help[:,i].reshape(m,1)
        t=(t-mean(t))/std(t)
        X_scal=hstack((X_scal,t))
    return X_scal

def betaUnscaling(beta_scal,X_help):
    """
    n: number of features +1
    """
    n = len(beta_scal)
    beta_new=zeros((n,1))
    beta_sum1TOn=0 #
    for i in range(1,n):
        t = X_help[:,i]
        beta_new[i]=beta_scal[i]/std(t)
        beta_sum1TOn+=beta_new[i]*mean(t)
        beta_new[0]=beta_scal[0]-beta_sum1TOn
    return beta_new
    
#
alpha=0.5
IterNum=400
X_scal=featureScaling(X_help)
beta_scal=BGD(X_scal,y,alpha,beta,IterNum)  #cost=0.232445
#result of beta_scal: array([[ 1.9323593 ],[ 2.90257857],[ 2.97776722]])
beta_new = betaUnscaling(beta_scal,X_help)
#result of beta_new: array([[-8.66255801],[ 0.10168347],[ 0.10288889]])
drawBoundary(beta_new,X_help,y,1)

"""
Conclusion:
without feature scaling:
alpha=0.001, Iteration:1000000, and we get result
array([[-8.83215253],[ 0.10395482],[ 0.10515304]])
with feature scaling:
alpha=0.001, Iteration:400, and we get result
array([[-8.66255801],[ 0.10168347],[ 0.10288889]])
most of the time, gradient descent with feature scaling
works much more better
"""

#h)
D1V=scipy.io.loadmat("D1_Validierung.mat")
yv=D1V['yv']
Xv=D1V['Xv']


def classification(X_help,y,beta):
    """
    Input:
        X: feature-matrix without help-vetor x0=1,1,1,...
        y: labels/outputs
        beta: model coefficient-vector, result after calculation
    Output:
        ER: 'Erkennungsrate'
    Desciption:
        build a compare-vector with the help of indices of z=dot(X_help,beta)>=0,
        which means y==1, compare the real labels of y and the comapare vector,
        get the rate by dividing the sum of successful comparision and the total
        number of labels        
    """
    #find indecies of y==1
    z = dot(X_help,beta)
    t = find(z>=0) 
    m=len(y)
    #build compare vector
    compVector=zeros((m,1))
    for i in t:
        compVector[i,0]=1
    
    #count right predictions
    count=0.0
    for i in range(m):
        if compVector[i]==y[i]:
            count+=1
    ER = count/m
    
    return ER

Xv_help=helpMatrix(Xv)
classification(Xv_help,yv,beta_new)

#i)
#BFGS
"""
scipy.optimize.minimize(fun, x0, args=(), method='BFGS', 
jac=None, tol=None, callback=None, options={'disp': False, 
'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 
'return_all': False, 'maxiter': None, 'norm': inf})
"""
def cf(beta,X_help,y):
    """
    return result of cost function 
    """
    m= len(y)
    h= sigmoid(dot(X_help,beta))
    cost=zeros((m,1))
    for i in range(m):
        if y[i]==1:
            cost[i]=-log(h[i])
        else:
            cost[i]=-log(1-h[i])
    C = sum(cost)/m
    return C
res = scipy.optimize.minimize(cf,beta,args=(X_help,y),method='BFGS',options={'disp': True})
beta_new = res.x
# array([-9.29813692,  0.10920503,  0.1102504 ])
drawBoundary(beta_new,X_help,y,1)


#j)

a=4 #3,4,6,8
Xq_a=buildPoly(X[:,0],X[:,1],a)
beta=zeros((len(Xq_a[0,:]),1))+0.001
res = scipy.optimize.minimize(cf,beta,args=(Xq_a,y),method='BFGS',options={'disp': True})
beta_new=res.x
drawBoundary(beta_new,Xq_a,y,a)


#k)
Xvq=buildPoly(Xv[:,0],Xv[:,1],a)
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
figure(2)
plot(111)
drawPoints(x1,x2,y)
axis([-2,2,-2,2])

#c)
a=2
Xq_a2=buildPoly(x1,x2,a)
n=size(Xq_a2,axis=1)

beta=zeros((n,1))+0.1
res = scipy.optimize.minimize(cf,beta,args=(Xq_a2,y),method='BFGS',options={'disp': True})
beta_a2 = res.x    

figure(2)
plot(111)
drawBoundary(beta_a2,Xq_a2,y,a)
























#-----------------------------------------

#-----------------sklearn-----------------

#-----------------------------------------

#Logistic Regression
import os
os.chdir("F:/ML/data")
import scipy.io
D1_Training=scipy.io.loadmat("D1_Training.mat")
#D1_Training.keys()
y=D1_Training['y']
X=D1_Training['X']

D1_Validierung=scipy.io.loadmat("D1_Validierung.mat")
yv=D1_Training['yv']
Xv=D1_Training['Xv']

#data preprocessing:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X)
X_test_std = sc.transform(Xv)

X_combined_std=vstack((X_train_std,X_test_std))
y_combined_std=hstack((y,yv))   

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y)
plot_decision_regions(X_combined_std,
... y_combined, classifier=lr,
... test_idx=range(105,150))