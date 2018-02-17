#Logistic Regression
"""
This conclusion is based on matlab data "data.mat"
The algorithmus can be used in basic situation, depends on
different data source and computation platform, details can
be changed for better automatization
"""

#1. import datasets
#from  data.mat
import scipy.io
myData=scipy.io.loadmat("data_path")
#myData.keys() 
#...

#from data.xlsx
import xlrd
workbook=xlrd.open_workbook("data_path")
sheet0=workbook.sheet_by_index(0)
#...

#2. draw points
def drawPoints(x1,x2,y):
    PS=find(y==1) # find index of y when y==1
    NS=find(y==0)
    xlabel("x1")
    ylabel("x2")
    plot(x1[PS],x2[PS],'g+') #points y==1
    plot(x1[NS],x2[NS],'ro')
    legend(['y==1','y==0'])
    return

#3.basic functions of logistic regression
#some preparation of data
def helpMatrix(X):
    """
    add help-vector: x0
    """
    m = len(X[:,0])
    x0=ones((m,1))
    X_help=hstack((x0,X))
    return X_help
    
#sigmoid function
def sigmoid(z):
    """
    z: beta0+x1*beta1+x2*beta2+...xn*betan
    """
    g=1.0/(1.0+exp(-z))
    return g

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

#4.choose algorithmus, depends on the view of points etc.
#Plan a: gradient descent(BGD,?,?)
def BGD(X_help,y,alpha,beta,IterNum):
    """
    Batch Gradient Descent
    ----------------------
    Input:
        X: feature-matrix with x0=1,1,1,...
        y: labels/outputs
        alpha: learning rate
        beta: Initial coefficients-vector
        IterNum: iteration number
    Output:
        beta: new beta after IterNums itreational calculation
    """
    for i in range(IterNum):
        C,gC = costfunction(beta,X_help,y)
        print "Iteration %d |Cost: %f" % (i,C)
        beta = beta - alpha*gC
    return beta

#BGD with the help of feature scaling is easy to realise
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
"""
Together with feature scaling, after running gradient descent,
the ~beta we get are related to ~x, needs to be converted to 
beta which related to the original data x 
"""    
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
    
#Plan b:BFGS quasi-Newton
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

#5.model test with decision boudary drawing

#build polynomial matrix or array
"""
for more complicated situation, polynomial regression makes
a more detailed boundary
"""
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
       
#choose learning rate alpha:
alpha = 0.001
IterNum = 1000
beta_new = BGD(X_help,y,alpha,beta,IterNum)
drawBoundary(beta_new,X_help,y,Degree)

#import test data, calculate and draw figures, comparision
#"Erkennungsrate"
def classification(X,y,beta):
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
    m=len(y)
    x0=ones((m,1))
    X_help = hstack((x0,X))
    z = dot(X_help,beta)
    t = find(z>=0) 
    
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
