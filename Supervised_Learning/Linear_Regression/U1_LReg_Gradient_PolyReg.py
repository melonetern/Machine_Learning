# linear regression
# we have variables(x0,x1...xn)as input, y as output, look for (beta_0,beta_1,beta_2...)
# y=beta_0+beta_1*x0+beta_2*x1+...

# one variable 
# y=beta_0+beta_1*x
# slope: beta_1= cov(x,y)/var(x)
# intercept: beta_0=mean(y)-beta_1*mean(x)
# hypothese h(x): h=beta_0+beta_1*x=dot(X,beta)
# cost curve C(x): c=sum((y-h)**2)/(2*len(y))   #smaller C(x) better prediction, C(x)-->0
# SST=sum((y-mean(y))**2)		#sum of square for total
# SSE=sum((y-h)**2)				#sum of square for error
# SSReg=sum((h-mean(y))**2)		#sum of square for regression
# r2=SSReg/SST               	# coefficient of determination

%pylab   #interaction environement under ipython numpy+matplotlib.pyplot
import os
os.chdir('mydatapath')
os.listdir('.')

import xlrd
workbook=xlrd.open_workbook('Autos_DE.xlsx')
sheet0=workbook.sheet_by_index(0)
sheet0.cell_value(1,0)
sheet0.ncols
sheet0.nrows

#a) load data
workbook=xlrd.open_workbook('Autos_DE.xlsx')
sheet0=workbook.sheet_by_index(0)
#b) extract columns
 y = array(sheet0.col_values(0,3))  #Verbrauch col_values(@colx,@start_rowx)
x3 = array(sheet0.col_values(3,3))  #Leistung
x4 = array(sheet0.col_values(4,3))  #Gewicht
x5 = array(sheet0.col_values(5,3))  #Beschleunigung

#c) display points
fig=figure(1)
#plot()
sbplt=subplot(311)
scatter(x3,y)
xlabel('x3')
ylabel('y')
sbplt=subplot(312)
scatter(x4,y)
sbplt=subplot(313)
scatter(x5,y)

#d) model calculation
#var()
#cov(x3,y) #output include covarince
beta_1=sum((x3-mean(x3))*(y-mean(y)))/sum((x3-mean(x3))**2)	 #result:0.0866
beta_0=mean(y)-beta_1*mean(x3)
h= beta_0+beta_1*x3			#model
c=sum((y-h)**2)/(2*len(y))	#result: 2.07337
#test model, drawing regression line
x3_test=arange(251)
y_test=beta_0+beta_1*x3_test
plot(x3_test,y_test)
#pick out and compare values
h[56]	#prediction: 8.2217
y[56]	#real value: 9.0384615

SST=sum((y-mean(y))**2)		#sum of square for total
SSE=sum((y-h)**2)			#sum of square for error
SSReg=sum((h-mean(y))**2)	#sum of square for regression
 
r2=SSReg/SST                 #0.7271 good prediction


#---------------------------------------

#with more variables (more variables,better prediction)
#a) load data
workbook=xlrd.open_workbook('Autos_DE.xlsx')
sheet0=workbook.sheet_by_index(0)
#b) extract columns
y = array(sheet0.col_values(0,3))  #Verbrauch
x3 = array(sheet0.col_values(3,3))  #Leistung
x6 = array(sheet0.col_values(6,3))  #Beschleunigung
xh = ones((len(x3),1))
xh.shape
y=y.reshape(len(y),1)
x3=x3.reshape(len(x3),1)
x6=x6.reshape(len(x6),1)
X=hstack((xh,x3,x6))

X_inv=inv(dot(X.T,X))
beta= dot(X_inv,X.T).dot(y)  # beta_0=23.1337  beta_1=0.0762; beat_2=-0.2617

#model test
from mpl_toolkits.mplot3d import Axes3D

x3_test=arange(1,250)
x6_test=arange(1,250)
X3t,X6t=meshgrid(x3_test,x6_test)
h=beta[0]+beta[1]*X3t+beta[1]*X6t
#X_test=vstack((xh_test,x3_test,x6_test))
#h=dot(beta.T,X_test)


ax=figure(0).gca(projection='3d')
surf=ax.plot_surface(X3t,X6t,h,rstride=1, cstride=1, antialiased=True)
scat=ax.scatter(x3,x6,y)


#f)
y = array(sheet0.col_values(0,3)).reshape(len(y),1)  #Verbrauch
x1=array(sheet0.col_values(1,3)).reshape(len(y),1)
x2=array(sheet0.col_values(2,3)).reshape(len(y),1)
x3=array(sheet0.col_values(3,3)).reshape(len(y),1)  #Leistung
x4=array(sheet0.col_values(4,3)).reshape(len(y),1)
x5=array(sheet0.col_values(5,3)).reshape(len(y),1)
x6=array(sheet0.col_values(6,3)).reshape(len(y),1)  #Beschleunigung
x7=array(sheet0.col_values(7,3)).reshape(len(y),1)
xh=ones((len(y),1))
X=hstack((xh,x1,x2,x3,x4,x5,x6,x7))
X_inv=inv(dot(X.T,X))
beta= dot(X_inv,X.T).dot(y)


#---------------------------------

#Gradient Descent
#1.Batch Gradient Descent
#2.Stochastic Gradient Descent
#3.Mini-batch Gradient Descent
#When the number of features/variables more than 10000,the computation
#of normal calculation of betas will be very difficult and expensive.
#Then iterative methods, e.g. Gradient Descent will be used.
#Algorithmus:
#1.set beta with good start value, like beta=0
#2.beta=beta-alpha*(dC(beta)/dbeta)
#3.(dC(beta)/dbeta)=(X.T*X*beta-X.T*y)/m
#4.0<alpha<=2/lambda_max,lambda_max is the largest eigenvalue of 
#(X.T*X)/m, when we have lots of data to use, alpha=1/lambda_max 
#is a good choice.
#reduce "eigen value spread",feature scaling
#~x_i=(x_i-mean(x_i))/std(x_i)
#beta_0=~beta_0- sum(~beta_i*~x_i/std(x_i))
#beta_i=~beta_i/std(x_i)


import os
os.chdir("workspace")
import xlrd
workbook=xlrd.open_workbook("Autos_DE.xlsx")
sheet0=workbook.sheet_by_index(0)
y=array(sheet0.col_values(0,3))     
x3=array(sheet0.col_values(3,3))
x0=ones((len(x3),1))
y=y.reshape(len(y),1)       #output
x3=x3.reshape(len(x3),1)
X=hstack((x0,x3))           #construct input matrix(features/variables)
#c)
R=dot(X.T,X)/len(y)         #correlation matrix
lamb=linalg.eigvals(R)           #eigenvalues  ??(1.18914895e-01,1.23810549e+04)??
#d)
x3_scal=(x3-mean(x3))/std(x3)    #standardization/feature scaling
X_scal=hstack((x0,x3_scal))
R_scal=dot(X_scal.T,X_scal)/len(y)
lamb_scal=linalg.eigvals(R_scal)          #??(1,1)??

#e)f)
X_inv=inv(dot(X.T,X))       
beta= dot(X_inv,X.T).dot(y)  #comparation:   2.15756752,0.08663033
#loss/cost function          #cost function: 2.0733726784232762
def cost(X,y,beta):
    hypothesis = dot(X,beta)
    loss= hypothesis-y
    cost =sum(loss**2)/(2*len(loss))
    return cost

    
#gradient descent model
def BGD(X,y,alpha,beta,IterNum):
    """
    Batch Gradient Descent
    """
    for i in range(IterNum):
        h = dot(X,beta)      #hypothesis
        loss= h-y
        cost =sum(loss**2)/(2*len(loss)) 
        print "Iteration %d |Cost: %f" % (i,cost)
        gradient=dot(X.T,loss)/len(loss)
        beta=beta-alpha*gradient
    return beta

#-----------------------------------
#model test without feature scaling
#-----------------------------------
beta=array([[0],[0]])
alpha=0.00001
IterNum=100000
BGD(X,y,alpha,beta,IterNum)
#result: cost= 2.291406  beta=(0.24268453,0.10278469)

beta=array([[0.5],[0.5]])
alpha=0.00001
IterNum=100000
BGD(X,y,alpha,beta,IterNum)
#result: cost = 2.202690 beta=(0.68284966,0.09907136)

beta=array([[0.7],[0.1]])
alpha=0.00001
IterNum=100000
BGD(X,y,alpha,beta,IterNum)
#result: cost = 2.172962   beta=(0.86340962,0.09754812)

beta=array([[2],[0.8]])
alpha=0.00001
IterNum=100000
BGD(X,y,alpha,beta,IterNum)
#result: cost 2.074627   beta=(2.01233288,0.08785556)

#-----------------------------------
#model test with feature scaling
#-----------------------------------
beta=array([[0],[0]])
alpha=1
IterNum=10
beta_scal=BGD(X_scal,y,alpha,beta,IterNum)   #array([[ 11.20530851],[  3.32404501]])
beta_0=beta_scal[0]-beta_scal[1]*mean(x3)/std(x3)   #2.15756754
beta_1=beta_scal[1]/std(x3)         # 0.08663033

#Coclusion: with feature scaling arrives the min(costfunction) with 
#less  iteration and fast.  x3 (46-230) compare to x0=1 is big difference, 
#without feature scaling convergences very slow


#-------------------------------
#Polynomial Regression
#a)
import scipy.io
mat = scipy.io.loadmat("Lebenserwartung_Training.mat")
x=mat['x']
y=mat['y']
matD = scipy.io.loadmat("Lebenserwartung_Development.mat")
xD=matD['xD']
yD=matD['yD']

#b)
fig=figure(1)
plt=plot(1)
xlabel("x")
ylabel("y")
scatter(x,y)
scatter(xD,yD)
axis([10,45,78,88])


    
def polyRegression(x,y,a,fig,splt): 
    #-------model--------------
    X=ones((len(x),1))
    for i in range(1,a+1):
        X=hstack((X,x**i))
    X_inv=inv(dot(X.T,X))
    beta=dot(X_inv,X.T).dot(y)
    #h=dot(X,beta) #hypothesis
    
    #-------test---------------
    points=linspace(10,45,100)
    x_plt=points.reshape(len(points),1)
    X_plt=ones((len(x_plt),1))
    for i in range(1,a+1):
        X_plt=hstack((X_plt,x_plt**i))
    h_plt=dot(X_plt,beta)
    fig
    splt
    plot(x_plt,h_plt)
    return beta

#coefficient of determination
def DetCoeff(x,y,beta):
    X=ones((len(x),1))
    for i in range(1,len(beta)):
        X=hstack((X,x**i))
    h=dot(X,beta)
    y_mean=mean(y)
    SStot=sum((y-y_mean)**2)
    SSreg=sum((h-y_mean)**2)
    r2=SSreg/SStot
    return r2


# coefficient of determination>1 ????? 
#Warning: As you can see, the underlying assumptions for R-squared
#aren’t true for nonlinear regression. Yet, most statistical software 
#packages still calculate R-squared for nonlinear regression. 
#Reference:http://blog.minitab.com/blog/adventures-in-statistics-2/why-is-there-no-r-squared-for-nonlinear-regression   

#more reliable way, watch the plot with your own eyes




