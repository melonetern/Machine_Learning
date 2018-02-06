#übungsblatt 0

#cmd
#cd ..
#cd c:/python27/scripts
#ipython
#or after setting the system environment
#ipython -pylab

%pylab   #build similar environment as MATLAB

import os
os.getcwd()  #get current work directory
os.chdir('new work space')    # change directory

4/5
4.0/5
4.0//5  #only return integer
2**3
log10(100)
sin(pi/2)
exp(1j*pi/2)
whos
1.1**inf
inf - inf
help(nan)


#Aufgabe 1
sin(pi/3)
(2+3j)*exp(1j*pi/6)
10*log10(1000)

#numpy
import numpy as np
data = np.array([[1,2,3],[4,5,6]])
data.shape  #对象的结构，如（2,3）
data.dtype  #对象元素的数据类型
data.ndim  #对象的维度


a = arange(5)
m = array([arange(2),arange(2)])
m.shape
float64(9)
int8(9)
sctypeDict.keys()

t = dtype('Float64')
t.char
t.type
t.str

zeros((3,1))
zeros_like()
eye(3,3)
linspace(1,2,5)

#Aufgabe D2

D=eye(4,4)*4
empty()
empty_like()
a=zeros((2,6))
b=ones((3,6))
ones_like()
vstack((a,b))


a=arange(1,4)
b=arange(4,7)
a.dot(b)
a*b
a.size
a.shape
a.reshape(3,1)

#Aufgabe D3
a=arange(11)
b=arange(11)
for i in arange(1,11):
	a=vstack((a,i*b))
a[:,0]=arange(11)

a[5,1:]
a[0:6:5,:]

#Aufgabe D4
def mysum(n):
	return sum(arange(n+1))
	
mysum(3)

x=linspace(-3,3,200)
y=x*2
figure(1)
clf()
plot(x,y)
title("Graph der Funktion y=x*2")
xlabel("x")
ylabel("y")
grid()
grid(0)

x=linspace(-4,4,200)
y1=(2*x+1)**2-1
y2=(2*x)**2
y3=(2*x-1)**2+1
plot(x,y1,x,y2,x,y3)
title('Graph der Funktion y=(2*x-a)/2+a')
xlabel('x')
ylabel('y')
grid()
legend(['a=-1','a=0','a=1'])
axis((-4,4,-2,10))

#Aufgabe D5
x=linspace(-4*pi,4*pi,200)
y1=sin(x)
y2=sin(x-pi/4)
y3=sin(x-pi/2)
figure(1)
clf()
plot(x,y1,x,y2,x,y3)
title("Graph der Funktion y=sin(x-a)")
xlabel("x")
ylabel("y")
grid()
legend(['a=-1','a=0','a=1'])


#3D-plot
from mpl_toolkits.mplot3d import Axes3D

x = linspace(-2,2,100)
y = linspace(-2,2,100)
X,Y = meshgrid(x,y)
z=X**2 - Y**2

fig=figure(1)
clf()
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,z,rstride=1, cstride=1, antialiased=True)

#Aufgabe D6
x=arange(1,10.1,0.1)
y=arange(1,10.1,0.1)
X,Y=meshgrid(x,y)
#Z=sin(2*X*pi)/(2*X*pi)+sin(2*Y*pi)/(2*Y*pi)
Z=sinc(2*X*pi)+sinc(2*Y*pi)

fig=figure(1)
clf()
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,Z,rstride=1, cstride=1, antialiased=True)

#Aufgabe D7
import scipy.io
mat = scipy.io.loadmat('file.mat')
savemat()
whosmat()

fig=figure(1)
clf()
lena=mat['lena']
einstein=mat['einstein']
noise=mat['noise']
imshow(lena)
imshow(lena.T)
imshow(lena*0.1)
imshow(lena*2.0)
imshow(lena*einstein)
imshow(lena+1.0)
imshow(lena-1.0)
imshow(lena+einstein)
imshow(lena+noise)

#linear equations
#Aufgabe D8
A=array([[2,3,1],[4,1,4],[3, 4, 6]])
inv(A)
det(A)
eig(A)
b=array([[-4],[9],[0]])

solve(A,b)

#Aufgabe D9
x1,x2=meshgrid(arange(-2,2.1,.1),arange(-2,2.1,.1))
x3_Gl1=2*x1-1

fig=figure(1)
clf()
ax=fig.gca(projection='3d')
surf=ax.plot_surface(x1,x2,x3_Gl1)
...


#complex number
x=2+3j
x=2.1+3j
imag(x)
real(x)
conj(x)
abs(x)
angle(x)


#Parallelschaltung von R und C
#Berechnung der komplexen Impedanz Z fuer f = 1kHz
f=1000; #Frequenz
w=2*pi*f; #Kreisfrequenz
R=100; #Widerstandswert
C=10e-6; #Kapazitaet
Z=1/(1/R+1j*w*C) #Berechnung der komplexen Impedanz

#Erstellung einer Ortskurve fuer Z
f = linspace(1,10000,1000); #Frequenz: 1000 Werte zwischen 1 und 10k Hz
w = 2*pi*f; #Kreisfrequenz: Vektor mit 1000 Werten
Z=1./(1/R+1j*w*C) #./ ist hier notwendig, da w jetzt ein Vektor ist
ZR = real(Z); #Realteil von Z
ZI = imag(Z); #Imaginaerteil von Z

fig=figure(1)
clf()
ax=Axes3D(fig)
ax.scatter(w,ZR,ZI,marker='o',color='r')  #Grafische Darstellung der Ortskurve

xlabel('w in 1/s')
ylabel('Realteil von Z in w');
zlabel('Imaginaerteil von Z in w');




#Aufgabe D10

