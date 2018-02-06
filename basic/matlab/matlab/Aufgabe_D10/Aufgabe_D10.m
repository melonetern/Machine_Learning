f=1000;
w=2*pi*f;
R=100;
C=400e-9;
L=0.1;
Z=R+1/j*w*C+j*w*L

f=linspace(0,10000,1000);
w=2*pi*f;
Z=R+1/j*w*C+j*w*L;
ZR=real(Z);
ZI=imag(Z);
plot3(w,ZR,ZI,'LineWidth',2)
xlabel('\omega in 1/s');
ylabel('Realteil von Z in \Omega');
zlabel('Imaginaerteil von Z in \Omega');
grid on