f=1000;
w=2*pi*f;
R=100;
C=10e-6;
Z=1/(1/R+j*w*C)

f=linspace(0,10000,1000);
w=2*pi*f;
Z=1./(1/R+j*w*C);
ZR=real(Z);
ZI=imag(Z);
plot3(w,ZR,ZI,'LineWidth',2)
xlabel('\omega in 1/s');
ylabel('Realteil von Z in \Omega');
zlabel('Imaginaerteil von Z in \Omega');
grid on
