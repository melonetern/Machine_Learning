%Aufgabe D11
[s,w]=meshgrid(-1:.1:1,-1:.1:1);
t=linspace(0,10,100);
b=exp((s+w*1j)*t);

figure(1);
clf;
plot3(s,w,b,'LineWidth',2)
xlabel('\sigma');
ylabel('\omega');

title('Basisfunktion der Laplace-Transformation');
grid on