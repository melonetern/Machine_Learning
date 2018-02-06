%Aufgabe D5
function []=MyPlot(n)
x=linspace(-4*pi,4*pi,n);
y1=sin(x);
y2=sin(x-pi/4);
y3=sin(x-pi/2);

figure(1);
clf;
plot(x,y1,x,y2,x,y3);
grid on;
xlabel('x \rightarrow');
ylabel('y \rightarrow');
title('Graph der Funktion y=sin(x-a)');
legend('a=0','a=\pi/4','a=\pi/2');
end