%Aufgabe 6
function []=MyPlot_3D()
x=linspace(-1,1,20);
y=linspace(-1,1,20);
[X,Y]=meshgrid(x,y);
%[X,Y]=meshgrid(-1:.1:1,-1:.1:1);
F=sinc(2*X)+sinc(2*Y);

figure(1);
clf;
grid on;
surf(X,Y,F);
shading interp; %kontinuierliche Interpolation der Farben
xlabel('x \rightarrow') %Beschriftung x-Achse
ylabel('y \rightarrow') %Beschriftung y-Achse
zlabel('z \rightarrow') %Beschriftung y-Achse
end