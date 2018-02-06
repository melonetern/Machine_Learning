%Aufgabe D9
%a)
[x1,x2] = meshgrid(-2:.1:2,-2:.1:2);
x3_Gl1=2*x1-1;
surf(x1,x2,x3_Gl1,1*ones(size(x3_Gl1)));shading flat;
hold on; axis(2*[-1 1 -1 1 -1 1]); caxis([1 3]);

%b)
x3_Gl2=x1/4+x2*15/4-3;
surf(x1,x2,x3_Gl2,2*ones(size(x3_Gl2)) ); shading flat;

%c)
x3_Gl3=3*x1+x2-3;
surf(x1,x2,x3_Gl3,3*ones(size(x3_Gl3)) ); shading flat;

%d)
x3_Gl3=Gl3(0.1,x1,x2);
surf(x1,x2,x3_Gl3,3*ones(size(x3_Gl3)) ); shading flat;
x3_Gl3=Gl3(0.2,x1,x2);
surf(x1,x2,x3_Gl3,3*ones(size(x3_Gl3)) ); shading flat;
x3_Gl3=Gl3(0.3,x1,x2);
surf(x1,x2,x3_Gl3,3*ones(size(x3_Gl3)) ); shading flat;

%e)
x3_Gl3=Gl3(-0.1,x1,x2);
surf(x1,x2,x3_Gl3,3*ones(size(x3_Gl3)) ); shading flat;
x3_Gl3=Gl3(-0.2,x1,x2);
surf(x1,x2,x3_Gl3,3*ones(size(x3_Gl3)) ); shading flat;
x3_Gl3=Gl3(-0.3,x1,x2);
surf(x1,x2,x3_Gl3,3*ones(size(x3_Gl3)) ); shading flat;