%Übung 1: Lineare Regression mit einer/mehrer Eingangsvariable(n)


%Aufgabe D1: Auto-Daten mit einer Eingangsvariable
%a)
data= xlsread('F:\Machine_Learning\ML_OTH\Aufgaben\¨¹bung_1\Autos_DE.xlsx');
%b)
y =data(:,[1]);
x4=data(:,[4]);  %Leistung    
x5=data(:,[5]);  %Gewicht
x6=data(:,[6]);  %Beschleunigung
%c)
subplot(1,3,1)
scatter(x4,y)
xlabel('x4\rightarrow')
ylabel('y\rightarrow')

subplot(1,3,2)
scatter(x5,y)
xlabel('x5\rightarrow')
ylabel('y\rightarrow')

subplot(1,3,3)
scatter(x6,y)
xlabel('x6\rightarrow')
ylabel('y\rightarrow')


%d)
%x4 Leistung
beta_1= sum((y-mean(y)).*(x4-mean(x4)))/sum((x4-mean(x4)).^2) %0.0866
%beta_1= cov(x4,y)/var(x4);
%beta_1=beta_1(1,2)
beta_0=mean(y)-beta_1*mean(x4)	
hyp=beta_0+beta_1.*x4;          %hypothese h(x)  
C = sum((y-hyp).^2)/(2*length(y))  %Kostenfunktion: 2.0734
x_p = [0:250];              %x_plot
y_p = beta_1*x_p+beta_0;  %y_plot
subplot(3,1,1)
hold on
plot(x_p,y_p)
xlabel('Leistung\rightarrow')
ylabel('Verbrauch\rightarrow')

hyp(101)  %Vohersage: 9.7810
y(101)  %Realität: 13.0556

%x5 Gewicht
beta_1=cov(x5,y)/var(x5);
beta_1=beta_1(1,2)  %0.0090
beta_0=mean(y)-beta_1*mean(x5);  %beta_0 == -0.8999
hyp=beta_0+beta_1.*x5;  
C = sum((y-hyp)^2)/(2*length(y));  %C == 1.6424

x_p=[1:2500];
y_p=beta_0+beta_1.*x_p;
subplot(1,3,2)
hold on
plot(x_p,y_p)
xlabel('Gewicht\rightarrow')
ylabel('Verbrauch\rightarrow')

hyp(101) %Vohersage: 11.4121
y(101) %Realität: 13.0556

%x6 Beschleunigung
beta_1=cov(x6,y)/var(x6);
beta_1=beta_1(1,2) %-0.6290
beta_0=mean(y)-mean(x6)*beta_1 % 21.2840
hyp=beta_1.*x6+beta_0;
C=sum((y-hyp).^2)/(2*length(y))   %6.0147

x_p=[1:30];
y_p=beta_0+beta_1.*x_p;
subplot(1,3,3)
hold on
plot(x_p,y_p)
xlabel('Beschleunigung\rightarrow')
ylabel('Verbrauch\rightarrow')
hyp(101)  %Vohersage: 10.5935
y(101)  %Realität: 13.0556


%f)
%mit dem selben axis-Bereich kann man besser vergeleichen
subplot(1,3,1)
axis([0 1000 0 30])
subplot(1,3,2)
axis([0 1000 0 30])
subplot(1,3,3)
axis([0 1000 0 30])
%beta_1 ist das Slope, umso steiler, umso mehreres Einfluss
%beta_1 >0 positives Einfluss
%beta_1 <0 negatives Einfluss
%dazu haben wir:
%0.0866 das Leistung hat meheres positives Einfluss
%0.0090 das Gewicht hat weinigestes positives Einfluss,
%-0.6290 die Beschleunigung hat größestes Einfluss, zwar negatives
%Bemerkung:es gibt Möglichkeit, das Einfluss nicht direkt wirkt.

%g)
% Lösung sieh %d) C ist Kostenfuntktion

%h)
%je niediger die Kostenfunktion ist, desto besser zur Vorhersage
%wenn C=0, you know the feature
%für Leistung,Gewicht und Beschleunigung sind die Ergebnisse der Kostenfunktion wie folgende 
%Leistung: 2.0734  
%Gewicht:  1.6424  hier beste Vohersage
%Beschleunigung: 6.0147  hier schlechte Vohersage

%wenn die Vohersagen und Realität vergleicht werden,
%besser mehr Daten, weil es gibt es einige größe-abweichende Parametersätze.

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%mit Regression-Funktionnen
beta=polyfit(Lst,Vb,1); %Lst==x4, Vb==y
beta_1=beta(1) %0.0866
beta_0=beta(2) %2.1576
p_x=[1:250];                  %plot x
p_y=beta_1*p_x+beta_0;      %plot y
subplot(3,1,1);
scatter(Lst,Vb)
hold on
plot(p_x,p_y)
h=beta_1*Lst+beta_0;       %hypothese h(x)
SST=sum((Vb-mean(Vb)).^2)    %sum of square for total
SSE=sum((Vb-h).^2)           %sum of square for error
SSReg=sum((h-mean(Vb)).^2)   %sum of square for regression
r2=SSReg/SST                 %0.7271 gute Vorhersage

beta=polyfit(Gw,Vb,1);
beta_1=beta(1)             %0.0090             
beta_0=beta(2)             %-0.8999
p_x=[500:2500];
p_y=beta_1*p_x+beta_0;
subplot(3,1,2);
scatter(Gw,Vb)
hold on
plot(p_x,p_y)
h=beta_1*Gw+beta_0;       %hypothese h(x)
SST=sum((Vb-mean(Vb)).^2)    %sum of square for total
SSE=sum((Vb-h).^2)           %sum of square for error
SSReg=sum((h-mean(Vb)).^2)   %sum of square for regression
r2=SSReg/SST                 %0.7838  gute Vorhersage

beta=polyfit(Besl,Vb,1);
beta_1=beta(1)             % -0.6290
beta_0=beta(2)             %21.2840
p_x=[5:.1:30];
p_y=beta_1*p_x+beta_0;
subplot(3,1,3);
scatter(Besl,Vb)
hold on
plot(p_x,p_y)
h=beta_1*Besl+beta_0;       %hypothese h(x)
SST=sum((Vb-mean(Vb)).^2)    %sum of square for total
SSE=sum((Vb-h).^2)           %sum of square for error
SSReg=sum((h-mean(Vb)).^2)   %sum of square for regression
r2=SSReg/SST                 %0.2084 schlechte Vorhersage

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


%Aufgabe D2: Auto-Daten mit mehreren Eingangsvariablen

%a)
data= xlsread('E:\ML\MATLAB\?bung_1\Autos_DE.xlsx');
%b)
y =data(:,[1]);
x3=data(:,[4]); %Leistung
x6=data(:,[7]); %Baujahr
x0= ones(length(y),1); %Hilfsvariable
X=[x0 x3 x6];

%c)
beta = inv(X'*X)*X'*y  % 23.1337;0.0762;-0.2617
%d)
x3_p= [1:1:230];
x6_p= [1:1:230];
[X3,X6] = meshgrid(x3_p,x6_p);
y_p = beta(1)+beta(2)*X3+beta(3)*X6;
figure(1)
subplot(2,1,1)
scatter(x3,y)
hold on
plot(x3_p, y_p)
title('x3y-Ebene')

subplot(2,1,2)
scatter(x6,y)
hold on 
plot(x6_p, y_p)
title('x6y-Ebene')

%e)
%D1: beta1==0.0866;   beta2=-0.5905 (mit einer EV)
%D2: beta1==0.0762;   beta2=-0.2617 (mit 2 EVn)
%x3 und x6 beeinflussen einandere
beta2=cov(x6,y)/var(x6)
beta2=beta1(2)
beta0=mean(y)-beta1*mean(x6)
y6_P=beta0+beta2*x6_p;
figure(2)
subplot(2,1,1)
scatter(x6,y)
hold on
plot(x6_p,y6_P)

%f)
y =data(:,[1]);
x0=data(:,[2]); %Zylinder Anzahl
x1=data(:,[3]); %Hubraum
x2=data(:,[4]); %Leistung
x3=data(:,[5]); %Gewicht
x4=data(:,[6]); %Beschleunigung
x5=data(:,[7]); %Baujahr
x6=data(:,[8]); %Herkunft
xh= ones(length(y),1); %Hilfsvariable
X=[xh x0 x1 x2 x3 x4 x5 x6];
beta=inv(X'*X)*X'*y;
%beta0:    21.7236
%beta1:    0.3407
%beta2:   -0.3373
%beta3:    0.0271
%beta4:    0.0058
%beta5:    0.0663
%beta6:   -0.2979
%beta7:   -0.2111

%g)
SST=sum((y-mean(y)).^2) % 6.0328e+03 
%SST:Sum of Squares for Total
%SSE:Sum of Squares for Error
%SSReg:Sum of Squares for Regression
%SST=SSE+SSReg
%R^2=SSReg/SST
X=[xh x0 x1];
beta = inv(X'*X)*X'*y
%    4.0216
%    0.3710
%    1.6266
y_h=beta(1)+beta(2)*x0+beta(3)*x1;
SSE=sum((y-y_h).^2)     % 1.4869e+03
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.7535

X=[xh x0 x1 x2];
beta = inv(X'*X)*X'*y
%    2.1207
%    0.4725
%    0.7164
%    0.0405
y_h=beta(1)+beta(2)*x0+beta(3)*x1+beta(4)*x2;
SSE=sum((y-y_h).^2)     % 1.2956e+03
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.7852

X=[xh x0 x1 x2 x3];
beta = inv(X'*X)*X'*y
%   -0.7743
%    0.3344
%   -0.0909
%    0.0330
%    0.0052
y_h=beta(1)+beta(2)*x0+beta(3)*x1+beta(4)*x2+ beta(5)*x3;
SSE=sum((y-y_h).^2)     %  1.0993e+03
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.8178

X=[xh x0 x1 x2 x3 x4];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x0+beta(3)*x1+beta(4)*x2+ beta(5)*x3+beta(6)*x4;
SSE=sum((y-y_h).^2)     %  1.0829e+03
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST %  0.8205

X=[xh x0 x1 x2 x3 x4 x5];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x0+beta(3)*x1+beta(4)*x2+ beta(5)*x3+beta(6)*x4...
+beta(7)*x5;
SSE=sum((y-y_h).^2)     %  697.7442
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST %  0.8843

X=[xh x0 x1 x2 x3 x4 x5 x6];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x0+beta(3)*x1+beta(4)*x2+ beta(5)*x3+beta(6)*x4...
+beta(7)*x5+beta(8)*x6;
SSE=sum((y-y_h).^2)     %  691.1813
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST %  0.8854

%mit mehrere Eingangsvariablen, steigt sich R2, neigt sich nach 1
%mehrere EVn --> bessere Vorhersage

%h)
X=[xh x0];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x0;
SSE=sum((y-y_h).^2)
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.7049

X=[xh x1];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x1;
SSE=sum((y-y_h).^2)   
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.7510

X=[xh x2];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x2;
SSE=sum((y-y_h).^2)   
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.7271

X=[xh x3];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x3;
SSE=sum((y-y_h).^2)    
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.7838

X=[xh x4];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x4;
SSE=sum((y-y_h).^2)   
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.2084

X=[xh x5];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x5;
SSE=sum((y-y_h).^2)  
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.3117

X=[xh x6];
beta = inv(X'*X)*X'*y
y_h=beta(1)+beta(2)*x6;
SSE=sum((y-y_h).^2)  
SSR=sum((y_h-mean(y)).^2)
R2=SSR/SST % 0.2820
%) die Beste x3(Gewicht) 0.7838

%Aufgabe D3: Gradientenverfahren
%a)
data= xlsread('E:\ML\MATLAB\?bung_1\Autos_DE.xlsx');
%b)
y=data(:,[1]);
x3=data(:,[4]); %Leistung
x0=ones(length(y),1);
X=[x0 x3];
%c)
R=X'*X/length(X)  %Korrelationsmatrix
lambda=eig(R)  %Eigenwerte
lambda=lambda(2)
%d)
x3_skal=(x3-mean(x3))/sqrt(var(x3));
X_skal=[x0 x3_skal];
R=X_skal'*X_skal/length(X_skal)
lambda=eig(R)
lambda=lambda(2)

beta=0;
a=1/lambda;
delta_C =(X_skal'*X_skal*beta-X_skal'*y)/length(y);

