%Aufgabe D3
%a)
a=linspace(0,10,11);
b=a';
M=b*a;
M(1,:)=a;
M(:,1)=b;
M(1)=NaN

%b)
V5=M(6,2:end)

%c)
M5=M([1;6],:)