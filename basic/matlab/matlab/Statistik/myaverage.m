function ave = myaverage(x,N)
sizex = size(x);
sizeN = size(N);
if sizex(2) ~= sizeN(2)
disp('�������ݱ��������ͬ��ά����')
else
total = sum(N);
s = x.*N;
ave = sum(s)/total;
end