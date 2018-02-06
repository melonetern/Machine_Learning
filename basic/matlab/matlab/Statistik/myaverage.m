function ave = myaverage(x,N)
sizex = size(x);
sizeN = size(N);
if sizex(2) ~= sizeN(2)
disp('错误：数据必须具有相同的维数。')
else
total = sum(N);
s = x.*N;
ave = sum(s)/total;
end